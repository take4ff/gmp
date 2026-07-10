# --- petra/eval/weights.py ---
"""PETRA の representativeness 重みを移植する（Weighted Recall 用）。

PETRA (petra/PETra/data_preprocess/process_usher.py) の実装から確認した式:
    weight(country, yymm) = proc( population(country) / seq_count(country, yymm) )
    proc(x): x<1e5 → x ; x>1e7 → 1e6 ; それ以外 → 1e5 * sqrt(x/1e5)

representativeness = 「その国・月の配列が人口比で見て多いか少ないか」の逆数（sqrt で平滑化）。
少ししか配列を出していない国・月のサンプルほど重みが大きくなる（過小代表の補正）。

country は DB 再構築で samples.country が入って初めて使える（現行の古い DB には無い）。
本体のデータローダ／collate_fn は変更しない。既存 batch の
  batch[3] = strain 名, batch[9] = collection_date
だけを使い、系統→国の多数決マッピングを DB へ1回問い合わせて突き合わせる。
"""

import os
import re
from collections import defaultdict

# PETRA (process_usher.py replace_keys) から移植した国名の表記ゆれ補正
_REPLACE_KEYS = {
    'changzhou': 'jiangsu', 'guangzhou': 'guangdong', 'hangzhou': 'zhejiang', 'pingxiang': 'jiangxi',
    'shangrao': 'jiangxi', 'shaoxing': 'zhejiang', 'weifang': 'shandong', 'yingtan': 'jiangxi',
    'harbin': 'heilongjiang', 'jian': 'jiangxi', 'jiujiang': 'jiangxi', 'changde': 'hunan',
    'lishui': 'zhejiang', 'foshan': 'guangdong', 'jining': 'shandong', 'xinyu': 'jiangxi', 'nanchang': 'jiangxi',
    'fuzhou': 'fujian', 'yichun': 'jiangxi', 'tianmen': 'hubei', 'kashgar': 'xinjiang',
    'cotedivoirecotedivoire': 'cotedivoire', 'chinay': 'china', 'brasil': 'brazil', 'mexicomex': 'mexico',
    'urumqi': 'xinjiang', 'luan': 'anhui', 'chilema': 'chile', 'shulan': 'jilin', 'taly': 'italy',
    'cotedivoireafrica': 'cotedivoire', 'gd': 'guangdong', 'tianjn': 'tianjin', 'ialy': 'italy', 'spaiin': 'spain',
    'fance': 'france', 'romnaia': 'romania', 'lka': 'srilanka', 'wuhan': 'hubei', 'shenzhen': 'guangdong',
    'jingzhou': 'hubei', 'ganzhou': 'jiangxi', 'mauritanie': 'mauritania', 'cameroun': 'cameroon', 'us': 'usa',
    'spainandplub': 'saotomeandprincipe', 'mex': 'mexico', 'qingdao': 'shandong', 'saudi': 'saudiarabia',
    'botswna': 'botswana', 'dji': 'djibouti', 'pdl': 'portugal', 'zambai': 'zambia', 'saintmartin': 'saintmarten',
    'congo': 'republicofthecongo', 'africa': 'southafrica', 'dom': 'dominicanrepublic',
    'drcongo': 'democraticrepublicofthecongo',
    'andorre': 'andorra', 'afganistan': 'afghanistan', 'unitedkingom': 'unitedkingdom', 'guyane': 'guyana',
    'tahiti': 'frenchpolynesia', 'westbank': 'palestine', 'macedonia': 'northmacedonia',
    'antigua': 'antiguaandbarbuda',
    'austraila': 'australia', 'erbil': 'iraq', 'macau': 'macao', 'easttimor': 'timorleste',
    'turksandcaicos': 'turksandcaicosislands',
    'italia': 'italy', 'newcaledonie': 'newcaledonia', 'tibet': 'xizang', 'argentino': 'argentina',
    'brazi': 'brazil',
    'spin': 'spain', 'mexicoi': 'mexico', 'wallisetfutuna': 'newcaledonia', "china//guangxi": 'guangxi',
    'jersey': 'england',
    'korea': 'southkorea',
}

DEFAULT_POPULATION_CSV = os.path.join('petra', 'PETra', 'data_preprocess', 'country_population.txt')


def normalize_country_name(name):
    """PETRA と同じ正規化: 空白除去・&→and・小文字化・表記ゆれ補正。"""
    if not name:
        return ''
    n = str(name).replace(' ', '').replace('&', 'and').lower()
    return _REPLACE_KEYS.get(n, n)


def parse_country_population(path=None):
    """country_population.txt（Worldometer 形式）から {正規化国名: 人口} を作る。"""
    path = path or DEFAULT_POPULATION_CSV
    pop_dic = {}
    if not os.path.exists(path):
        return pop_dic
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            cols = line.rstrip('\r\n').split('\t')
            if len(cols) < 2:
                continue
            # 先頭が数字(順位)なら国名は2列目、そうでなければ1列目
            if cols[0][:1].isdigit():
                name, pop = cols[1], cols[2] if len(cols) > 2 else None
            else:
                name, pop = cols[0].split('[')[0], cols[1] if len(cols) > 1 else None
            if not pop:
                continue
            try:
                pop_val = int(pop.replace(',', '').strip())
            except ValueError:
                continue
            pop_dic[normalize_country_name(name)] = pop_val
    return pop_dic


def proc(x):
    """PETRA の重み平滑化・クランプ（process_usher.py の proc()）。"""
    import math
    if x < 100_000:
        return x
    if x > 10_000_000:
        return 1_000_000.0
    return 100_000 * math.sqrt(x / 100_000)


def compute_representativeness_weight(country, yymm, pop_dic, count_dic):
    """weight(country, yymm) = proc(population / count)。データ欠損時は 1.0（中立）。"""
    c = normalize_country_name(country)
    pop = pop_dic.get(c)
    cnt = count_dic.get(c, {}).get(yymm)
    if not pop or not cnt:
        return 1.0
    return proc(pop / cnt)


def get_strain_country_map(db_path, split_type):
    """DB の指定 split から 系統(strain)→多数決国名 のマッピングを作る。

    samples.country が無い（未再構築の古い DB）場合は空 dict を返し、呼び出し側は
    重み=1.0（Average と同一）にフォールバックする。
    """
    from transformer_260707.db.connection import connect_db
    from transformer_260707.db.queries import get_split_col
    from transformer_260707 import config

    con = connect_db(db_path, read_only=True)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info('samples')").fetchall()]
        if 'country' not in cols:
            return {}, {}
        sn_col = 'strain_name_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strain_name_ncbi'
        split_col = get_split_col()
        rows = con.execute(f"""
            SELECT st.{sn_col}, s.country, s.collection_date
            FROM samples s JOIN strains st ON s.strain_id = st.strain_id
            WHERE s.{split_col} = ? AND s.country IS NOT NULL AND s.country != ''
        """, [split_type]).fetchall()
    finally:
        con.close()

    strain_country_votes = defaultdict(lambda: defaultdict(int))
    count_dic = defaultdict(lambda: defaultdict(int))
    for strain, country, date in rows:
        c = normalize_country_name(country)
        strain_country_votes[strain][c] += 1
        yymm = _to_yymm(date)
        if yymm:
            count_dic[c][yymm] += 1

    strain_country = {s: max(votes, key=votes.get) for s, votes in strain_country_votes.items()}
    return strain_country, {k: dict(v) for k, v in count_dic.items()}


def _to_yymm(date_str):
    """'YYYY-MM-DD' → 'YYYYMM' 形式（PETRA の yymm キーに合わせる）。"""
    if not date_str:
        return None
    m = re.match(r'(\d{4})-(\d{2})', str(date_str))
    if not m:
        return None
    return m.group(1) + m.group(2)
