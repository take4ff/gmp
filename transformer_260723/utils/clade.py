# --- utils/clade.py ---
# 提案4: 主要系統クレード埋め込み（分布シフト対策）用の系統名→クレードID写像。
#
# strain 名（Pango 系統, 例: 'AY.44', 'BA.1.1', 'BQ.1', 'XBB.1.5', 'JN.1'）を
# WHO/Nextstrain の主要クレード群へ写像する。Pango のエイリアス空間は膨大なため
# 完全網羅は目的とせず、流行の中心を占めた主要系統をプレフィックス規則で拾う。
# 未知・非該当は clade_id=0（Embedding の padding_idx＝ゼロベクトル）に落とす。
#
# 使い方:
#   from ..utils.clade import lineage_to_clade_id, NUM_CLADES, CLADE_NAMES
#   cid = lineage_to_clade_id('BA.5.2')   # -> 8
#
# model.py は NUM_CLADES を埋め込みサイズに、train/evaluate は
# lineage_to_clade_id をバッチ strain の写像に用いる。

# clade_id=0 は「unknown / other」（padding, ゼロベクトル）に予約する。
CLADE_NAMES = [
    'unknown',   # 0
    'Wuhan',     # 1  祖先型 A / B / B.1 系（Alpha・Delta の直接祖先を除く一般 B.1）
    'Alpha',     # 2  B.1.1.7 / Q.*
    'Beta',      # 3  B.1.351
    'Gamma',     # 4  P.1
    'Delta',     # 5  B.1.617.2 / AY.*
    'BA.1',      # 6  Omicron BA.1*
    'BA.2',      # 7  Omicron BA.2*（BA.2.86/JN 系を除く）
    'BA.4-5',    # 8  Omicron BA.4* / BA.5* / BQ.* 等
    'XBB',       # 9  XBB* および XBB 由来（EG/FL/HK/HV 等）
    'JN',        # 10 BA.2.86 / JN.* / KP.* / LB.* 等
]
NUM_CLADES = len(CLADE_NAMES)

_NAME_TO_ID = {name: i for i, name in enumerate(CLADE_NAMES)}


def _tokens(lineage):
    """'BA.5.2' -> ['BA', '5', '2']。区切りは '.'。"""
    return lineage.split('.')


# プレフィックス（ドット区切りトークン列）→ clade 名。長いプレフィックスを優先評価する。
# 例: ('BA', '2', '86') は ('BA', '2') より先に判定する必要があるため、
# 評価時にトークン長の降順でソートして最長一致を採る。
_PREFIX_RULES = {
    # --- Delta ---
    ('AY',):            'Delta',
    ('B', '1', '617', '2'): 'Delta',
    # --- Alpha ---
    ('B', '1', '1', '7'):   'Alpha',
    ('Q',):             'Alpha',
    # --- Beta / Gamma ---
    ('B', '1', '351'):  'Beta',
    ('P', '1'):         'Gamma',
    # --- Omicron BA.2.86 / JN 系（BA.2 より先に最長一致で拾う）---
    ('BA', '2', '86'):  'JN',
    ('JN',):            'JN',
    ('KP',):            'JN',
    ('LB',):            'JN',
    ('LP',):            'JN',
    # --- Omicron XBB 系（組換え・派生を含む）---
    ('XBB',):           'XBB',
    ('EG',):            'XBB',
    ('FL',):            'XBB',
    ('FE',):            'XBB',
    ('GK',):            'XBB',
    ('HK',):            'XBB',
    ('HV',):            'XBB',
    ('JD',):            'XBB',
    ('JG',):            'XBB',
    # --- Omicron BA.4 / BA.5 / BQ 系 ---
    ('BA', '4'):        'BA.4-5',
    ('BA', '5'):        'BA.4-5',
    ('BE',):            'BA.4-5',
    ('BF',):            'BA.4-5',
    ('BQ',):            'BA.4-5',
    ('CE',):            'BA.4-5',
    ('DL',):            'BA.4-5',
    # --- Omicron BA.2 系（BA.2.86/JN を除く一般 BA.2 派生）---
    ('BA', '2'):        'BA.2',
    ('BJ',):            'BA.2',
    ('BM',):            'BA.2',
    ('BN',):            'BA.2',
    ('BR',):            'BA.2',
    ('BS',):            'BA.2',
    ('CH',):            'BA.2',
    ('CM',):            'BA.2',
    ('DV',):            'BA.2',
    # --- Omicron BA.1 系 ---
    ('BA', '1'):        'BA.1',
    ('BC',):            'BA.1',
    ('BD',):            'BA.1',
    # --- 祖先型 Wuhan / 一般 B 系（最短プレフィックスなので最後に評価される）---
    ('A',):             'Wuhan',
    ('B',):             'Wuhan',
}

# 最長一致を保証するため、トークン長の降順でルールを並べておく。
_SORTED_RULES = sorted(_PREFIX_RULES.items(), key=lambda kv: len(kv[0]), reverse=True)


def lineage_to_clade_id(lineage):
    """Pango 系統名を主要クレード ID（0〜NUM_CLADES-1）へ写像する。

    未知・非該当・空文字は 0（unknown）を返す。
    """
    if not lineage or not isinstance(lineage, str):
        return 0
    toks = _tokens(lineage.strip())
    if not toks or not toks[0]:
        return 0
    for prefix, name in _SORTED_RULES:
        n = len(prefix)
        if len(toks) >= n and tuple(toks[:n]) == prefix:
            return _NAME_TO_ID[name]
    return 0


def clade_ids_tensor(strains, device=None):
    """strain 名のリストを clade_id の LongTensor [B] に変換する。"""
    import torch
    ids = [lineage_to_clade_id(s) for s in strains]
    t = torch.tensor(ids, dtype=torch.long)
    return t.to(device) if device is not None else t
