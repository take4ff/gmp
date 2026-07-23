# --- transformer_260723/db/growth_features.py ---
"""亜系統（Pango 上位2階層）の流行ダイナミクスから因果的な成長特徴を作る。

各サンプル (系統 L, 収集日 t) に対し、**t 以前のデータだけ**で計算した4値を付与する:
  0) log_count_recent : log(1 + 直近K週の件数合計)              … 規模の因果版
  1) growth_rate      : 直近K週の log(件数+1) の線形回帰の傾き     … 勢い（正=拡大/負=縮小）
  2) rel_growth_adv   : 直近K週の logit(シェア) の傾き            … 相対成長優位（logistic fitness）
  3) growth_accel     : growth_rate(w) − growth_rate(w-1)         … 加速/減速（ピークアウト検知）

「縮小期の件数」のような未来量は使わない（リーク防止）。位相判定は growth_rate の符号で代替する。
epidemic の件数は系列メタCSV（全配列の系統×週）から構築するのが望ましい。

このモジュールは純粋な集計のみで DB / torch に依存しないため単体テスト可能。
"""

import math
from collections import defaultdict

# エポック（1970-01-01, 木曜）からの通し週番号を出すための基準
_EPOCH_ORDINAL = None  # 遅延初期化


def week_index(date_str):
    """'YYYY-MM-DD'（や 'YYYY-MM'）→ 通し週番号(int)。パース不可なら None。"""
    if not date_str:
        return None
    s = str(date_str).strip()
    if not s or s.lower() in ('nan', 'none', 'unknown'):
        return None
    parts = s.split('-')
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        d = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    except (ValueError, IndexError):
        return None
    if not (1 <= m <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100):
        return None
    import datetime
    try:
        ordinal = datetime.date(y, m, d).toordinal()
    except ValueError:
        return None
    return ordinal // 7


def pango_group(lineage):
    """フル系統名を Pango 上位2階層に丸める（BA.2.86.1→BA.2）。ドット無しはそのまま。"""
    if not lineage:
        return 'unknown'
    s = str(lineage).strip()
    if not s or s.lower() in ('nan', 'none', 'unknown'):
        return 'unknown'
    parts = s.split('.')
    return '.'.join(parts[:2]) if len(parts) >= 2 else s


def _slope(xs, ys):
    """(xs, ys) の最小二乗回帰の傾き。点が2未満/分散0なら 0.0。"""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return num / denom


def build_growth_lut(records, window_weeks=4):
    """(lineage_raw, date_str) の列から (lineage_group, week) → (4値) の LUT を因果的に構築する。

    Args:
        records: iterable of (lineage_raw, date_str)
        window_weeks: 傾き計算に使う直近ウィンドウ幅 K（週）
    Returns:
        dict[(lineage_group, week)] -> (log_count_recent, growth_rate, rel_growth_adv, growth_accel)
    """
    K = max(2, int(window_weeks))

    # 1. 件数行列 n(L, w) と週合計 total(w)
    n_lw = defaultdict(int)          # (L, w) -> count
    total_w = defaultdict(int)       # w -> Σ_L count
    weeks_of = defaultdict(set)      # L -> {weeks}
    for lineage_raw, date_str in records:
        w = week_index(date_str)
        if w is None:
            continue
        L = pango_group(lineage_raw)
        n_lw[(L, w)] += 1
        total_w[w] += 1
        weeks_of[L].add(w)

    # 2. まず全 (L, w) の log_count_recent / growth_rate / rel_growth_adv を計算
    log_count = {}
    growth_rate = {}
    rel_adv = {}
    for L, weeks in weeks_of.items():
        for w in weeks:
            win = [w - k for k in range(K)]  # [w-K+1 .. w]（w 以前のみ＝因果的）
            cnts = [n_lw.get((L, ww), 0) for ww in win]
            log_count[(L, w)] = math.log1p(sum(cnts))
            # growth_rate: log(count+1) の傾き
            growth_rate[(L, w)] = _slope(win, [math.log1p(c) for c in cnts])
            # rel_growth_adv: logit(share) の傾き（total>0 の週のみ）
            xs, ys = [], []
            for ww in win:
                tot = total_w.get(ww, 0)
                if tot <= 0:
                    continue
                f = n_lw.get((L, ww), 0) / tot
                f = min(max(f, 1e-6), 1 - 1e-6)  # クリップ
                xs.append(ww)
                ys.append(math.log(f / (1 - f)))
            rel_adv[(L, w)] = _slope(xs, ys)

    # 3. growth_accel = growth_rate(w) - growth_rate(w-1)（w-1 が無ければ 0）
    lut = {}
    for (L, w), lc in log_count.items():
        gr = growth_rate[(L, w)]
        gr_prev = growth_rate.get((L, w - 1), 0.0)
        accel = gr - gr_prev
        lut[(L, w)] = (lc, gr, rel_adv[(L, w)], accel)
    return lut


# 特徴の数と順序（config.NUM_CHEM_FEATURES / _NUM_FEATURE_NAMES と整合させること）
N_GROWTH_FEATURES = 4
GROWTH_FEATURE_NAMES = [
    'lineage_log_count_recent',
    'lineage_growth_rate',
    'lineage_rel_growth_adv',
    'lineage_growth_accel',
]
_GROWTH_ZERO = (0.0,) * N_GROWTH_FEATURES


def growth_for(lut, lineage_raw, date_str):
    """(系統, 日付) → 成長4値タプル。LUT に無ければ 0 埋め（中立扱い）。"""
    w = week_index(date_str)
    if w is None:
        return _GROWTH_ZERO
    return lut.get((pango_group(lineage_raw), w), _GROWTH_ZERO)
