# --- transformer_260817/scripts/visualization/_fold_annotate.py ---
# 月別チャート（x軸=年月の文字列リスト）に、scripts/eval/walk_forward.py の FOLDS 日付窓から
# 算出した fold 境界の点線 + fold番号/優勢系統eraラベルを重ね描きする共通ヘルパー。
# DB・torch 非依存（FOLDS の日付比較のみ）で、既存の可視化スクリプトに軽量に追加できる。


# fold1（walk-forward最初のtest窓、2021-01-01〜）より前の期間の呼称。
# monthly_top_variants.csv（scripts/analysis/aggregate_variants.py の出力）実測で
# B/A（原株）→ B.1/B.1.1 → B.1.177 → B.1.1.7（Alpha）へ優勢系統が遷移する期間。
FOLD0_DESC = 'Original strain / B.1 lineages era (pre walk-forward)'


def _month_to_fold(month_str):
    from transformer_260817.scripts.eval.walk_forward import FOLDS
    ymd = f"{month_str}-01"
    for fid, _train_start, split_date, split_end, desc in FOLDS:
        if split_date is not None and ymd < split_date:
            continue
        if split_end is not None and ymd >= split_end:
            continue
        return fid, desc
    # FOLDS のどの窓にも入らない = fold1 開始日より前（walk-forward対象外の学習専用期間）
    return 0, FOLD0_DESC


def add_fold_annotations(ax, months, fontsize=5.5, y=1.01):
    """ax（x軸=range(len(months))相当）に fold 境界線とラベルを追加する。

    fold の日付窓に入らない月（walk-forward test期間外の学習専用データ）はラベルを付けない。
    ラベルはタイトルの下、軸のすぐ上に1行で統一配置する。
    """
    fold_ids, fold_desc = [], {}
    for m in months:
        fid, desc = _month_to_fold(m)
        fold_ids.append(fid)
        if fid is not None:
            fold_desc[fid] = desc

    def _label(fid):
        desc = fold_desc.get(fid, '')
        return f'fold{fid}: {desc}' if desc else f'fold{fid}'

    prev = None
    seg_start = 0
    for i, fid in enumerate(fold_ids):
        if fid != prev:
            if i > 0:
                ax.axvline(i - 0.5, color='black', ls='--', lw=0.8)
            if prev is not None:
                mid = (seg_start + i - 1) / 2
                ax.text(mid, y, _label(prev), transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=fontsize, color='black')
            seg_start = i
        prev = fid
    if prev is not None:
        mid = (seg_start + len(fold_ids) - 1) / 2
        ax.text(mid, y, _label(prev), transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=fontsize, color='black')
