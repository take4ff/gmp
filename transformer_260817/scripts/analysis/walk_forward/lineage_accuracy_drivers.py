# --- transformer_260817/scripts/analysis/walk_forward/lineage_accuracy_drivers.py ---
"""系統別の位置予測精度(position_hit_rate_pct)を左右する要因を、単純相関だけでなく
多変量回帰で統制した上で切り分ける（walk-forward全fold横断）。

背景: 既存の可視化（test_labeldiversity_vs_accuracy.png 等）でラベル多様度(entropy_norm)と
精度の強い負の相関が確認できている一方、「サンプル数が多いのに精度がほぼ0%」という
反例（fold2のBA.*系統）が見つかり、多様度と件数(n)が交絡している疑いがある。
本スクリプトは以下5要因を系統(fold, lineage)粒度でプールし、相関行列・散布図に加えて
多変量OLS/WLS回帰で「他要因を統制してもnは独立に効くか」に答える。

  1. 多様度       entropy_norm      … 既存 test_lineage_metrics.csv の db_target_entropy_norm
  2. 件数         n                 … 既存 test_lineage_metrics.csv の num_samples
  3. 系統距離     phylo_dist        … analyze_lineage_difficulty.py の alias文字列LCA距離を
                                       このfold固有のtrain窓の系統集合に対して計算し直す
  4. recency      recency_months    … このfold test窓での採取日中央値 - fold split_date（月換算）
  5. founder効果  frac_branching    … input_path_str（raw_pathの末尾ステップを除いた履歴）を
                                       共有するサンプル数K（K>=branching_threshold で「分岐」扱い）
                                       の系統内での割合、および mean_K・mean_M（共起数）

推論・チェックポイント読み込みは一切行わない（1,2は既存CSVを再利用、3-5は読み取り専用DB
クエリのみ）。`split_type_wf` 等の共有DB列には触れないため、他の walk_forward/学習/XAI
プロセスと並行実行して安全（petra/eval/eval_tail_by_daterange.py 等と同じ、日付範囲を直接
指定する方式）。

DB母集団の方針: recency・founder(K/M)の集計は、test_lineage_metrics.csv の元になった
実評価母集団（max_cooccurrence <= config.MAX_CO_OCCURRENCE、EVAL_MAX_Y_CO_OCCURRENCE超過の
除外）と揃える（ユーザー確認済み）。生DBそのままだと「同じサンプル集合について複数要因を
測っている」という前提が崩れるため。

Usage:
  python -m transformer_260817.scripts.analysis.walk_forward.lineage_accuracy_drivers \\
      --walk_forward_dir outputs/transformer_260817/results/walk_forward/<timestamp>
  # 一部foldのみ: --folds 1 3
  # 分岐しきい値変更: --branching_threshold 3
"""
import argparse
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from transformer_260817 import config
from transformer_260817.utils.logging import force_print
from transformer_260817.db.connection import get_db_path, connect_db
from transformer_260817.scripts.analysis.xai import _xai_common as X
from transformer_260817.scripts.analysis.analyze_lineage_difficulty import (
    load_alias, expand, compute_min_phylo_dist, ALIAS_KEY_PATH,
)

STRAIN_COL = 'strain_name_usher' if getattr(config, 'STRENGTH_SOURCE', 'ncbi') == 'usher' else 'strain_name_ncbi'

FACTOR_COLS = ['entropy_norm', 'log1p_n', 'norm_phylo_dist', 'recency_months', 'frac_branching']
FACTOR_LABELS = {
    'entropy_norm':     'Target-position entropy (normalized)',
    'log1p_n':          'log1p(sample count)',
    'norm_phylo_dist':  'Normalized phylo distance to train',
    'recency_months':   'Recency (months since fold train cutoff)',
    'frac_branching':   'Fraction of samples in branching (K>=thr) groups',
}


# ------------------------------------------------------------------ #
# DB クエリ（フィルタ付き、date-range直接指定、split_type_wf非依存）
# ------------------------------------------------------------------ #
def _train_lineages_for_fold(con, train_start, split_date):
    """このfoldのtrain窓（[train_start, split_date)）で観測された系統集合を返す。"""
    where = f"st.{STRAIN_COL} IS NOT NULL AND st.{STRAIN_COL} != 'unclassifiable' "
    where += "AND s.collection_date IS NOT NULL AND s.collection_date != '' "
    params = []
    if train_start is not None:
        where += "AND RPAD(s.collection_date, 10, '-01-01') >= ? "
        params.append(train_start)
    where += "AND RPAD(s.collection_date, 10, '-01-01') < ? "
    params.append(split_date)
    where += "AND s.max_cooccurrence <= ?"
    params.append(config.MAX_CO_OCCURRENCE)

    rows = con.execute(f"""
        SELECT DISTINCT st.{STRAIN_COL}
        FROM samples s JOIN strains st ON s.strain_id = st.strain_id
        WHERE {where}
    """, params).fetchall()
    return {r[0] for r in rows}


def _test_window_rows(con, split_date, split_end):
    """このfoldのtest窓（[split_date, split_end)）の (strain, raw_path, collection_date, targets_blob) を返す。

    recency・branching(K/M)の両方をこの1クエリで賄う。EVAL_MAX_Y_CO_OCCURRENCE超過の除外は
    呼び出し側で pickle.loads(targets_blob) の長さを見て行う（db/dataset.py::_get_sample_ids と同じロジック）。
    """
    where = f"st.{STRAIN_COL} IS NOT NULL AND st.{STRAIN_COL} != 'unclassifiable' "
    where += "AND s.collection_date IS NOT NULL AND s.collection_date != '' "
    where += "AND RPAD(s.collection_date, 10, '-01-01') >= ? "
    params = [split_date]
    if split_end is not None:
        where += "AND RPAD(s.collection_date, 10, '-01-01') < ? "
        params.append(split_end)
    where += "AND s.max_cooccurrence <= ?"
    params.append(config.MAX_CO_OCCURRENCE)

    rows = con.execute(f"""
        SELECT st.{STRAIN_COL}, s.raw_path, s.collection_date, l.targets
        FROM samples s
        JOIN strains st ON s.strain_id = st.strain_id
        JOIN labels  l  ON s.sample_id = l.sample_id
        WHERE {where}
    """, params).fetchall()

    eval_max_y_co = getattr(config, 'EVAL_MAX_Y_CO_OCCURRENCE', None)
    kept = []
    n_excluded = 0
    for strain, raw_path, collection_date, targets_blob in rows:
        if eval_max_y_co is not None:
            targets = pickle.loads(targets_blob)
            if len(targets) > eval_max_y_co:
                n_excluded += 1
                continue
        kept.append((strain, raw_path, collection_date))
    force_print(f"[INFO]   test窓: 生{len(rows):,}件 -> Y共起除外{n_excluded:,}件 -> {len(kept):,}件")
    return kept


# ------------------------------------------------------------------ #
# 要因別の系統別集計
# ------------------------------------------------------------------ #
def _phylo_dist_per_lineage(test_lineages, train_lineages, alias_key):
    train_fulls = [expand(l, alias_key) for l in train_lineages]
    test_fulls_map = {l: expand(l, alias_key) for l in test_lineages}
    dist_map = compute_min_phylo_dist(list(test_fulls_map.values()), train_fulls)
    return {l: dist_map[test_fulls_map[l]] for l in test_lineages}


def _recency_per_lineage(rows, split_date):
    """系統ごとに採取日の中央値を求め、fold split_dateからの経過月数を返す。"""
    dates_by_lineage = defaultdict(list)
    for strain, _raw_path, collection_date in rows:
        d = _normalize_date(collection_date)
        if d is not None:
            dates_by_lineage[strain].append(d)

    split_ts = pd.Timestamp(split_date)
    out = {}
    for lineage, dates in dates_by_lineage.items():
        median_date = pd.Series(dates).median()
        out[lineage] = (median_date - split_ts).days / 30.44
    return out


def _normalize_date(d):
    if not d or not isinstance(d, str):
        return None
    d = d.strip()
    if len(d) == 7:
        d = d + '-01'
    try:
        return pd.Timestamp(d)
    except Exception:
        return None


def _branching_per_lineage(rows, threshold):
    """input_path_str（raw_pathの末尾ステップを除いた履歴）の窓全体でのグループサイズKと
    最終ステップの共起数Mを、系統ごとに frac_branching/mean_K/mean_M へ集計する。
    """
    group_size = Counter()
    parsed = []
    for strain, raw_path, _cdate in rows:
        if not raw_path:
            continue
        parts = raw_path.split('>')
        input_path_str = '>'.join(parts[:-1]) if len(parts) > 1 else ''
        m = len(parts[-1].split(','))
        group_size[input_path_str] += 1
        parsed.append((strain, input_path_str, m))

    acc = defaultdict(lambda: {'n': 0, 'branch_hits': 0, 'k_sum': 0, 'm_sum': 0})
    for strain, input_path_str, m in parsed:
        k = group_size[input_path_str]
        a = acc[strain]
        a['n'] += 1
        a['branch_hits'] += int(k >= threshold)
        a['k_sum'] += k
        a['m_sum'] += m

    out = {}
    for lineage, a in acc.items():
        n = a['n']
        out[lineage] = {
            'frac_branching': a['branch_hits'] / n if n else float('nan'),
            'mean_K': a['k_sum'] / n if n else float('nan'),
            'mean_M': a['m_sum'] / n if n else float('nan'),
        }
    return out


# ------------------------------------------------------------------ #
# fold単位のテーブル構築
# ------------------------------------------------------------------ #
def _build_fold_table(fold_id, era, metrics_csv_path, con, alias_key, fold_windows,
                      min_samples, branching_threshold):
    try:
        df = pd.read_csv(metrics_csv_path)
    except Exception as e:
        force_print(f"[WARNING] fold{fold_id}: {metrics_csv_path} 読み込み失敗 ({e})、スキップ")
        return None

    df = df.rename(columns={
        'num_samples': 'n',
        'db_target_entropy_norm': 'entropy_norm',
        'position_hit_rate_pct': 'hit_rate',
    })
    df = df[df['n'] >= min_samples].copy()
    if df.empty:
        force_print(f"[WARNING] fold{fold_id}: n>={min_samples} を満たす系統がありません、スキップ")
        return None

    train_start, split_date, split_end = fold_windows[fold_id]
    force_print(f"[INFO] fold{fold_id} ({era}): train窓 train_start={train_start}, "
                f"test窓 [{split_date}, {split_end})")

    train_lineages = _train_lineages_for_fold(con, train_start, split_date)
    test_rows = _test_window_rows(con, split_date, split_end)

    test_lineages = df['lineage'].tolist()
    phylo_dist = _phylo_dist_per_lineage(test_lineages, train_lineages, alias_key)
    recency = _recency_per_lineage(test_rows, split_date)
    branching = _branching_per_lineage(test_rows, branching_threshold)

    df['phylo_dist'] = df['lineage'].map(phylo_dist)
    max_dist = df['phylo_dist'].max()
    df['norm_phylo_dist'] = df['phylo_dist'] / max_dist if max_dist and max_dist > 0 else 0.0
    df['recency_months'] = df['lineage'].map(recency)
    df['frac_branching'] = df['lineage'].map(lambda l: branching.get(l, {}).get('frac_branching'))
    df['mean_K'] = df['lineage'].map(lambda l: branching.get(l, {}).get('mean_K'))
    df['mean_M'] = df['lineage'].map(lambda l: branching.get(l, {}).get('mean_M'))
    df['log1p_n'] = np.log1p(df['n'])
    df['fold'] = fold_id
    df['era'] = era

    n_missing = df[['phylo_dist', 'recency_months', 'frac_branching']].isna().any(axis=1).sum()
    if n_missing:
        force_print(f"[WARNING] fold{fold_id}: {n_missing}系統でjoin欠損（test窓DBクエリ結果と"
                    f"metrics CSVの系統名が一致しなかった可能性）")

    keep_cols = ['fold', 'era', 'lineage', 'n', 'hit_rate', 'entropy_norm', 'log1p_n',
                'phylo_dist', 'norm_phylo_dist', 'recency_months',
                'frac_branching', 'mean_K', 'mean_M']
    return df[keep_cols], n_missing


# ------------------------------------------------------------------ #
# 相関・散布図・回帰
# ------------------------------------------------------------------ #
def _correlation_outputs(pooled, out_dir):
    cols = ['hit_rate'] + FACTOR_COLS
    sub = pooled[cols].dropna()

    pearson = sub.corr(method='pearson')
    spearman = sub.corr(method='spearman')
    X.save_csv(pearson.reset_index().rename(columns={'index': 'factor'}), out_dir,
              'correlation_matrix_pearson.csv')
    X.save_csv(spearman.reset_index().rename(columns={'index': 'factor'}), out_dir,
              'correlation_matrix_spearman.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title in zip(axes, [pearson, spearman], ['Pearson', 'Spearman']):
        im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f'{mat.values[i, j]:.2f}', ha='center', va='center', fontsize=7)
        ax.set_title(f'{title} correlation (n={len(sub)})')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'correlation_heatmap.png')
    return pearson, spearman


def _scatter_panels(pooled, out_dir):
    fig, axes = plt.subplots(1, len(FACTOR_COLS), figsize=(5 * len(FACTOR_COLS), 5))
    folds_sorted = sorted(pooled['fold'].unique())
    cmap = plt.get_cmap('viridis', max(len(folds_sorted), 2))
    fold_color = {f: cmap(i) for i, f in enumerate(folds_sorted)}

    for ax, col in zip(axes, FACTOR_COLS):
        sub = pooled[[col, 'hit_rate', 'n', 'fold']].dropna()
        for fid in folds_sorted:
            s = sub[sub['fold'] == fid]
            if s.empty:
                continue
            ax.scatter(s[col], s['hit_rate'], s=np.log1p(s['n']) * 6,
                      alpha=0.6, edgecolors='none', color=fold_color[fid], label=f'fold{fid}')
        ax.set_xlabel(FACTOR_LABELS[col])
        ax.set_ylabel('Position hit rate (%)')
        ax.grid(linestyle='--', alpha=0.3)
        try:
            r, p = stats.pearsonr(sub[col], sub['hit_rate'])
            z = np.polyfit(sub[col], sub['hit_rate'], 1)
            xline = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(xline, np.polyval(z, xline), 'r--', lw=1.2)
            ax.set_title(f'r={r:.3f}, p={p:.2e}, n={len(sub)}', fontsize=9)
        except Exception:
            ax.set_title(f'r=N/A, n={len(sub)}', fontsize=9)
    axes[0].legend(fontsize=6, loc='best')
    plt.suptitle('Lineage-level accuracy drivers (pooled across folds)', fontsize=12)
    plt.tight_layout()
    X.save_fig(fig, out_dir, 'drivers_scatter_panels.png')


def _ols_regression(pooled, out_dir, weighted):
    sub = pooled[['hit_rate', 'n'] + FACTOR_COLS].dropna().copy()
    z_cols = []
    for col in FACTOR_COLS:
        z_col = f'z_{col}'
        std = sub[col].std()
        sub[z_col] = (sub[col] - sub[col].mean()) / std if std > 0 else 0.0
        z_cols.append(z_col)

    X_mat = sm.add_constant(sub[z_cols])
    y = sub['hit_rate']

    results = {}
    for label, model in [
        ('ols', sm.OLS(y, X_mat).fit()),
        ('wls', sm.WLS(y, X_mat, weights=sub['n']).fit()),
    ]:
        results[label] = model

    summary_text = ''
    coefficients = {}
    for label, model in results.items():
        summary_text += f'\n{"="*70}\n{label.upper()} (weighted={label=="wls"})\n{"="*70}\n'
        summary_text += str(model.summary())
        coefficients[label] = {
            'r_squared': float(model.rsquared),
            'r_squared_adj': float(model.rsquared_adj),
            'n_obs': int(model.nobs),
            'terms': {
                term: {
                    'coef': float(model.params[term]),
                    'std_err': float(model.bse[term]),
                    'p_value': float(model.pvalues[term]),
                    'ci_low': float(model.conf_int().loc[term, 0]),
                    'ci_high': float(model.conf_int().loc[term, 1]),
                }
                for term in model.params.index
            },
        }

    # VIF (collinearity diagnostic) — computed once on the unweighted design matrix
    vif = {}
    for i, col in enumerate(z_cols):
        try:
            vif[col] = float(variance_inflation_factor(sub[z_cols].values, i))
        except Exception:
            vif[col] = float('nan')
    coefficients['vif'] = vif
    summary_text += f'\n\nVIF (collinearity): {vif}\n'

    with open(f'{out_dir}/ols_summary.txt', 'w') as f:
        f.write(summary_text)
    force_print(f"[INFO] Saved {out_dir}/ols_summary.txt")
    X.save_json(coefficients, out_dir, 'ols_coefficients.json')

    # 標準化係数を primary(weighted) として解釈を印字
    primary = results['wls'] if weighted else results['ols']
    force_print(f"\n[解釈サマリ] ({'WLS' if weighted else 'OLS'}, R^2={primary.rsquared:.3f}, n={int(primary.nobs)})")
    for col in FACTOR_COLS:
        term = f'z_{col}'
        coef = primary.params[term]
        p = primary.pvalues[term]
        sig = '有意 (p<0.05)' if p < 0.05 else '非有意'
        force_print(f"  {FACTOR_LABELS[col]:<48} 標準化係数={coef:+.3f}  p={p:.3e}  [{sig}]")
    n_term = primary.pvalues['z_log1p_n']
    verdict = ("他要因を統制しても件数(n)は有意に精度と関連している"
              if n_term < 0.05 else
              "他要因（特に多様度）を統制すると件数(n)の効果は有意でなくなる＝交絡の可能性が高い")
    force_print(f"  => 件数(n)についての結論: {verdict}")

    return coefficients


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--walk_forward_dir', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    ap.add_argument('--min_samples', type=int,
                    default=getattr(config, 'DIVERSITY_PLOT_MIN_SAMPLES', 10))
    ap.add_argument('--branching_threshold', type=int, default=2)
    ap.add_argument('--weighted', action='store_true', default=True,
                    help='解釈サマリでWLS(n重み付き)を主として使う（既定True）')
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    from transformer_260817.scripts.eval.walk_forward import FOLDS
    fold_desc = {f[0]: f[4] for f in FOLDS}
    fold_windows = X.get_fold_windows()

    fold_csvs = X.discover_fold_files(args.walk_forward_dir, f'{args.split}_lineage_metrics.csv', args.folds)
    if not fold_csvs:
        raise SystemExit(f"[ERROR] {args.split}_lineage_metrics.csv が見つかりません: {args.walk_forward_dir}")
    force_print(f"[INFO] 発見したフォールド: {sorted(fold_csvs)}")

    alias_key, _rev = load_alias(ALIAS_KEY_PATH)
    con = connect_db(get_db_path(), read_only=True)

    tables = []
    skipped = []
    join_miss = {}
    try:
        for fold_id in sorted(fold_csvs):
            era = fold_desc.get(fold_id, 'unknown')
            result = _build_fold_table(fold_id, era, fold_csvs[fold_id], con, alias_key,
                                       fold_windows, args.min_samples, args.branching_threshold)
            if result is None:
                skipped.append(fold_id)
                continue
            df, n_missing = result
            tables.append(df)
            join_miss[fold_id] = int(n_missing)
    finally:
        con.close()

    if not tables:
        raise SystemExit("[ERROR] 有効なfoldがありません。")

    pooled = pd.concat(tables, ignore_index=True)

    out_dir = X.make_output_dir('lineage_accuracy_drivers', args.output_dir)
    X.save_csv(pooled, out_dir, 'lineage_accuracy_drivers_pooled.csv')

    _correlation_outputs(pooled, out_dir)
    _scatter_panels(pooled, out_dir)
    _ols_regression(pooled, out_dir, weighted=args.weighted)

    meta = {
        'walk_forward_dir': args.walk_forward_dir,
        'split': args.split,
        'min_samples': args.min_samples,
        'branching_threshold': args.branching_threshold,
        'discovered_folds': sorted(fold_csvs),
        'skipped_folds': skipped,
        'per_fold_n_lineages': {int(fid): int((pooled['fold'] == fid).sum()) for fid in sorted(fold_csvs) if fid not in skipped},
        'per_fold_join_miss': join_miss,
        'n_rows_pooled': int(len(pooled)),
    }
    X.save_json(meta, out_dir, 'meta.json')
    force_print(f"\n[INFO] 完了。出力先: {out_dir}")


if __name__ == '__main__':
    main()
