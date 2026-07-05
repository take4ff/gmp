# --- utils/plotting.py ---
# グラフ生成（学習曲線・メトリクス可視化・カテゴリ比較）

import os
from . import logging as _log


def _get_plot_path(output_dir, filename):
    """主力先を plots/ に振り分け"""
    target_dir = os.path.join(output_dir, 'plots')
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)


def plot_training_curve(training_log, output_dir):
    """学習曲線（train_loss / val_loss）をプロットして保存する。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not training_log:
            return

        epochs = [log['epoch'] for log in training_log]
        train_losses = [log['train_loss'] for log in training_log]
        val_losses = [log['val_loss'] for log in training_log]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Curve', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 最小値をマーク
        min_val_idx = val_losses.index(min(val_losses))
        ax.axvline(x=epochs[min_val_idx], color='green', linestyle='--', alpha=0.5,
                   label=f'Best epoch: {epochs[min_val_idx]}')
        ax.scatter([epochs[min_val_idx]], [val_losses[min_val_idx]], color='green', s=100, zorder=5)

        plt.tight_layout()
        path = _get_plot_path(output_dir, 'training_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        _log.force_print(f"[INFO] Training curve saved to {path}")

    except ImportError:
        _log.force_print("[WARNING] matplotlib not available, skipping training curve plot")


def plot_metrics_by_timestep(metrics_by_ts, output_dir, prefix="val"):
    """タイムステップ別のメトリクスをグラフ化する (Hit Rate, Precision, Recall, F1, MAE)。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        timesteps = sorted(metrics_by_ts.keys())

        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(f'{prefix.upper()} - Metrics by Timestep', fontsize=14)

        colors = {
            'region': '#2ecc71', 'position': '#3498db', 'aa_pos': '#9b59b6',
            'codon': '#e67e22', 'synonymous': '#e74c3c'
        }

        # 1. Hit Rate比較
        ax = axes[0, 0]
        ax.plot(timesteps, [metrics_by_ts[t]['region_hit_rate'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_hit_rate'] for t in timesteps], 's-', label='Base Pos', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['aa_pos_hit_rate'] for t in timesteps], '^-', label='AA Pos', color=colors['aa_pos'])
        ax.plot(timesteps, [metrics_by_ts[t].get('codon_pos_hit_rate', 0) for t in timesteps], 'D-', label='Codon Pos', color=colors['codon'])
        ax.plot(timesteps, [metrics_by_ts[t].get('synonymous_hit_rate', 0) for t in timesteps], 'x-', label='Synonymous', color=colors['synonymous'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Hit Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Precision (適合率)
        ax = axes[0, 1]
        ax.plot(timesteps, [metrics_by_ts[t]['region_precision'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_precision'] for t in timesteps], 's-', label='Base Pos', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['aa_pos_precision'] for t in timesteps], '^-', label='AA Pos', color=colors['aa_pos'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Precision (%)')
        ax.set_title('Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Recall (再現率)
        ax = axes[1, 0]
        ax.plot(timesteps, [metrics_by_ts[t]['region_recall'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_recall'] for t in timesteps], 's-', label='Base Pos', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['aa_pos_recall'] for t in timesteps], '^-', label='AA Pos', color=colors['aa_pos'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Recall (%)')
        ax.set_title('Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. F1 Score
        ax = axes[1, 1]
        ax.plot(timesteps, [metrics_by_ts[t]['region_f1'] for t in timesteps], 'o-', label='Region', color=colors['region'])
        ax.plot(timesteps, [metrics_by_ts[t]['position_f1'] for t in timesteps], 's-', label='Base Pos', color=colors['position'])
        ax.plot(timesteps, [metrics_by_ts[t]['aa_pos_f1'] for t in timesteps], '^-', label='AA Pos', color=colors['aa_pos'])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('F1 Score (%)')
        ax.set_title('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Strength MAE
        ax = axes[2, 0]
        ax.plot(timesteps, [metrics_by_ts[t]['strength_mae'] for t in timesteps], 'D-', color='#e74c3c', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('MAE')
        ax.set_title('Strength Prediction MAE')
        ax.grid(True, alpha=0.3)

        # 6. サンプル数
        ax = axes[2, 1]
        ax.bar(timesteps, [metrics_by_ts[t]['num_samples'] for t in timesteps], color='#34495e', alpha=0.7)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Count')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = _get_plot_path(output_dir, f'{prefix}_metrics_plot.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        _log.force_print(f"[INFO] Metrics plot saved to {path}")

    except ImportError:
        _log.force_print("[WARNING] matplotlib not available, skipping plots")


def plot_metrics_by_date(val_metrics_ym, test_metrics_ym, output_dir):
    """月別（Year-Month）メトリクスをグラフ化する。

    valid（青・実線）と test（赤・破線）を同一グラフに重ね描きし、
    X 軸を YYYY-MM の時系列で表示する。

    Args:
        val_metrics_ym:  evaluate() の final_metrics_by_ym（valid セット）
        test_metrics_ym: evaluate() の final_metrics_by_ym（test セット）
        output_dir:      保存先ディレクトリ
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        # 凡例色・線種
        val_style  = {'color': '#3498db', 'linestyle': '-',  'marker': 'o',
                      'linewidth': 2, 'markersize': 4}
        test_style = {'color': '#e74c3c', 'linestyle': '--', 'marker': '^',
                      'linewidth': 2, 'markersize': 4}

        metric_configs = [
            ('region_hit_rate',    'Region Hit Rate (%)',     (0, 0)),
            ('position_hit_rate',  'Base Pos Hit Rate (%)',   (0, 1)),
            ('aa_pos_hit_rate',    'AA Pos Hit Rate (%)',     (1, 0)),
            ('codon_pos_hit_rate', 'Codon Pos Hit Rate (%)',  (1, 1)),
            ('synonymous_hit_rate','Synonymous Hit Rate (%)', (2, 0)),
            ('strength_mae',       'Strength MAE',            (2, 1)),
        ]

        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('Metrics by Collection Date (Year-Month)', fontsize=14)

        # val と test の全月を統合してソート（unknown は末尾）
        all_yms_set = set(val_metrics_ym.keys() if val_metrics_ym else []) | \
                      set(test_metrics_ym.keys() if test_metrics_ym else [])
        all_yms = sorted(
            [ym for ym in all_yms_set if ym != 'unknown'],
        ) + (['unknown'] if 'unknown' in all_yms_set else [])
        x_indices = np.arange(len(all_yms))
        ym_to_idx = {ym: i for i, ym in enumerate(all_yms)}

        for metric_key, title, (row, col) in metric_configs:
            ax = axes[row, col]

            if val_metrics_ym:
                val_x = [ym_to_idx[ym] for ym in all_yms if ym in val_metrics_ym]
                val_y = [val_metrics_ym[ym].get(metric_key, np.nan) for ym in all_yms if ym in val_metrics_ym]
                ax.plot(val_x, val_y, label='Valid', **val_style)

            if test_metrics_ym:
                test_x = [ym_to_idx[ym] for ym in all_yms if ym in test_metrics_ym]
                test_y = [test_metrics_ym[ym].get(metric_key, np.nan) for ym in all_yms if ym in test_metrics_ym]
                ax.plot(test_x, test_y, label='Test', **test_style)

            ax.set_xticks(x_indices)
            ax.set_xticklabels(all_yms, rotation=45, ha='right', fontsize=7)
            ax.set_xlabel('Year-Month')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', labelsize=7)

        # サンプル数サブプロット（右下: valid と test のサンプル数バー）
        ax = axes[2, 1]
        ax.clear()
        bar_labels, bar_counts, bar_colors = [], [], []
        if val_metrics_ym:
            for ym in sorted(val_metrics_ym.keys()):
                bar_labels.append(f'V:{ym}')
                bar_counts.append(val_metrics_ym[ym]['num_samples'])
                bar_colors.append('#3498db')
        if test_metrics_ym:
            for ym in sorted(test_metrics_ym.keys()):
                bar_labels.append(f'T:{ym}')
                bar_counts.append(test_metrics_ym[ym]['num_samples'])
                bar_colors.append('#e74c3c')
        if bar_labels:
            x_pos = np.arange(len(bar_labels))
            ax.bar(x_pos, bar_counts, color=bar_colors, alpha=0.75)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=7)
        ax.set_xlabel('Year-Month (V=Valid, T=Test)')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Count by Date')
        ax.grid(True, alpha=0.3, axis='y')
        # 凡例パッチ
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#3498db', alpha=0.75, label='Valid'),
            Patch(color='#e74c3c', alpha=0.75, label='Test'),
        ])

        plt.tight_layout()
        path = _get_plot_path(output_dir, 'metrics_by_date.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        _log.force_print(f"[INFO] Date metrics plot saved to {path}")

    except ImportError:
        _log.force_print("[WARNING] matplotlib not available, skipping date metrics plot")


def plot_category_metrics(cat_metrics, output_dir, prefix="val"):
    """流行度カテゴリ別のメトリクスをグラフ化する。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        timesteps = sorted(cat_metrics.keys())
        categories = ['low', 'medium', 'high']
        colors = {'low': '#e74c3c', 'medium': '#f39c12', 'high': '#27ae60'}

        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(f'{prefix.upper()} - Metrics by Strength Category', fontsize=14)

        # 1. Region Hit Rate by Category
        ax = axes[0, 0]
        for cat in categories:
            values = [cat_metrics[t][cat]['region_hit_rate'] for t in timesteps]
            ax.plot(timesteps, values, 'o-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Region Hit Rate by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Base Pos Hit Rate by Category
        ax = axes[0, 1]
        for cat in categories:
            values = [cat_metrics[t][cat]['position_hit_rate'] for t in timesteps]
            ax.plot(timesteps, values, 's-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Base Pos Hit Rate by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. AA Pos Hit Rate by Category
        ax = axes[1, 0]
        for cat in categories:
            values = [cat_metrics[t][cat]['aa_pos_hit_rate'] for t in timesteps]
            ax.plot(timesteps, values, '^-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('AA Pos Hit Rate by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Strength MAE by Category
        ax = axes[1, 1]
        for cat in categories:
            values = [cat_metrics[t][cat]['strength_mae'] for t in timesteps]
            ax.plot(timesteps, values, 'D-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('MAE')
        ax.set_title('Strength MAE by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Sample Distribution (Stacked Bar)
        ax = axes[2, 0]
        width = 0.6
        bottom = np.zeros(len(timesteps))
        for cat in categories:
            values = [cat_metrics[t][cat]['num_samples'] for t in timesteps]
            ax.bar(timesteps, values, width, label=cat.capitalize(), bottom=bottom, color=colors[cat], alpha=0.8)
            bottom += np.array(values)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Distribution by Category')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Codon Pos Hit Rate by Category
        ax = axes[2, 1]
        for cat in categories:
            values = [cat_metrics[t][cat].get('codon_pos_hit_rate', 0) for t in timesteps]
            ax.plot(timesteps, values, 'x-', label=cat.capitalize(), color=colors[cat])
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Hit Rate (%)')
        ax.set_title('Codon Pos Hit Rate by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = _get_plot_path(output_dir, f'{prefix}_category_plot.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        _log.force_print(f"[INFO] Category plot saved to {path}")

    except ImportError:
        _log.force_print("[WARNING] matplotlib not available, skipping plots")


def plot_combined_val_test_metrics(val_metrics, test_metrics, output_dir):
    """ValidationとTestのメトリクスを1つのグラフで比較する。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not val_metrics and not test_metrics:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Validation vs Test - Metrics Comparison', fontsize=14)

        metric_configs = [
            ('region_hit_rate', 'Region Hit Rate (%)', (0, 0)),
            ('position_hit_rate', 'Base Position Hit Rate (%)', (0, 1)),
            ('aa_pos_hit_rate', 'AA Position Hit Rate (%)', (0, 2)),
            ('codon_pos_hit_rate', 'Codon Position Hit Rate (%)', (1, 0)),
            ('synonymous_hit_rate', 'Synonymous Hit Rate (%)', (1, 1)),
            ('strength_mae', 'Strength MAE', (1, 2)),
        ]

        val_timesteps = sorted(val_metrics.keys()) if val_metrics else []
        test_timesteps = sorted(test_metrics.keys()) if test_metrics else []

        for metric_key, title, (row, col) in metric_configs:
            ax = axes[row, col]

            if val_metrics and val_timesteps:
                val_values = [val_metrics[t].get(metric_key, 0) for t in val_timesteps]
                ax.plot(val_timesteps, val_values, 'o-',
                        label='Validation', color='#3498db', linewidth=2, markersize=6)

            if test_metrics and test_timesteps:
                test_values = [test_metrics[t].get(metric_key, 0) for t in test_timesteps]
                ax.plot(test_timesteps, test_values, '^--',
                        label='Test', color='#e74c3c', linewidth=2, markersize=6)

            ax.set_xlabel('Timestep (Path Length)')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = _get_plot_path(output_dir, 'combined_val_test_metrics.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        _log.force_print(f"[INFO] Combined Val/Test metrics plot saved to {path}")

    except ImportError:
        _log.force_print("[WARNING] matplotlib not available, skipping combined plot")


def plot_combined_category_comparison(val_details, test_details, strength_thresholds, output_dir):
    """流行度カテゴリ別のVal/Test比較グラフを生成する。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        if not val_details and not test_details:
            return

        low_max, med_max = strength_thresholds

        def get_strength_category(score):
            if score < low_max:
                return 'low'
            elif score < med_max:
                return 'medium'
            else:
                return 'high'

        def calc_category_metrics(details):
            category_stats = {'low': [], 'medium': [], 'high': []}
            for d in details:
                cat = get_strength_category(d['strength_score'])
                category_stats[cat].append({
                    'region': d['hit_region'],
                    'position': d['hit_position'],
                    'aa_pos': d['hit_aa_pos'],
                    'codon': d.get('hit_codon_pos', 0),
                    'synonymous': d.get('hit_synonymous', 0)
                })

            result = {}
            for cat in ['low', 'medium', 'high']:
                if category_stats[cat]:
                    result[cat] = {
                        'count': len(category_stats[cat]),
                        'region': np.mean([x['region'] for x in category_stats[cat]]) * 100,
                        'position': np.mean([x['position'] for x in category_stats[cat]]) * 100,
                        'synonymous': np.mean([x['synonymous'] for x in category_stats[cat]]) * 100,
                    }
                else:
                    result[cat] = {'count': 0, 'region': 0, 'position': 0, 'synonymous': 0}
            return result

        val_cat = calc_category_metrics(val_details) if val_details else None
        test_cat = calc_category_metrics(test_details) if test_details else None

        categories = ['low', 'medium', 'high']
        x = np.arange(len(categories))
        width = 0.35

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Val vs Test by Strength Category (Low<{low_max}, Medium<{med_max}, High≥{med_max})', fontsize=12)

        metrics = [
            ('region', 'Region Hit Rate (%)', 0),
            ('position', 'Base Position Hit Rate (%)', 1),
            ('synonymous', 'Synonymous Hit Rate (%)', 2),
        ]

        for metric_key, title, idx in metrics:
            ax = axes[idx]

            val_values = [val_cat[cat][metric_key] if val_cat else 0 for cat in categories]
            test_values = [test_cat[cat][metric_key] if test_cat else 0 for cat in categories]

            bars1 = ax.bar(x - width / 2, val_values, width, label='Validation', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x + width / 2, test_values, width, label='Test', color='#e74c3c', alpha=0.8)

            ax.set_xlabel('Strength Category')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        path = _get_plot_path(output_dir, 'combined_category_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        _log.force_print(f"[INFO] Combined category comparison plot saved to {path}")

    except ImportError:
        _log.force_print("[WARNING] matplotlib not available, skipping combined category plot")


def print_combined_report(val_details, test_details, strength_thresholds=None):
    """ValidationとTestを統合した階層的評価レポートを標準出力に表示する。

    Args:
        val_details: Validation評価の詳細結果リスト
        test_details: Test評価の詳細結果リスト
        strength_thresholds: (low_max, med_max) の動的閾値タプル。Noneの場合はconfigの値を使用
    """
    import numpy as np
    import pandas as pd
    from tabulate import tabulate
    from .. import config

    if not val_details and not test_details:
        print("[WARNING] No results to display")
        return

    df_val = pd.DataFrame(val_details) if val_details else pd.DataFrame()
    df_test = pd.DataFrame(test_details) if test_details else pd.DataFrame()

    print("\n" + "=" * 100)
    print("【統合評価レポート - Validation vs Test】")
    print("=" * 100)

    # 動的閾値の設定
    if strength_thresholds is not None:
        low_max, med_max = strength_thresholds
    else:
        low_max = config.STRENGTH_CATEGORY_LOW_MAX
        med_max = config.STRENGTH_CATEGORY_MED_MAX

    def get_strength_category(score, low_max, med_max):
        if score < low_max:
            return 'low'
        elif score < med_max:
            return 'medium'
        else:
            return 'high'

    def calc_summary(df):
        if len(df) == 0:
            return {'サンプル': '-', 'タンパク質領域': '-', '塩基配列位置': '-',
                    'アミノ酸配列位置': '-', 'コドン位置': '-', 'シノニマス': '-'}
        return {
            'サンプル': len(df),
            'タンパク質領域': f"{df['hit_region'].mean() * 100:.1f}%",
            '塩基配列位置': f"{df['hit_position'].mean() * 100:.1f}%",
            'アミノ酸配列位置': f"{df['hit_aa_pos'].mean() * 100:.1f}%",
            'コドン位置': f"{df['hit_codon_pos'].mean() * 100:.1f}%" if 'hit_codon_pos' in df.columns else '-',
            'シノニマス': f"{df['hit_synonymous'].mean() * 100:.1f}%" if 'hit_synonymous' in df.columns else '-'
        }

    # =================================================================
    # 1. Executive Summary
    # =================================================================
    print("\n" + "-" * 100)
    print("=== 1. Executive Summary (流行度別: High/Medium/Low) ===")
    print("-" * 100)
    print(f"\n  閾値: Low(<{low_max}), Medium(<{med_max}), High(≥{med_max})\n")

    summary_rows = []
    for category in ['high', 'medium', 'low']:
        row = {'カテゴリ': category}

        if len(df_val) > 0:
            df_val['strength_cat'] = df_val['strength_score'].apply(lambda x: get_strength_category(x, low_max, med_max))
            v_cat = df_val[df_val['strength_cat'] == category]
        else:
            v_cat = pd.DataFrame()
        v_s = calc_summary(v_cat)
        row['Val_サンプル'] = v_s['サンプル']
        row['Val_タンパク質領域'] = v_s['タンパク質領域']
        row['Val_塩基配列位置'] = v_s['塩基配列位置']
        row['Val_アミノ酸配列位置'] = v_s['アミノ酸配列位置']
        row['Val_コドン位置'] = v_s['コドン位置']
        row['Val_シノニマス'] = v_s['シノニマス']

        if len(df_test) > 0:
            df_test['strength_cat'] = df_test['strength_score'].apply(lambda x: get_strength_category(x, low_max, med_max))
            t_cat = df_test[df_test['strength_cat'] == category]
        else:
            t_cat = pd.DataFrame()
        t_s = calc_summary(t_cat)
        row['Test_サンプル'] = t_s['サンプル']
        row['Test_タンパク質領域'] = t_s['タンパク質領域']
        row['Test_塩基配列位置'] = t_s['塩基配列位置']
        row['Test_アミノ酸配列位置'] = t_s['アミノ酸配列位置']
        row['Test_コドン位置'] = t_s['コドン位置']
        row['Test_シノニマス'] = t_s['シノニマス']

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print(tabulate(df_summary, headers='keys', tablefmt='simple', showindex=False))

    print("\n" + "=" * 100 + "\n")


# ==========================================
# 5. 変異位置分布 (plot_mutation_dist)
# ==========================================

def _plot_generic_dist(df, target_col, pred_col, prob_col, title, xlabel, out_path, max_x=None, is_categorical=False, vocab_map=None):
    import matplotlib.pyplot as plt
    # 正解ラベルが共起（複数変異）のものは除外
    df_single = df[df[target_col].apply(len) == 1].copy()
    
    if len(df_single) == 0:
        return

    df_single['target_pos_val'] = df_single[target_col].apply(lambda x: x[0])
    df_single['pred_pos_val'] = df_single[pred_col].apply(lambda x: x[0] if len(x) > 0 else -1)

    target_counts = df_single['target_pos_val'].value_counts().sort_index()
    pred_counts = df_single['pred_pos_val'].value_counts().sort_index()

    fig, ax1 = plt.subplots(figsize=(15, 10))
    plt.rcParams.update({'font.size': 16})

    if is_categorical:
        # Categorical variable (like regions) usually better plotted as bars or points, but step is also fine
        # We might want to fix the x-ticks to be region names
        all_x = sorted(list(set(target_counts.index.tolist() + [x for x in pred_counts.index.tolist() if x != -1])))
        if max_x is not None and len(all_x) > 0 and max(all_x) < max_x:
            all_x = list(range(max_x))
            
        t_vals = [target_counts.get(x, 0) for x in all_x]
        p_vals = [pred_counts.get(x, 0) for x in all_x]
        
        ax1.bar([x - 0.2 for x in all_x], t_vals, width=0.4, label='Ground Truth (Single Target)', color='blue', alpha=0.6)
        ax1.bar([x + 0.2 for x in all_x], p_vals, width=0.4, label='Prediction (Top-1)', color='red', alpha=0.6)
        if vocab_map:
            ax1.set_xticks(all_x)
            ax1.set_xticklabels([vocab_map.get(x, str(x)) for x in all_x], rotation=45, ha='right')
    else:
        ax1.step(target_counts.index, target_counts.values, label='Ground Truth (Single Target)', color='blue', alpha=0.6, where='mid', linewidth=1.5)
        ax1.step(pred_counts.index, pred_counts.values, label='Prediction (Top-1)', color='red', alpha=0.6, where='mid', linewidth=1.5)
        if max_x is not None:
            ax1.set_xlim(0, max_x)

    ax1.set_title(title, fontsize=22, fontweight='bold', pad=20)
    ax1.set_xlabel(xlabel, fontsize=18, fontweight='bold')
    ax1.set_ylabel('Frequency (Count)', fontsize=18, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)

    if prob_col in df_single.columns:
        ax2 = ax1.twinx()
        valid_preds = df_single[df_single['pred_pos_val'] != -1]
        pred_probs = valid_preds.groupby('pred_pos_val')[prob_col].mean().sort_index()

        ax2.scatter(
            pred_probs.index, pred_probs.values,
            label='Mean Pred Probability',
            color='purple', alpha=0.4, s=15, marker='x'
        )
        ax2.set_ylabel('Output Probability (Softmax)', fontsize=18, fontweight='bold', color='purple')
        ax2.tick_params(axis='y', labelsize=14, labelcolor='purple')
        ax2.set_ylim(-0.05, 1.05)
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=16, prop={'weight': 'bold'})
    else:
        ax1.legend(fontsize=16, prop={'weight': 'bold'})

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_mutation_dist(csv_path: str, out_path: str = None):
    """塩基、アミノ酸、タンパク質領域の頻度分布プロットを一括で生成して保存する。"""
    try:
        import ast
        import matplotlib
        matplotlib.use('Agg')
        import pandas as pd
        from .. import config

        if not os.path.exists(csv_path):
            _log.force_print(f"[ERROR] File not found: {csv_path}")
            return

        _log.force_print(f"[INFO] Loading {csv_path} for distribution plots...")
        df = pd.read_csv(csv_path)

        def parse_list(x):
            try:
                return ast.literal_eval(x) if pd.notnull(x) else []
            except Exception:
                return []

        # Generate Base Position Dist
        if 'target_position' in df.columns:
            df['target_position'] = df['target_position'].apply(parse_list)
            df['pred_position'] = df['pred_position'].apply(parse_list)
            base_dir = os.path.dirname(os.path.dirname(csv_path)) if 'csv' in os.path.basename(os.path.dirname(csv_path)) else os.path.dirname(csv_path)
            plots_dir = os.path.join(base_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            filename = os.path.basename(csv_path).replace('.csv', '_mutation_dist.png')
            out_file = out_path if out_path else os.path.join(plots_dir, filename)
            _plot_generic_dist(
                df, 'target_position', 'pred_position', 'pred_prob_position',
                f'Mutation Position Frequency & Probability\n({os.path.basename(csv_path)}, Excl. Co-occurrence)',
                'Genomic Position', out_file, max_x=30000, is_categorical=False
            )
            _log.force_print(f"[INFO] Base Mutation dist plot saved to: {out_file}")

        # Generate AA Position Dist
        if 'target_aa_pos' in df.columns:
            df['target_aa_pos'] = df['target_aa_pos'].apply(parse_list)
            df['pred_aa_pos'] = df['pred_aa_pos'].apply(parse_list)
            filename_aa = os.path.basename(csv_path).replace('.csv', '_aa_mutation_dist.png')
            aa_out_file = os.path.join(plots_dir, filename_aa)
            _plot_generic_dist(
                df, 'target_aa_pos', 'pred_aa_pos', 'pred_prob_aa_pos',
                f'AA Position Frequency & Probability\n({os.path.basename(csv_path)}, Excl. Co-occurrence)',
                'Amino Acid Position', aa_out_file, max_x=12000, is_categorical=False
            )
            _log.force_print(f"[INFO] AA Mutation dist plot saved to: {aa_out_file}")

        # Generate Region Dist
        if 'target_region' in df.columns:
            df['target_region'] = df['target_region'].apply(parse_list)
            df['pred_region'] = df['pred_region'].apply(parse_list)
            region_out_file = csv_path.replace('.csv', '_region_dist.png')
            region_map = {v: k for k, v in config.PROTEIN_VOCABS.items()} if hasattr(config, 'PROTEIN_VOCABS') else None
            _plot_generic_dist(
                df, 'target_region', 'pred_region', 'pred_prob_region',
                f'Protein Region Frequency & Probability\n({os.path.basename(csv_path)}, Excl. Co-occurrence)',
                'Protein Region', region_out_file, max_x=None, is_categorical=True, vocab_map=region_map
            )
            _log.force_print(f"[INFO] Region dist plot saved to: {region_out_file}")

    except ImportError as e:
        _log.force_print(f"[WARNING] Required library not available or error in plotting: {e}")


def run_mutation_dist_plot():
    """plot_mutation_dist のコマンドラインエントリポイント。

    Usage:
        python -m transformer_260416.utils.plotting mutation_dist \\
            --csv_path outputs/.../test_predictions.csv
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot mutation position distribution from evaluation CSV results."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="outputs/transformer_260416/results/test_predictions.csv",
        help="Path to test_predictions.csv"
    )
    parser.add_argument(
        "--out_path", type=str, default=None,
        help="Path to save the plot"
    )
    args = parser.parse_args()
    plot_mutation_dist(args.csv_path, args.out_path)


# ==========================================
# 6. 語彙ネットワーク可視化 (plot_vocab_network)
# ==========================================

# 遺伝子領域カラーマップ
REGION_COLOR_MAP = {
    # 構造タンパク質 (暖色系)
    'S': '#e94560', 'N': '#ff6b35', 'M': '#ffc300', 'E': '#ff85a1',
    # ORFタンパク質 (寒色系)
    'ORF3a': '#00b4d8', 'ORF6': '#0096c7', 'ORF7a': '#0077b6',
    'ORF7b': '#023e8a', 'ORF8': '#48cae4', 'ORF10': '#90e0ef',
    # NSPタンパク質 (緑系)
    'nsp1': '#2d6a4f', 'nsp2': '#40916c', 'nsp3': '#52b788',
    'nsp4': '#74c69d', 'nsp5': '#95d5b2', 'nsp6': '#b7e4c7',
    'nsp7': '#d8f3dc', 'nsp8': '#1b4332', 'nsp9': '#2d6a4f',
    'nsp10': '#40916c', 'nsp12': '#52b788', 'nsp13': '#74c69d',
    'nsp14': '#95d5b2', 'nsp15': '#b7e4c7', 'nsp16': '#d8f3dc',
}
_DEFAULT_REGION_COLOR = '#555555'  # non_coding


def _get_region_color(region: str) -> str:
    return REGION_COLOR_MAP.get(region, _DEFAULT_REGION_COLOR)


def load_position_to_region(codon_csv_path: str) -> dict:
    """codon_mutation CSVから 塩基位置→遺伝子領域 のマッピングを読み込む。"""
    import pandas as pd
    df = pd.read_csv(codon_csv_path, encoding='utf-8-sig')
    pos_to_region = {}
    for _, row in df.iterrows():
        pos_to_region[int(row['base_pos'])] = str(row['protein'])
    return pos_to_region


def extract_embedding_weights(model, layer_name: str):
    """指定した埋め込み層の重み行列を numpy 配列で返す。"""
    import numpy as np
    embed_layer = getattr(model.input_embed, layer_name)
    return embed_layer.weight.detach().cpu().numpy()


def _compute_cosine_similarity_matrix(weights):
    """コサイン類似度行列を計算する。"""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(weights)


# ----- 塩基埋め込み -----

def plot_base_embedding_network(model, cfg, output_dir: str, version: str = "",
                                threshold: float = -999):
    """塩基埋め込み (base_embed) のコサイン類似度ネットワーク＋ヒートマップを生成する。

    Args:
        model: ロード済みの HierarchicalTransformer
        cfg:   対応する config モジュール
        output_dir: 出力先ディレクトリ
        version: ファイル名に付与するバージョン文字列
        threshold: エッジの最小類似度 (-999 で全エッジ表示)
    """
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import networkx as nx

        base_labels_map = {v: k for k, v in cfg.BASE_VOCABS.items() if k != 'PAD'}
        base_indices = sorted(base_labels_map.keys())
        base_labels = [base_labels_map[i] for i in base_indices]

        weights = extract_embedding_weights(model, 'base_embed')
        weights_no_pad = weights[base_indices]
        sim_matrix = _compute_cosine_similarity_matrix(weights_no_pad)

        _log.force_print(f"[INFO] Base vocab: {base_labels}, embed_dim={weights.shape[1]}")

        # --- ネットワークグラフ ---
        G = nx.Graph()
        n = len(base_labels)
        for i in range(n):
            G.add_node(base_labels[i])
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > threshold:
                    G.add_edge(base_labels[i], base_labels[j],
                               weight=float(sim_matrix[i, j]))

        fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
        edges = list(G.edges(data=True))
        edge_weights = [d['weight'] for _, _, d in edges]

        if edge_weights:
            mn, mx = min(edge_weights), max(edge_weights)
            cmap = plt.get_cmap('RdYlBu_r')
            norm = Normalize(vmin=mn, vmax=mx)
            edge_colors = [cmap(norm(w)) for w in edge_weights]
            edge_widths = [1.0 + 4.0 * (w - mn) / (mx - mn + 1e-8) for w in edge_weights]
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                                   width=edge_widths, alpha=0.7)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label('Cosine Similarity', color='white', fontsize=12)
            cbar.ax.tick_params(colors='white')

        node_colors = ['#e94560' if nd in ('A', 'G') else
                       '#0f3460' if nd in ('T', 'C') else '#533483'
                       for nd in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000, node_color=node_colors,
                               edgecolors='white', linewidths=2.0, alpha=0.95)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=16, font_weight='bold',
                                font_color='white', font_family='monospace')
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8,
                                     font_color='#aaaacc', font_family='monospace',
                                     bbox=dict(boxstyle='round,pad=0.1',
                                               facecolor='#1a1a2e', edgecolor='none', alpha=0.8))
        legend_patches = [
            mpatches.Patch(color='#e94560', label='Purine (A, G)'),
            mpatches.Patch(color='#0f3460', label='Pyrimidine (T, C)'),
            mpatches.Patch(color='#533483', label='Other (N, n)'),
        ]
        ax.legend(handles=legend_patches, loc='upper left', fontsize=10,
                  facecolor='#16213e', edgecolor='#e94560', labelcolor='white')
        title = f'Base Nucleotide Embedding — Cosine Similarity Network'
        if version:
            title += f'\n({version})'
        ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
        ax.axis('off')
        plt.tight_layout()
        net_path = _get_plot_path(output_dir, f'vocab_network_base_{version}.png')
        plt.savefig(net_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        _log.force_print(f"[INFO] Base network saved: {net_path}")

        # --- ヒートマップ ---
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        im = ax.imshow(sim_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(base_labels, fontsize=12, color='white', fontweight='bold')
        ax.set_yticklabels(base_labels, fontsize=12, color='white', fontweight='bold')
        for i in range(n):
            for j in range(n):
                tc = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha='center', va='center',
                        fontsize=10, color=tc, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Cosine Similarity', color='white', fontsize=12)
        cbar.ax.tick_params(colors='white')
        ax.set_title(f'Base Nucleotide Embedding — Cosine Similarity ({version})',
                     fontsize=14, fontweight='bold', color='white', pad=15)
        plt.tight_layout()
        hm_path = _get_plot_path(output_dir, f'vocab_heatmap_base_{version}.png')
        plt.savefig(hm_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        _log.force_print(f"[INFO] Base heatmap saved: {hm_path}")

    except ImportError as e:
        _log.force_print(f"[WARNING] Required library not available: {e}")


# ----- 位置埋め込み: 閾値ネットワーク -----

def plot_position_threshold_network(model, pos_to_region: dict, output_dir: str,
                                    version: str = "", threshold: float = None,
                                    max_nodes: int = 500):
    """位置埋め込み (pos_embed) の高類似度ペアのみのネットワークグラフを生成する。

    Args:
        model: ロード済みモデル
        pos_to_region: {塩基位置: 遺伝子領域名} の辞書
        output_dir: 出力先ディレクトリ
        version: ファイル名バージョン
        threshold: エッジ閾値。None の場合は 99.9%ile を自動設定
        max_nodes: 表示するノードの最大数
    """
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import networkx as nx

        weights = extract_embedding_weights(model, 'pos_embed')
        max_genome_pos = min(weights.shape[0] - 1, 30000)
        pos_indices = list(range(1, max_genome_pos + 1))
        weights_no_pad = weights[pos_indices]

        # 類似度分布をサンプリングして確認
        n = len(weights_no_pad)
        np.random.seed(42)
        sample_size = min(2000, n)
        sample_idx = np.random.choice(n, sample_size, replace=False)
        from sklearn.metrics.pairwise import cosine_similarity
        sample_sim = cosine_similarity(weights_no_pad[sample_idx])
        upper_vals = sample_sim[np.triu_indices(sample_size, k=1)]
        _log.force_print(
            f"[INFO] Pos sim distribution (sample={sample_size}): "
            f"min={upper_vals.min():.4f}, max={upper_vals.max():.4f}, "
            f"mean={upper_vals.mean():.4f}"
        )
        for pct in [99.0, 99.5, 99.9]:
            _log.force_print(f"  {pct}%ile: {np.percentile(upper_vals, pct):.4f}")

        if threshold is None:
            threshold = float(np.percentile(upper_vals, 99.9))
            _log.force_print(f"[INFO] Auto threshold (99.9%ile): {threshold:.4f}")

        sim_matrix = cosine_similarity(weights_no_pad)
        np.fill_diagonal(sim_matrix, 0)
        max_sims = sim_matrix.max(axis=1)
        top_indices = np.argsort(max_sims)[-max_nodes:]

        G = nx.Graph()
        for idx in top_indices:
            p = pos_indices[idx]
            G.add_node(p, region=pos_to_region.get(p, 'unknown'))
        for i in top_indices:
            for j in top_indices:
                if i < j and sim_matrix[i, j] > threshold:
                    G.add_edge(pos_indices[i], pos_indices[j],
                               weight=float(sim_matrix[i, j]))

        for isolate in list(nx.isolates(G)):
            G.remove_node(isolate)

        if len(G.nodes()) == 0:
            _log.force_print(f"[WARNING] No edges found with threshold={threshold}. Lower the threshold.")
            return

        _log.force_print(f"[INFO] Threshold network: nodes={len(G.nodes())}, edges={len(G.edges())}")

        fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        pos_layout = nx.spring_layout(G, k=1.5, iterations=80, seed=42)
        node_colors = [_get_region_color(G.nodes[nd].get('region', '')) for nd in G.nodes()]
        edges = list(G.edges(data=True))
        edge_weights = [d['weight'] for _, _, d in edges]
        if edge_weights:
            cmap = plt.get_cmap('RdYlBu_r')
            norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
            edge_colors = [cmap(norm(w)) for w in edge_weights]
        else:
            edge_colors = ['gray']
        nx.draw_networkx_edges(G, pos_layout, ax=ax, edge_color=edge_colors,
                               width=0.5, alpha=0.4)
        nx.draw_networkx_nodes(G, pos_layout, ax=ax, node_size=40,
                               node_color=node_colors, edgecolors='none', alpha=0.8)

        shown_regions = set(G.nodes[nd].get('region', 'unknown') for nd in G.nodes())
        legend_patches = []
        for region in sorted(shown_regions):
            if not region.startswith('non_coding'):
                legend_patches.append(
                    mpatches.Patch(color=_get_region_color(region), label=region))
        if any(r.startswith('non_coding') for r in shown_regions):
            legend_patches.append(
                mpatches.Patch(color=_DEFAULT_REGION_COLOR, label='non_coding'))
        ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2,
                  facecolor='#16213e', edgecolor='#444', labelcolor='white')
        ax.set_title(f'Position Embedding — Threshold Network\n({version})',
                     fontsize=14, fontweight='bold', color='white', pad=20)
        ax.axis('off')
        plt.tight_layout()
        out_path = _get_plot_path(output_dir, f'pos_network_threshold_{version}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        _log.force_print(f"[INFO] Threshold network saved: {out_path}")

    except ImportError as e:
        _log.force_print(f"[WARNING] Required library not available: {e}")


# ----- 位置埋め込み: t-SNE 散布図 -----

def plot_position_tsne(model, pos_to_region: dict, output_dir: str,
                       version: str = "", perplexity: int = 50):
    """位置埋め込みを t-SNE で2D可視化し、遺伝子領域で色分けする。"""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        weights = extract_embedding_weights(model, 'pos_embed')
        max_genome_pos = min(weights.shape[0] - 1, 30000)
        pos_indices = list(range(1, max_genome_pos + 1))
        weights_no_pad = weights[pos_indices]

        _log.force_print(f"[INFO] Running t-SNE (perplexity={perplexity}, n={len(weights_no_pad)}) ...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    n_iter=1000, learning_rate='auto', init='pca')
        coords = tsne.fit_transform(weights_no_pad)

        regions = [pos_to_region.get(p, 'unknown') for p in pos_indices]
        region_groups: dict = {}
        for i, region in enumerate(regions):
            key = 'non_coding' if region.startswith('non_coding') else region
            region_groups.setdefault(key, []).append(i)

        fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        if 'non_coding' in region_groups:
            idx = region_groups['non_coding']
            ax.scatter(coords[idx, 0], coords[idx, 1], c=_DEFAULT_REGION_COLOR,
                       s=3, alpha=0.15, label='non_coding', rasterized=True)
        for region in sorted(region_groups.keys()):
            if region == 'non_coding':
                continue
            idx = region_groups[region]
            ax.scatter(coords[idx, 0], coords[idx, 1], c=_get_region_color(region),
                       s=8, alpha=0.6, label=region, rasterized=True)

        ax.legend(fontsize=7, ncol=3, loc='upper right',
                  facecolor='#16213e', edgecolor='#444', labelcolor='white', markerscale=3)
        ax.set_title(f'Position Embedding — t-SNE (colored by gene region)\n({version})',
                     fontsize=14, fontweight='bold', color='white', pad=15)
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_color('#333333')
        ax.set_xlabel('t-SNE 1', color='#aaaaaa', fontsize=10)
        ax.set_ylabel('t-SNE 2', color='#aaaaaa', fontsize=10)
        plt.tight_layout()
        out_path = _get_plot_path(output_dir, f'pos_tsne_{version}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        _log.force_print(f"[INFO] t-SNE plot saved: {out_path}")

    except ImportError as e:
        _log.force_print(f"[WARNING] Required library not available: {e}")


# ----- 位置埋め込み: クラスタ間ネットワーク -----

def plot_position_cluster_network(model, pos_to_region: dict, output_dir: str,
                                  version: str = "", n_clusters: int = 20):
    """位置埋め込みを k-means でクラスタリングしてクラスタ重心間のネットワークを生成する。"""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import networkx as nx
        from sklearn.cluster import KMeans

        weights = extract_embedding_weights(model, 'pos_embed')
        max_genome_pos = min(weights.shape[0] - 1, 30000)
        pos_indices = list(range(1, max_genome_pos + 1))
        weights_no_pad = weights[pos_indices]

        _log.force_print(f"[INFO] k-means clustering (k={n_clusters}) ...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(weights_no_pad)
        centroids = kmeans.cluster_centers_

        cluster_info = {}
        for c in range(n_clusters):
            mask = cluster_labels == c
            cluster_positions = [pos_indices[i] for i in range(len(pos_indices)) if mask[i]]
            cluster_regions = [pos_to_region.get(p, 'unknown') for p in cluster_positions]
            region_counts: dict = {}
            for r in cluster_regions:
                key = 'non_coding' if r.startswith('non_coding') else r
                region_counts[key] = region_counts.get(key, 0) + 1
            dominant = max(region_counts, key=region_counts.get)
            cluster_info[c] = {
                'size': int(mask.sum()),
                'dominant_region': dominant,
                'dominant_pct': region_counts[dominant] / len(cluster_regions) * 100,
                'region_counts': region_counts,
            }

        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(centroids)
        G = nx.Graph()
        for c in range(n_clusters):
            info = cluster_info[c]
            label = f"C{c}\n{info['dominant_region']}\n({info['size']})"
            G.add_node(c, label=label, region=info['dominant_region'], size=info['size'])
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if sim_matrix[i, j] > 0.3:
                    G.add_edge(i, j, weight=float(sim_matrix[i, j]))

        fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        pos_layout = nx.spring_layout(G, k=3.0, iterations=100, seed=42)
        sizes = [cluster_info[nd]['size'] for nd in G.nodes()]
        max_size = max(sizes) if sizes else 1
        node_sizes = [300 + 3000 * (s / max_size) for s in sizes]
        node_colors = [_get_region_color(cluster_info[nd]['dominant_region']) for nd in G.nodes()]
        edges = list(G.edges(data=True))
        edge_weights_list = [d['weight'] for _, _, d in edges]
        if edge_weights_list:
            cmap = plt.get_cmap('RdYlBu_r')
            norm = Normalize(vmin=min(edge_weights_list), vmax=max(edge_weights_list))
            ec = [cmap(norm(w)) for w in edge_weights_list]
            ew = [0.5 + 3.0 * (w - min(edge_weights_list)) /
                  (max(edge_weights_list) - min(edge_weights_list) + 1e-8)
                  for w in edge_weights_list]
        else:
            ec, ew = ['gray'], [1.0]
        nx.draw_networkx_edges(G, pos_layout, ax=ax, edge_color=ec, width=ew, alpha=0.5)
        nx.draw_networkx_nodes(G, pos_layout, ax=ax, node_size=node_sizes,
                               node_color=node_colors, edgecolors='white',
                               linewidths=1.5, alpha=0.9)
        labels_dict = {nd: G.nodes[nd]['label'] for nd in G.nodes()}
        nx.draw_networkx_labels(G, pos_layout, labels_dict, ax=ax,
                                font_size=7, font_color='white', font_weight='bold')
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos_layout, edge_labels, ax=ax,
                                     font_size=6, font_color='#aaaacc',
                                     bbox=dict(boxstyle='round,pad=0.1',
                                               facecolor='#1a1a2e', edgecolor='none', alpha=0.8))
        if edge_weights_list:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
            cbar.set_label('Cosine Similarity (centroids)', color='white', fontsize=10)
            cbar.ax.tick_params(colors='white')

        ax.set_title(f'Position Embedding — Cluster Network (k={n_clusters})\n({version})',
                     fontsize=14, fontweight='bold', color='white', pad=20)
        ax.axis('off')
        plt.tight_layout()
        out_path = _get_plot_path(output_dir, f'pos_cluster_network_{version}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        _log.force_print(f"[INFO] Cluster network saved: {out_path}")

    except ImportError as e:
        _log.force_print(f"[WARNING] Required library not available: {e}")


# ----- 主要変異位置ネットワーク -----

def plot_major_mutation_network(model, pos_to_region: dict, freq_csv: str,
                                codon_csv: str, output_dir: str,
                                version: str = "", top_n: int = 50):
    """高頻度変異位置のみを抽出し、位置埋め込みのコサイン類似度ネットワークを描画する。

    ノードサイズは変異頻度、ノードカラーは遺伝子領域に対応する。

    Args:
        model: ロード済みモデル
        pos_to_region: {塩基位置: 遺伝子領域名}
        freq_csv: 変異頻度CSV のパス
        codon_csv: コドンCSV のパス（タンパク質位置情報取得用）
        output_dir: 出力先
        version: ファイル名バージョン
        top_n: 上位何位置を表示するか
    """
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import networkx as nx
        from sklearn.metrics.pairwise import cosine_similarity

        freq_df = pd.read_csv(freq_csv)
        mut_cols = [c for c in freq_df.columns if c != 'position']
        freq_df['total_freq'] = freq_df[mut_cols].sum(axis=1)

        codon_df = pd.read_csv(codon_csv, encoding='utf-8-sig')

        _log.force_print(f"[INFO] Total freq sum: {freq_df['total_freq'].sum():.0f}")

        top_positions = freq_df.nlargest(top_n, 'total_freq')
        positions = top_positions['position'].tolist()
        freqs = top_positions['total_freq'].tolist()

        pos_weights = extract_embedding_weights(model, 'pos_embed')
        embeddings, valid_positions, valid_freqs = [], [], []
        for p, f in zip(positions, freqs):
            if p < pos_weights.shape[0]:
                embeddings.append(pos_weights[p])
                valid_positions.append(p)
                valid_freqs.append(f)
        embeddings = np.array(embeddings)
        sim_matrix = cosine_similarity(embeddings)

        labels = []
        for p in valid_positions:
            row = codon_df[codon_df['base_pos'] == p]
            if len(row) > 0:
                protein = str(row.iloc[0]['protein'])
                protein_pos = str(row.iloc[0]['protein_pos'])
                label = f"{p}\n(nc)" if protein.startswith('non_coding') else f"{p}\n{protein}:{protein_pos}"
            else:
                label = str(p)
            labels.append(label)

        upper_tri = sim_matrix[np.triu_indices(len(valid_positions), k=1)]
        edge_threshold = float(np.percentile(upper_tri, 80)) if len(upper_tri) > 0 else 0.0
        _log.force_print(f"[INFO] Edge threshold (80%ile): {edge_threshold:.4f}")

        G = nx.Graph()
        for i, (p, lab) in enumerate(zip(valid_positions, labels)):
            region = pos_to_region.get(p, 'unknown')
            region_key = 'non_coding' if region.startswith('non_coding') else region
            freq_row = freq_df[freq_df['position'] == p]
            if len(freq_row) > 0:
                dominant_mut = mut_cols[np.argmax(freq_row[mut_cols].values[0])]
            else:
                dominant_mut = ''
            G.add_node(i, label=lab, region=region_key, freq=valid_freqs[i],
                       position=p, dominant_mut=dominant_mut)
        for i in range(len(valid_positions)):
            for j in range(i + 1, len(valid_positions)):
                if sim_matrix[i, j] > edge_threshold:
                    G.add_edge(i, j, weight=float(sim_matrix[i, j]))

        _log.force_print(f"[INFO] Major mutation network: nodes={len(G.nodes())}, edges={len(G.edges())}")

        fig, ax = plt.subplots(1, 1, figsize=(16, 14), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        pos_layout = nx.spring_layout(G, k=2.5, iterations=120, seed=42)
        max_freq = max(valid_freqs)
        node_sizes = [200 + 2500 * (G.nodes[nd]['freq'] / max_freq) for nd in G.nodes()]
        node_colors = [_get_region_color(G.nodes[nd]['region']) for nd in G.nodes()]
        edges = list(G.edges(data=True))
        edge_weights_list = [d['weight'] for _, _, d in edges]
        if edge_weights_list:
            cmap = plt.get_cmap('RdYlBu_r')
            norm = Normalize(vmin=min(edge_weights_list), vmax=max(edge_weights_list))
            ec = [cmap(norm(w)) for w in edge_weights_list]
            ew = [0.3 + 2.5 * (w - min(edge_weights_list)) /
                  (max(edge_weights_list) - min(edge_weights_list) + 1e-8)
                  for w in edge_weights_list]
        else:
            ec, ew = ['gray'], [1.0]
        nx.draw_networkx_edges(G, pos_layout, ax=ax, edge_color=ec, width=ew, alpha=0.4)
        nx.draw_networkx_nodes(G, pos_layout, ax=ax, node_size=node_sizes,
                               node_color=node_colors, edgecolors='white',
                               linewidths=1.0, alpha=0.9)
        labels_dict = {nd: G.nodes[nd]['label'] for nd in G.nodes()}
        nx.draw_networkx_labels(G, pos_layout, labels_dict, ax=ax,
                                font_size=6, font_color='white', font_weight='bold')
        if edge_weights_list:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
            cbar.set_label('Cosine Similarity', color='white', fontsize=10)
            cbar.ax.tick_params(colors='white')
        shown_regions = set(G.nodes[nd]['region'] for nd in G.nodes())
        legend_patches = []
        for region in sorted(shown_regions):
            legend_patches.append(mpatches.Patch(
                color=_DEFAULT_REGION_COLOR if region == 'non_coding' else _get_region_color(region),
                label=region
            ))
        ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2,
                  facecolor='#16213e', edgecolor='#444', labelcolor='white')
        ax.set_title(
            f'Major Mutation Positions — Embedding Similarity Network (top {top_n})\n({version})',
            fontsize=14, fontweight='bold', color='white', pad=20
        )
        ax.axis('off')
        plt.tight_layout()
        out_path = _get_plot_path(output_dir, f'pos_major_mutations_{version}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        _log.force_print(f"[INFO] Major mutation network saved: {out_path}")

    except ImportError as e:
        _log.force_print(f"[WARNING] Required library not available: {e}")


def run_vocab_network_plot():
    """語彙ネットワーク可視化のコマンドラインエントリポイント。

    Usage:
        python -m transformer_260416.utils.plotting vocab_network \\
            --model_path outputs/transformer_260416/results/<ts>/best_model.pth
    """
    import argparse
    import importlib
    import torch

    parser = argparse.ArgumentParser(description='語彙ネットワーク描画')
    parser.add_argument('--model_path', type=str, required=True,
                        help='学習済みモデルのパス (.pth)')
    parser.add_argument('--output_dir', type=str, default='outputs/vocab_network',
                        help='出力先ディレクトリ')
    parser.add_argument('--codon_csv', type=str, default='reference/codon/codon_mutation4.csv')
    parser.add_argument('--freq_csv', type=str,
                        default='outputs/table_heatmap/251031/table_set/table_set.csv')
    parser.add_argument('--threshold', type=float, default=-999)
    parser.add_argument('--pos_threshold', type=float, default=None)
    parser.add_argument('--n_clusters', type=int, default=20)
    parser.add_argument('--top_n', type=int, default=50)
    parser.add_argument('--skip_base', action='store_true')
    parser.add_argument('--skip_position', action='store_true')
    parser.add_argument('--skip_major', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # モデルパスからバージョンを抽出
    import os as _os
    parts = _os.path.normpath(args.model_path).split(_os.sep)
    version = ''
    for i, part in enumerate(parts):
        if part == 'results' and i + 1 < len(parts):
            version = parts[i + 1]
            break
    if not version:
        version = _os.path.basename(_os.path.dirname(args.model_path))

    # configとmodelを動的ロード（出力パスから対応パッケージを推定）
    cfg = None
    model_module = None
    for i, part in enumerate(parts):
        if part.startswith('transformer_') and i > 0 and parts[i - 1] == 'outputs':
            pkg_name = part
            try:
                cfg = importlib.import_module(f'{pkg_name}.config')
                model_module = importlib.import_module(f'{pkg_name}.model')
                _log.force_print(f"[INFO] Using {pkg_name}.config / model")
                break
            except ImportError:
                pass
    if cfg is None:
        from transformer_260416 import config as cfg
    if model_module is None:
        from transformer_260416 import model as model_module

    _log.force_print(f"[INFO] Loading model: {args.model_path}")
    m = model_module.HierarchicalTransformer()
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    m.load_state_dict(state_dict)
    m.eval()

    pos_to_region = load_position_to_region(args.codon_csv)

    if not args.skip_base:
        _log.force_print("[INFO] === 塩基埋め込み可視化 ===")
        plot_base_embedding_network(m, cfg, args.output_dir, version, args.threshold)

    if not args.skip_position:
        _log.force_print("[INFO] === 位置埋め込み可視化 ===")
        plot_position_threshold_network(m, pos_to_region, args.output_dir, version,
                                        args.pos_threshold)
        plot_position_tsne(m, pos_to_region, args.output_dir, version)
        plot_position_cluster_network(m, pos_to_region, args.output_dir, version,
                                      args.n_clusters)

    if not args.skip_major:
        _log.force_print("[INFO] === 主要変異位置ネットワーク ===")
        plot_major_mutation_network(m, pos_to_region, args.freq_csv, args.codon_csv,
                                    args.output_dir, version, args.top_n)

    _log.force_print(f"[INFO] Done. Output: {args.output_dir}")


# ─────────────────────────────────────────────────────────────
# 追加グラフ関数
# ─────────────────────────────────────────────────────────────

def plot_strength_calibration(details, output_dir, prefix="test"):
    """Strength（流行度）の予測値 vs 実際値の散布図を保存する。

    Args:
        details: evaluate() が返す detailed_results
        output_dir: 保存先ディレクトリ
        prefix: ファイル名プレフィックス
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    pred_vals   = [r.get('pred_strength', 0.0) for r in details]
    target_vals = [r.get('strength_score', 0.0)  for r in details]

    if not pred_vals:
        return

    pred_arr   = np.array(pred_vals)
    target_arr = np.array(target_vals)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    sc = ax.scatter(target_arr, pred_arr, alpha=0.35, s=12,
                    c=target_arr, cmap='plasma', edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Actual Strength', pad=0.02)

    # 完全予測の対角線
    lo = min(target_arr.min(), pred_arr.min())
    hi = max(target_arr.max(), pred_arr.max())
    ax.plot([lo, hi], [lo, hi], '--', color='#e94560', linewidth=1.5, label='Perfect')

    # 相関係数
    corr = np.corrcoef(target_arr, pred_arr)[0, 1]
    mae  = np.mean(np.abs(pred_arr - target_arr))
    ax.text(0.05, 0.92, f'r = {corr:.3f}\nMAE = {mae:.3f}',
            transform=ax.transAxes, color='white', fontsize=11)

    ax.set_xlabel('Actual Strength', color='white')
    ax.set_ylabel('Predicted Strength', color='white')
    ax.set_title(f'Strength Calibration ({prefix.upper()})', color='white', fontsize=13)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.legend(framealpha=0.3, labelcolor='white')

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_strength_calibration.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Strength calibration plot saved to {path}")


def plot_per_position_recall(details, output_dir, prefix="test", top_n=40):
    """出現頻度上位 N 塩基位置の Recall@1 を棒グラフで保存する。

    Args:
        details: evaluate() が返す detailed_results
        output_dir: 保存先ディレクトリ
        prefix: ファイル名プレフィックス
        top_n: 表示する上位位置数
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import os

    pos_tp  = {}
    pos_cnt = {}
    for r in details:
        for p in r.get('targets_position', set()):
            pos_cnt[p] = pos_cnt.get(p, 0) + 1
            if p in r.get('preds_position', set()):
                pos_tp[p] = pos_tp.get(p, 0) + 1

    if not pos_cnt:
        return

    # 出現数で上位 N 位置を抽出
    sorted_pos = sorted(pos_cnt, key=lambda x: pos_cnt[x], reverse=True)[:top_n]
    labels     = [str(p) for p in sorted_pos]
    recalls    = [pos_tp.get(p, 0) / pos_cnt[p] * 100 for p in sorted_pos]
    counts     = [pos_cnt[p] for p in sorted_pos]

    cmap   = cm.get_cmap('RdYlGn')
    colors = [cmap(r / 100) for r in recalls]

    fig, ax1 = plt.subplots(figsize=(max(12, top_n * 0.4), 5), facecolor='#1a1a2e')
    ax1.set_facecolor('#16213e')

    x = np.arange(len(labels))
    bars = ax1.bar(x, recalls, color=colors, alpha=0.85, edgecolor='none')
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Recall@1 (%)', color='white')
    ax1.set_xlabel('Position ID (top by frequency)', color='white')
    ax1.set_title(f'Per-Position Recall@1 ({prefix.upper()}, top {top_n})', color='white', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, fontsize=7, color='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.axhline(y=sum(recalls)/len(recalls), color='#e94560', linestyle='--',
                linewidth=1, label=f'avg {sum(recalls)/len(recalls):.1f}%')
    ax1.legend(framealpha=0.3, labelcolor='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_per_position_recall.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Per-position recall plot saved to {path}")


def plot_topk_precision(topk_results, output_dir, prefix="test"):
    """Top-K Precision / Hit Rate を K ごとにグループ棒グラフで保存する。

    Args:
        topk_results: evaluate_topk() の返り値
        output_dir: 保存先ディレクトリ
        prefix: ファイル名プレフィックス
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    tasks = list(topk_results.keys())
    ks    = sorted(k for k in next(iter(topk_results.values())).keys() if isinstance(k, int))
    n_tasks = len(tasks)
    n_ks    = len(ks)

    hit_rates  = np.array([[topk_results[t][k]['hit_rate']  for t in tasks] for k in ks])
    precisions = np.array([[topk_results[t][k]['precision'] for t in tasks] for k in ks])

    x = np.arange(n_tasks)
    width = 0.8 / n_ks
    colors = ['#e94560', '#0f3460', '#533483', '#16213e', '#1a1a2e'][:n_ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#1a1a2e')
    for ax in (ax1, ax2):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, color='white', rotation=20)
        ax.set_ylim(0, 110)

    for i, (k, col) in enumerate(zip(ks, colors)):
        offset = (i - n_ks / 2 + 0.5) * width
        ax1.bar(x + offset, hit_rates[i],  width=width, label=f'K={k}', color=col, alpha=0.85)
        ax2.bar(x + offset, precisions[i], width=width, label=f'K={k}', color=col, alpha=0.85)

    ax1.set_ylabel('Hit Rate (%)', color='white')
    ax1.set_title(f'Top-K Hit Rate ({prefix.upper()})', color='white')
    ax1.legend(framealpha=0.3, labelcolor='white')

    ax2.set_ylabel('Precision (%)', color='white')
    ax2.set_title(f'Top-K Precision ({prefix.upper()})', color='white')
    ax2.legend(framealpha=0.3, labelcolor='white')

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_topk_precision.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Top-K precision plot saved to {path}")


def plot_strength_fine(details, output_dir, prefix="test"):
    """流行規模9区分別の Hit Rate を grouped bar chart で保存する。"""
    import math
    import matplotlib.pyplot as plt
    import numpy as np

    if not details:
        return

    bins = [
        ('~100',   math.log1p(100)),   ('~500',   math.log1p(500)),
        ('~1K',    math.log1p(1_000)), ('~5K',    math.log1p(5_000)),
        ('~10K',   math.log1p(10_000)),('~50K',   math.log1p(50_000)),
        ('~100K',  math.log1p(100_000)),('~500K', math.log1p(500_000)),
        ('>500K',  float('inf')),
    ]
    tasks    = ['region', 'position', 'aa_pos']
    task_labels = ['Region', 'NucPos', 'AAPos']
    colors   = ['#4c72b0', '#dd8452', '#55a868']

    fine_stats = {}
    for r in details:
        score = r.get('strength_score', 0.0)
        cat   = next((lbl for lbl, thr in bins if score < thr), bins[-1][0])
        if cat not in fine_stats:
            fine_stats[cat] = {'n': 0, 'hits': {t: 0 for t in tasks}}
        fine_stats[cat]['n'] += 1
        for t in tasks:
            fine_stats[cat]['hits'][t] += int(r.get(f'hit_{t}', False))

    categories = [lbl for lbl, _ in bins]
    n_cats = len(categories)
    n_tasks = len(tasks)
    width = 0.8 / n_tasks

    fig, ax = plt.subplots(figsize=(13, 5))
    ax2 = ax.twinx()

    ns = [fine_stats.get(c, {'n': 0})['n'] for c in categories]
    ax2.bar(range(n_cats), ns, color='gray', alpha=0.12, zorder=0)
    ax2.set_ylabel('Sample count', color='gray', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    x = np.arange(n_cats)
    for i, (t, lbl, col) in enumerate(zip(tasks, task_labels, colors)):
        rates = [
            fine_stats[c]['hits'][t] / fine_stats[c]['n'] * 100
            if fine_stats.get(c, {'n': 0})['n'] > 0 else 0.0
            for c in categories
        ]
        offset = (i - n_tasks / 2 + 0.5) * width
        ax.bar(x + offset, rates, width=width, label=lbl, color=col, alpha=0.85, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('sequence count by linage')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title(f'Hit Rate by Strength Category ({prefix.upper()})')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_strength_fine.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Strength fine plot saved to {path}")


def plot_lineage_metrics(details, output_dir, prefix="test", top_n=30):
    """系統別 Hit Rate（position）を横棒グラフで保存する。色でエントロピーを表現。"""
    import math
    from collections import Counter
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np

    if not details:
        return

    lineage_stats = {}
    for r in details:
        lin = r.get('strain', 'unknown') or 'unknown'
        if lin not in lineage_stats:
            lineage_stats[lin] = {'n': 0, 'hit_pos': 0, 'positions': []}
        lineage_stats[lin]['n'] += 1
        lineage_stats[lin]['hit_pos'] += int(r.get('hit_position', False))
        lineage_stats[lin].extend if False else lineage_stats[lin]['positions'].extend(
            list(r.get('targets_position', set()))
        )

    rows = []
    for lin, s in lineage_stats.items():
        n = s['n']
        pos_list = s['positions']
        entropy_norm = 0.0
        if pos_list:
            counts = Counter(pos_list)
            total  = len(pos_list)
            entr   = -sum((c / total) * math.log2(c / total) for c in counts.values())
            max_e  = math.log2(len(counts)) if len(counts) > 1 else 1.0
            entropy_norm = entr / max_e
        rows.append({
            'lineage': lin, 'n': n,
            'hit_rate': s['hit_pos'] / n * 100 if n > 0 else 0.0,
            'entropy_norm': entropy_norm,
        })

    rows.sort(key=lambda r: r['n'], reverse=True)
    rows = rows[:top_n]
    rows = rows[::-1]  # 上から大→小になるよう反転

    labels     = [f"{r['lineage']} (n={r['n']:,})" for r in rows]
    hit_rates  = [r['hit_rate'] for r in rows]
    entropies  = [r['entropy_norm'] for r in rows]

    cmap  = cm.RdYlGn_r
    norm  = mcolors.Normalize(vmin=0.0, vmax=1.0)
    colors = [cmap(norm(e)) for e in entropies]

    fig_h = max(6, len(rows) * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    bars = ax.barh(range(len(rows)), hit_rates, color=colors, alpha=0.85)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Position Hit Rate (%)')
    ax.set_title(f'Position Hit Rate by Lineage — top {top_n} ({prefix.upper()})\n'
                 f'Color: target entropy (green=low/easy, red=high/hard)')
    ax.set_xlim(0, 105)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Normalized entropy', pad=0.01, fraction=0.02)

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_lineage_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Lineage metrics plot saved to {path}")


def _lineage_group(strain):
    """フル系統名を Pango 上位2階層に丸める（例: BA.2.86.1→BA.2, XBB.1.5→XBB.1）。

    ドットを含まない clade 名（例: '22B (Omicron)'）はそのまま返す。
    lineage_metrics.csv と粒度を揃え、微小系統の乱立とカバレッジ変動を防ぐ。
    """
    if not strain or strain == 'unknown':
        return 'unknown'
    parts = str(strain).split('.')
    return '.'.join(parts[:2]) if len(parts) >= 2 else str(strain)


def _diversity_min_samples(min_samples=None):
    """min_samples 未指定時は config.DIVERSITY_PLOT_MIN_SAMPLES（既定10）を使う。"""
    if min_samples is not None:
        return min_samples
    from .. import config
    return getattr(config, 'DIVERSITY_PLOT_MIN_SAMPLES', 10)


def _lineage_diversity_rows(details, min_samples):
    """系統（Pango 上位2階層）ごとに n / position hit rate / 各種多様度を集計する。

    3種の散布図と CSV が共有する唯一の集計源。
      entropy_norm : 正規化 Shannon エントロピー
      unique_ratio : ユニーク位置数 / 全位置出現数
      simpson      : Gini–Simpson 多様度 (1 − Σpᵢ²、生態学標準)
    """
    import math
    from collections import Counter

    stats = {}
    for r in details:
        lin = _lineage_group(r.get('strain', 'unknown'))
        s = stats.setdefault(lin, {'n': 0, 'hit_pos': 0, 'positions': []})
        s['n'] += 1
        s['hit_pos'] += int(r.get('hit_position', False))
        s['positions'].extend(list(r.get('targets_position', set())))

    rows = []
    for lin, s in stats.items():
        n = s['n']
        if n < min_samples:
            continue
        pos_list = s['positions']
        entropy_norm = unique_ratio = simpson = 0.0
        if pos_list:
            counts = Counter(pos_list)
            total = len(pos_list)
            probs = [c / total for c in counts.values()]
            entr = -sum(p * math.log2(p) for p in probs)
            max_e = math.log2(len(counts)) if len(counts) > 1 else 1.0
            entropy_norm = entr / max_e
            unique_ratio = len(counts) / total
            simpson = 1.0 - sum(p * p for p in probs)
        rows.append({
            'lineage': lin, 'n': n,
            'hit_rate': s['hit_pos'] / n * 100,
            'entropy_norm': entropy_norm,
            'unique_ratio': unique_ratio,
            'simpson': simpson,
        })
    return rows


def save_lineage_diversity_csv(details, output_dir, prefix="test", min_samples=None):
    """散布図3種（entropy / unique / simpson）の裏づけとなる系統別テーブルを CSV 保存する。

    列: lineage, n, hit_rate, entropy_norm, unique_ratio, simpson。
    PNG はコード版に依存し比較しづらいため、この CSV を run 間比較・再描画の基準にする。
    """
    if not details:
        return None
    rows = _lineage_diversity_rows(details, _diversity_min_samples(min_samples))
    if not rows:
        return None
    import pandas as pd
    df = pd.DataFrame(rows).sort_values('n', ascending=False).reset_index(drop=True)
    target_dir = os.path.join(output_dir, 'csv')
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f'{prefix}_lineage_diversity.csv')
    df.to_csv(path, index=False)
    _log.force_print(f"[INFO] Lineage diversity table saved to {path}")
    return df


def plot_entropy_vs_accuracy(details, output_dir, prefix="test", min_samples=None):
    """エントロピー vs 正解率の散布図を系統単位で保存する。
    X軸: entropy_norm、Y軸: position hit rate、点サイズ: num_samples（対数）、色: 系統ファミリー。
    系統粒度は Pango 上位2階層、min_samples は config.DIVERSITY_PLOT_MIN_SAMPLES に従う。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not details:
        return
    min_samples = _diversity_min_samples(min_samples)
    rows = _lineage_diversity_rows(details, min_samples)
    if not rows:
        return

    # 系統ファミリーを判定（色分け用）
    def _family(lin):
        if lin.startswith('AY.'):  return 'AY.* (Delta sub)'
        if lin.startswith('BA.'):  return 'BA.* (Omicron)'
        if lin.startswith('B.1.'): return 'B.1.* (pre-Delta)'
        if lin.startswith('XBB'): return 'XBB (recombinant)'
        if lin.startswith('JN') or lin.startswith('EG') or lin.startswith('HK'):
            return 'JN/EG/HK (late Omicron)'
        return 'Other'

    family_palette = {
        'AY.* (Delta sub)':     '#e05c5c',
        'BA.* (Omicron)':       '#5c8de0',
        'B.1.* (pre-Delta)':    '#5cb85c',
        'XBB (recombinant)':    '#e0a85c',
        'JN/EG/HK (late Omicron)': '#a05ce0',
        'Other':                '#aaaaaa',
    }

    xs = np.array([r['entropy_norm'] for r in rows])
    ys = np.array([r['hit_rate'] for r in rows])
    ns = np.array([r['n'] for r in rows])
    log_ns = np.log1p(ns)
    sizes = 50 + (log_ns - log_ns.min()) / (log_ns.max() - log_ns.min() + 1e-9) * 750
    families = [_family(r['lineage']) for r in rows]
    colors = [family_palette[f] for f in families]

    fig, ax = plt.subplots(figsize=(9, 6))
    # ファミリーごとにプロット（凡例のため）
    plotted = set()
    for x, y, s, c, fam in zip(xs, ys, sizes, colors, families):
        lbl = fam if fam not in plotted else None
        ax.scatter(x, y, s=s, c=c, alpha=0.75, edgecolors='gray', linewidths=0.4, label=lbl)
        plotted.add(fam)

    ax.legend(title='Lineage family', fontsize=7, title_fontsize=8,
              loc='upper right', markerscale=0.6)

    # サンプル数上位系統にラベル
    rows_sorted = sorted(rows, key=lambda r: r['n'], reverse=True)
    label_targets = {r['lineage'] for r in rows_sorted[:8]}
    for r in rows:
        if r['lineage'] in label_targets:
            ax.annotate(r['lineage'], (r['entropy_norm'], r['hit_rate']),
                        fontsize=6.5, alpha=0.85,
                        xytext=(4, 2), textcoords='offset points')

    ax.set_xlabel('Normalized Target Position Entropy')
    ax.set_ylabel('Position Hit Rate (%)')
    ax.set_title(f'Entropy vs Position Hit Rate by Lineage ({prefix.upper()})\n'
                 f'Point size ∝ log(num_samples), n≥{min_samples} lineages only')
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-2, 105)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.grid(linestyle='--', alpha=0.35)

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_entropy_vs_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Entropy vs accuracy scatter saved to {path}")


def plot_labeldiversity_vs_accuracy(details, output_dir, prefix="test", min_samples=None):
    """ターゲット位置のユニーク率 vs 正解率の散布図を系統単位で保存する（エントロピーの代替・簡易版）。

    X軸: unique_ratio = ユニークな target 位置数 / target 位置の総出現数（系統内）
         0=少数の位置に集中（易）〜 1=全て異なる位置（多様・難）。
    「何割が異なる位置か」で直感的。ただしサンプル数が多い系統ほど機械的に下がる傾向があるため、
    サンプル数バイアスに強い指標が必要なら plot_simpson_vs_accuracy を併用のこと。
    系統粒度は Pango 上位2階層、min_samples は config.DIVERSITY_PLOT_MIN_SAMPLES に従う。
    Y軸: position hit rate、点サイズ: num_samples（対数）、色: 系統ファミリー。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not details:
        return
    min_samples = _diversity_min_samples(min_samples)
    rows = _lineage_diversity_rows(details, min_samples)
    if not rows:
        return

    def _family(lin):
        if lin.startswith('AY.'):  return 'AY.* (Delta sub)'
        if lin.startswith('BA.'):  return 'BA.* (Omicron)'
        if lin.startswith('B.1.'): return 'B.1.* (pre-Delta)'
        if lin.startswith('XBB'): return 'XBB (recombinant)'
        if lin.startswith('JN') or lin.startswith('EG') or lin.startswith('HK'):
            return 'JN/EG/HK (late Omicron)'
        return 'Other'

    family_palette = {
        'AY.* (Delta sub)':     '#e05c5c',
        'BA.* (Omicron)':       '#5c8de0',
        'B.1.* (pre-Delta)':    '#5cb85c',
        'XBB (recombinant)':    '#e0a85c',
        'JN/EG/HK (late Omicron)': '#a05ce0',
        'Other':                '#aaaaaa',
    }

    xs = np.array([r['unique_ratio'] for r in rows])
    ys = np.array([r['hit_rate'] for r in rows])
    ns = np.array([r['n'] for r in rows])
    log_ns = np.log1p(ns)
    sizes = 50 + (log_ns - log_ns.min()) / (log_ns.max() - log_ns.min() + 1e-9) * 750
    families = [_family(r['lineage']) for r in rows]
    colors = [family_palette[f] for f in families]

    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = set()
    for x, y, s, c, fam in zip(xs, ys, sizes, colors, families):
        lbl = fam if fam not in plotted else None
        ax.scatter(x, y, s=s, c=c, alpha=0.75, edgecolors='gray', linewidths=0.4, label=lbl)
        plotted.add(fam)

    ax.legend(title='Lineage family', fontsize=7, title_fontsize=8,
              loc='upper right', markerscale=0.6)

    rows_sorted = sorted(rows, key=lambda r: r['n'], reverse=True)
    label_targets = {r['lineage'] for r in rows_sorted[:8]}
    for r in rows:
        if r['lineage'] in label_targets:
            ax.annotate(r['lineage'], (r['unique_ratio'], r['hit_rate']),
                        fontsize=6.5, alpha=0.85,
                        xytext=(4, 2), textcoords='offset points')

    ax.set_xlabel('Unique / Total Target Positions per Lineage')
    ax.set_ylabel('Position Hit Rate (%)')
    ax.set_title(f'Label Uniqueness vs Position Hit Rate by Lineage ({prefix.upper()})\n'
                 f'X = unique target positions / total (0=concentrated/easy, 1=all distinct/hard), '
                 f'n≥{min_samples}')
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-2, 105)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.grid(linestyle='--', alpha=0.35)

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_labeldiversity_vs_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Label-uniqueness vs accuracy scatter saved to {path}")


def plot_simpson_vs_accuracy(details, output_dir, prefix="test", min_samples=None):
    """ターゲット位置の多様度 vs 正解率の散布図を系統単位で保存する（エントロピーの代替）。

    多様度は生態学標準の **Gini–Simpson 多様度指数 D = 1 − Σ pᵢ²** を使う。
    「その系統でランダムに2つのターゲット変異を選んだとき、別々の位置になる確率」で、
    生物系の読者に説明不要（Simpson の種多様度指数と同じ）。unique/total と違い
    サンプル数バイアスに強い。
      D → 0 : 少数の位置に集中（予測しやすい）
      D → 1 : 多くの位置に均等に分散（予測が難しい）
    系統粒度は Pango 上位2階層、min_samples は config.DIVERSITY_PLOT_MIN_SAMPLES に従う。
    Y軸: position hit rate、点サイズ: num_samples（対数）、色: 系統ファミリー。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not details:
        return
    min_samples = _diversity_min_samples(min_samples)
    rows = _lineage_diversity_rows(details, min_samples)
    if not rows:
        return

    def _family(lin):
        if lin.startswith('AY.'):  return 'AY.* (Delta sub)'
        if lin.startswith('BA.'):  return 'BA.* (Omicron)'
        if lin.startswith('B.1.'): return 'B.1.* (pre-Delta)'
        if lin.startswith('XBB'): return 'XBB (recombinant)'
        if lin.startswith('JN') or lin.startswith('EG') or lin.startswith('HK'):
            return 'JN/EG/HK (late Omicron)'
        return 'Other'

    family_palette = {
        'AY.* (Delta sub)':     '#e05c5c',
        'BA.* (Omicron)':       '#5c8de0',
        'B.1.* (pre-Delta)':    '#5cb85c',
        'XBB (recombinant)':    '#e0a85c',
        'JN/EG/HK (late Omicron)': '#a05ce0',
        'Other':                '#aaaaaa',
    }

    xs = np.array([r['simpson'] for r in rows])
    ys = np.array([r['hit_rate'] for r in rows])
    ns = np.array([r['n'] for r in rows])
    log_ns = np.log1p(ns)
    sizes = 50 + (log_ns - log_ns.min()) / (log_ns.max() - log_ns.min() + 1e-9) * 750
    families = [_family(r['lineage']) for r in rows]
    colors = [family_palette[f] for f in families]

    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = set()
    for x, y, s, c, fam in zip(xs, ys, sizes, colors, families):
        lbl = fam if fam not in plotted else None
        ax.scatter(x, y, s=s, c=c, alpha=0.75, edgecolors='gray', linewidths=0.4, label=lbl)
        plotted.add(fam)

    ax.legend(title='Lineage family', fontsize=7, title_fontsize=8,
              loc='upper right', markerscale=0.6)

    rows_sorted = sorted(rows, key=lambda r: r['n'], reverse=True)
    label_targets = {r['lineage'] for r in rows_sorted[:8]}
    for r in rows:
        if r['lineage'] in label_targets:
            ax.annotate(r['lineage'], (r['simpson'], r['hit_rate']),
                        fontsize=6.5, alpha=0.85,
                        xytext=(4, 2), textcoords='offset points')

    ax.set_xlabel('Target Position Diversity  (Gini–Simpson  D = 1 − Σ pᵢ²)')
    ax.set_ylabel('Position Hit Rate (%)')
    ax.set_title(f'Target Diversity (Simpson) vs Position Hit Rate by Lineage ({prefix.upper()})\n'
                 f'D=0: concentrated/easy … D→1: spread across many positions/hard, '
                 f'n≥{min_samples}')
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-2, 105)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.grid(linestyle='--', alpha=0.35)

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_diversity_simpson_vs_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Simpson-diversity vs accuracy scatter saved to {path}")


def plot_entropy_bin(entropy_bin_df, output_dir, prefix="test"):
    """エントロピービン別精度の棒グラフを保存する。"""
    import matplotlib.pyplot as plt
    import numpy as np

    if entropy_bin_df is None or entropy_bin_df.empty:
        return

    df = entropy_bin_df[entropy_bin_df['num_samples'] > 0].copy()
    if df.empty:
        return

    tasks = {
        'position': 'Position',
        'region': 'Region',
        'aa_pos': 'AA Position',
        'codon_pos': 'Codon Pos',
        'synonymous': 'Synonymous',
    }
    task_cols = [f'{t}_hit_rate_pct' for t in tasks]
    task_labels = list(tasks.values())

    bins = df['entropy_bin'].tolist()
    x = np.arange(len(bins))
    n_tasks = len(task_cols)
    width = 0.75 / n_tasks

    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (col, label) in enumerate(zip(task_cols, task_labels)):
        offset = (i - n_tasks / 2 + 0.5) * width
        vals = df[col].tolist()
        ax.bar(x + offset, vals, width=width, label=label, color=cmap(i), alpha=0.85)

    # サンプル数を各ビンの上に表示
    for j, (xi, row) in enumerate(zip(x, df.itertuples())):
        n = row.num_samples
        lbl = f'n={n:,}\n({row.num_strains}s)'
        ax.text(xi, 2, lbl, ha='center', va='bottom', fontsize=6, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=30, ha='right', fontsize=8)
    ax.set_xlabel('Normalized Target Position Entropy (bin)')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title(f'Hit Rate by Entropy Bin ({prefix.upper()})\n'
                 f'n=samples, s=strains per bin')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_entropy_bin.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Entropy bin bar chart saved to {path}")


def plot_calibration(calib_bins, output_dir, prefix="test"):
    """Reliability Diagram（キャリブレーション図）を保存する。"""
    import matplotlib.pyplot as plt
    import numpy as np

    if not calib_bins:
        return

    tasks = [t for t in ['region', 'position', 'aa_pos'] if t in calib_bins]
    if not tasks:
        return

    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        bins = [b for b in calib_bins[task]
                if b['avg_confidence'] is not None and b['avg_accuracy'] is not None]
        if not bins:
            continue
        confs = [b['avg_confidence'] for b in bins]
        accs  = [b['avg_accuracy']   for b in bins]
        cnts  = [b['count']          for b in bins]
        max_cnt = max(cnts) if cnts else 1

        # 完璧なキャリブレーション対角線
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')

        sc = ax.scatter(confs, accs, s=[50 + 200 * c / max_cnt for c in cnts],
                        c=cnts, cmap='Blues', alpha=0.85, edgecolors='steelblue', linewidths=0.5)
        ax.plot(confs, accs, color='steelblue', linewidth=1.2, alpha=0.7)

        plt.colorbar(sc, ax=ax, label='Bin count', fraction=0.04, pad=0.02)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Mean predicted confidence')
        ax.set_ylabel('Fraction of positives (accuracy)')
        ax.set_title(f'Reliability Diagram — {task} ({prefix.upper()})')
        ax.legend(fontsize=8)
        ax.grid(linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_calibration.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Calibration plot saved to {path}")


def plot_region_metrics(details, output_dir, prefix="test"):
    """遺伝子領域別の Precision / Recall / F1 を横棒グラフで保存する。"""
    import matplotlib.pyplot as plt
    import numpy as np
    from .. import config as _cfg

    if not details:
        return

    region_names = {v: k for k, v in _cfg.PROTEIN_VOCABS.items()}
    region_stats = {}
    for r in details:
        for reg_id in r.get('targets_region', set()):
            if reg_id not in region_stats:
                region_stats[reg_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'n': 0}
            region_stats[reg_id]['n'] += 1
            if reg_id in r.get('preds_region', set()):
                region_stats[reg_id]['tp'] += 1
            else:
                region_stats[reg_id]['fn'] += 1
        for pred_id in r.get('preds_region', set()):
            if pred_id not in r.get('targets_region', set()):
                if pred_id not in region_stats:
                    region_stats[pred_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'n': 0}
                region_stats[pred_id]['fp'] += 1

    rows = []
    for reg_id, s in region_stats.items():
        prec = s['tp'] / (s['tp'] + s['fp']) * 100 if (s['tp'] + s['fp']) > 0 else 0.0
        rec  = s['tp'] / (s['tp'] + s['fn']) * 100 if (s['tp'] + s['fn']) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({'name': region_names.get(reg_id, f'ID:{reg_id}'),
                     'n': s['n'], 'prec': prec, 'rec': rec, 'f1': f1})

    rows = [r for r in rows if r['n'] > 0]
    rows.sort(key=lambda r: r['rec'], reverse=True)

    labels = [f"{r['name']} (n={r['n']:,})" for r in rows]
    precs  = [r['prec'] for r in rows]
    recs   = [r['rec']  for r in rows]
    f1s    = [r['f1']   for r in rows]

    y     = np.arange(len(rows))
    width = 0.26
    fig_h = max(6, len(rows) * 0.30)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(y - width, precs, height=width, label='Precision', color='#4c72b0', alpha=0.85)
    ax.barh(y,         recs,  height=width, label='Recall',    color='#dd8452', alpha=0.85)
    ax.barh(y + width, f1s,   height=width, label='F1',        color='#55a868', alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Score (%)')
    ax.set_title(f'Region Metrics — Precision / Recall / F1 ({prefix.upper()})\n(sorted by Recall)')
    ax.set_xlim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_region_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Region metrics plot saved to {path}")


def plot_r_precision(topk_results, output_dir, prefix="test"):
    """R-Precision を Top-K と並べて比較する棒グラフを保存する。"""
    import matplotlib.pyplot as plt
    import numpy as np

    if not topk_results:
        return

    tasks = ['region', 'position', 'aa_pos', 'codon_pos', 'synonymous']
    tasks = [t for t in tasks if t in topk_results]

    int_ks = sorted(k for k in next(iter(topk_results.values())) if isinstance(k, int))
    has_rp = any('r_precision' in topk_results[t] for t in tasks)
    if not has_rp:
        return

    labels = [f'K={k}' for k in int_ks] + ['R-Prec']
    n_labels = len(labels)
    n_tasks  = len(tasks)
    width    = 0.8 / n_labels
    colors   = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b2', '#937860'][:n_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(n_tasks)

    for i, (lbl, col) in enumerate(zip(labels, colors)):
        hit_rates  = []
        precisions = []
        for t in tasks:
            if lbl.startswith('K='):
                k = int(lbl[2:])
                m = topk_results[t].get(k, {})
            else:
                m = topk_results[t].get('r_precision', {})
            hit_rates.append(m.get('hit_rate', 0.0))
            precisions.append(m.get('precision', 0.0))
        offset = (i - n_labels / 2 + 0.5) * width
        kw = dict(width=width, color=col, alpha=0.85,
                  linewidth=1.5 if lbl == 'R-Prec' else 0,
                  edgecolor='white' if lbl == 'R-Prec' else col)
        ax1.bar(x + offset, hit_rates,  label=lbl, **kw)
        ax2.bar(x + offset, precisions, label=lbl, **kw)

    for ax, ylabel, title in [
        (ax1, 'Hit Rate (%)',   f'Hit Rate: Top-K vs R-Precision ({prefix.upper()})'),
        (ax2, 'Precision (%)',  f'Precision: Top-K vs R-Precision ({prefix.upper()})'),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_r_precision.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] R-Precision plot saved to {path}")


def plot_attention_heatmap(model, sample_batch, output_dir, prefix="test"):
    """CoOccurrenceAttention と TransformerEncoder の Attention 重みをヒートマップで保存する。

    Hook を用いてモデルを変更せずに内部の Attention 重みを取得する。

    Args:
        model: HierarchicalTransformer
        sample_batch: dataloader の最初の1バッチ ((x_cat, x_num, mask), ...)
        output_dir: 保存先ディレクトリ
        prefix: ファイル名プレフィックス
    """
    import matplotlib.pyplot as plt
    import torch
    import os

    (x_cat, x_num, mask), *_ = sample_batch
    device = next(model.parameters()).device
    x_cat  = x_cat[:1].to(device)  # バッチサイズ1に絞る
    x_num  = x_num[:1].to(device)
    mask   = mask[:1].to(device)

    attn_weights = {}

    def _make_hook(name):
        def hook(module, inp, out):
            # MultiheadAttention の attn_output_weights を取得するため
            # need_weights=True を強制して forward を再呼び出している
            # ここでは out[1] で attention weights が取れる場合のみ保存
            if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                attn_weights[name] = out[1].detach().cpu()
        return hook

    hooks = []
    # CoOccurrenceAttention 内の MultiheadAttention
    if hasattr(model, 'co_attn') and hasattr(model.co_attn, 'attention'):
        h = model.co_attn.attention.register_forward_hook(_make_hook('co_occurrence'))
        hooks.append(h)
    # TransformerEncoder の各層
    for layer_i, layer in enumerate(model.transformer_encoder.layers):
        if hasattr(layer, 'self_attn'):
            h = layer.self_attn.register_forward_hook(_make_hook(f'transformer_layer_{layer_i}'))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        # need_weights を有効にするため MultiheadAttention を一時的に patch
        for m in model.modules():
            if isinstance(m, torch.nn.MultiheadAttention):
                m._need_weights = True
        model(x_cat, x_num, src_key_padding_mask=mask)

    for h in hooks:
        h.remove()

    if not attn_weights:
        print("[WARNING] No attention weights captured (may need need_weights=True in forward)")
        return

    n_plots = len(attn_weights)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), facecolor='#1a1a2e')
    if n_plots == 1:
        axes = [axes]

    for ax, (name, weights) in zip(axes, attn_weights.items()):
        # weights shape: [batch, heads, T_q, T_k] or [batch, T_q, T_k]
        w = weights[0]  # バッチ次元除去
        if w.dim() == 3:
            w = w.mean(0)  # head 方向に平均
        im = ax.imshow(w.numpy(), cmap='viridis', aspect='auto')
        ax.set_facecolor('#16213e')
        ax.set_title(name, color='white', fontsize=10)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        plt.colorbar(im, ax=ax)

    fig.suptitle(f'Attention Weights ({prefix.upper()})', color='white', fontsize=13)
    plt.tight_layout()
    path = _get_plot_path(output_dir, f'{prefix}_attention_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Attention heatmap saved to {path}")
