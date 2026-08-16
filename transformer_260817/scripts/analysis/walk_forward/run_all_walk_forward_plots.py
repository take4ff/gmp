# --- transformer_260817/scripts/analysis/walk_forward/run_all_walk_forward_plots.py ---
"""walk_forward（半年次フォールド検証）の全fold横断プロット群を1つの walk_forward_dir に対し一括実行する。

対象:
  position_tolerance_monthly     … 許容誤差付き月別位置精度CSV（チェックポイント再読込・DB使用）
  entropy_accuracy_analysis      … 系統別エントロピー vs 精度、Fano理論上限（CSVのみ）
  overlay_diversity_folds        … 系統別 entropy/unique_ratio/simpson vs 精度、全fold重畳（CSVのみ）
  overlay_nll_folds              … タスク別 held-out NLL の fold横断トレンド（CSVのみ）
  overlay_region_folds           … 遺伝子別 recall の fold横断トレンド（CSVのみ）
  overlay_strength_folds         … 流行度カテゴリ別精度の fold横断トレンド（CSVのみ）
  overlay_confident_subset_folds … 確信度ベース coverage 曲線の fold横断重ね描き（CSVのみ）
  monthly_diversity_accuracy     … 月別精度×多様度3種の色分け＋無色版、全fold接続（チェックポイント再読込・DB使用）

対象外（このスクリプトでは実行しない）:
  plot_position_tolerance_comparison … PETRA側CSV（別パイプライン産出物）が別途必要なため手動実行

各スクリプトは独立CLIなので subprocess で順に実行し、全出力を1つの親ディレクトリ
outputs/.../scripts/walk_forward_plots_all/<timestamp>/<analysis>/ にまとめる。1本が失敗しても継続する。

注意: position_tolerance_monthly / monthly_diversity_accuracy はモデルチェックポイントを
再読み込みしてDB共有カラム(split_type_wf)を書き換えるため、walk_forwardの全fold学習が
完全に終わってから実行すること（学習中の並行実行はデータ分割を破壊する）。他はfold内で
既に保存済みのCSVを読むだけなので、walk_forward実行中でも安全に実行できる。

Usage:
  python -m transformer_260817.scripts.analysis.walk_forward.run_all_walk_forward_plots \\
      --walk_forward_dir outputs/transformer_260817/results/walk_forward/<timestamp>
  # 一部だけ:  --only overlay_diversity_folds overlay_nll_folds
  # 除外:      --skip monthly_diversity_accuracy position_tolerance_monthly
  # 実行せずコマンド確認: --dry_run
  # GPU が埋まっている場合（チェックポイント使用系のみ）: --force_cpu
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

from transformer_260817.utils.logging import force_print

BASE = 'transformer_260817.scripts.analysis.walk_forward'

# name -> (--split を取るか, --force_cpu を取るか, 追加デフォルト引数)
ANALYSES = {
    'position_tolerance_monthly':     {'split': False, 'force_cpu': True,
                                        'extra': ['--tolerances', '0', '5', '10', '50']},
    'entropy_accuracy_analysis':      {'split': True,  'force_cpu': False, 'extra': []},
    'overlay_diversity_folds':        {'split': True,  'force_cpu': False, 'extra': []},
    'overlay_nll_folds':              {'split': True,  'force_cpu': False, 'extra': []},
    'overlay_region_folds':           {'split': True,  'force_cpu': False, 'extra': []},
    'overlay_strength_folds':         {'split': True,  'force_cpu': False, 'extra': []},
    'overlay_confident_subset_folds': {'split': True,  'force_cpu': False, 'extra': []},
    'monthly_diversity_accuracy':     {'split': False, 'force_cpu': True,  'extra': []},
}


def main():
    parser = argparse.ArgumentParser(description='Run all walk-forward cross-fold plot scripts')
    parser.add_argument('--walk_forward_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='--split を持つ分析への共通指定')
    parser.add_argument('--only', nargs='+', default=None, choices=list(ANALYSES),
                        help='これらのみ実行')
    parser.add_argument('--skip', nargs='+', default=None, choices=list(ANALYSES),
                        help='これらを除外')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='親出力ディレクトリ（省略時は自動）')
    parser.add_argument('--dry_run', action='store_true', help='実行せずコマンドのみ表示')
    args = parser.parse_args()

    if not args.dry_run and not os.path.isdir(args.walk_forward_dir):
        raise FileNotFoundError(f"walk_forward_dir not found: {args.walk_forward_dir}")

    names = [n for n in ANALYSES if (args.only is None or n in args.only)
             and (args.skip is None or n not in args.skip)]
    if not names:
        raise ValueError("実行対象がありません（--only/--skip を確認）")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    parent = args.output_dir or os.path.join(
        'outputs', 'transformer_260817', 'scripts', 'walk_forward_plots_all', ts)
    os.makedirs(parent, exist_ok=True)
    force_print(f"[INFO] walk_forward プロット一括実行 → {parent}\n[INFO] 対象: {', '.join(names)}")

    results = {}
    for name in names:
        spec = ANALYSES[name]
        cmd = [sys.executable, '-m', f'{BASE}.{name}',
               '--walk_forward_dir', args.walk_forward_dir,
               '--output_dir', os.path.join(parent, name)] + list(spec['extra'])
        if spec['split']:
            cmd += ['--split', args.split]
        if spec['force_cpu'] and args.force_cpu:
            cmd.append('--force_cpu')

        force_print(f"\n{'='*64}")
        force_print(f"  [{name}]")
        force_print(f"  {' '.join(cmd)}")
        force_print(f"{'='*64}")
        if args.dry_run:
            results[name] = 'DRY_RUN'
            continue
        rc = subprocess.call(cmd)
        results[name] = 'OK' if rc == 0 else f'FAILED (rc={rc})'

    force_print(f"\n{'='*64}\n  walk_forward プロット一括実行 サマリ\n{'='*64}")
    for n, s in results.items():
        force_print(f"  {n:32s} {s}")
    force_print(f"\n出力ルート: {parent}")
    force_print("[INFO] plot_position_tolerance_comparison は対象外です"
                "（PETRA側の月別CSVが別途必要なため手動実行してください）")
    n_fail = sum(1 for s in results.values() if s.startswith('FAILED'))
    if n_fail:
        force_print(f"[WARNING] {n_fail} 本が失敗しました。上のログを確認してください。")
        sys.exit(1)


if __name__ == '__main__':
    main()
