# --- transformer_260707/scripts/analysis/xai/run_all_xai_walk_forward.py ---
"""新規 XAI/生物学的分析スクリプト5本を、walk_forward の全フォールドに対して一括実行する。

run_all_xai.py が単一checkpointに対する一括実行なのに対し、こちらは
`--walk_forward_dir` を各分析スクリプトへそのまま渡す（--checkpoint の代わり）。
各分析スクリプトは全て「fold個別の結果を fold_N/ に保存 → 最後に全fold横断の
集約（合算/n重み付き平均/頻度集計、分析ごとに意味のある方式）を親ディレクトリ直下に保存」
という共通パターンで実装済み（詳細は各スクリプトのdocstring参照）。

対象（run_all_xai.py と同一5本）:
  genome_track                 … ①位置/遺伝子別重要度→ゲノム地図（fold毎に積算→合算）
  homoplasy_alignment          … ②重要度 vs homoplasy 相関（fold毎に積算→合算後に相関計算）
  cooccurrence_constellation   … ③共予測ペア vs 実共起（fold毎に積算→合算）
  probe_representation         … ⑤内部表現の線形プロービング（fold毎に個別プローブ→n重み付き平均）
  local_explain                … ④Integrated Gradients による局所説明（fold毎に個別説明→上位寄与頻度集計）

1本が失敗しても継続する。

Usage:
  python -m transformer_260707.scripts.analysis.xai.run_all_xai_walk_forward \\
      --walk_forward_dir outputs/transformer_260707/results/walk_forward/<timestamp>
  # 一部だけ:  --only genome_track homoplasy_alignment
  # 除外:      --skip local_explain
  # 対象fold限定: --folds 1 3 5
  # 実行せずコマンド確認: --dry_run
  # GPU が埋まっている場合: --force_cpu

注意: split_type_wf は他の walk_forward プロセスと共有のDB列。本体/PETRA双方の他の
walk_forwardプロセス（学習・評価）と同時に実行しないこと。
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

from transformer_260707.utils.logging import force_print

BASE = 'transformer_260707.scripts.analysis.xai'

# 実行順と各スクリプト固有のデフォルト引数（run_all_xai.py と同一）
ANALYSES = {
    'genome_track':               ['--n_batches', '100'],
    'homoplasy_alignment':        ['--n_batches', '100'],
    'cooccurrence_constellation': ['--n_batches', '100', '--top_k', '5'],
    'probe_representation':       ['--n_batches', '150'],
    'local_explain':              ['--n_samples', '8', '--ig_steps', '32'],
}


def main():
    parser = argparse.ArgumentParser(
        description='Run all new XAI/analysis scripts across all walk-forward folds')
    parser.add_argument('--walk_forward_dir', type=str, required=True)
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='対象フォールドを限定する（省略時は発見した全fold）')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--only', nargs='+', default=None, choices=list(ANALYSES),
                        help='これらのみ実行')
    parser.add_argument('--skip', nargs='+', default=None, choices=list(ANALYSES),
                        help='これらを除外')
    parser.add_argument('--n_batches', type=int, default=None,
                        help='--n_batches を持つ分析の共通上書き値')
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
        'outputs', 'transformer_260707', 'scripts', 'xai_all_walk_forward', ts)
    os.makedirs(parent, exist_ok=True)
    force_print(f"[INFO] XAI walk-forward 一括実行 → {parent}\n[INFO] 対象: {', '.join(names)}")

    results = {}
    for name in names:
        extra = list(ANALYSES[name])
        if args.n_batches is not None and '--n_batches' in extra:
            extra[extra.index('--n_batches') + 1] = str(args.n_batches)
        cmd = [sys.executable, '-m', f'{BASE}.{name}',
               '--walk_forward_dir', args.walk_forward_dir,
               '--split', args.split,
               '--output_dir', os.path.join(parent, name)] + extra
        if args.folds:
            cmd += ['--folds'] + [str(f) for f in args.folds]
        if args.force_cpu:
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

    force_print(f"\n{'='*64}\n  XAI walk-forward 一括実行 サマリ\n{'='*64}")
    for n, s in results.items():
        force_print(f"  {n:30s} {s}")
    force_print(f"\n出力ルート: {parent}")
    n_fail = sum(1 for s in results.values() if s.startswith('FAILED'))
    if n_fail:
        force_print(f"[WARNING] {n_fail} 本が失敗しました。上のログを確認してください。")
        sys.exit(1)


if __name__ == '__main__':
    main()
