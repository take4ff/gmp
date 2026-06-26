"""半年次 Walk-forward 検証エントリポイント。

フォールド定義と CLI 引数のみを管理する。
実行ロジック（config パッチ・学習・評価・集計）は main._run_entry() に集約。

Usage:
    python -m transformer_260625.scripts.eval.walk_forward
    python -m transformer_260625.scripts.eval.walk_forward --folds 2 6
"""

import argparse
import sys

# ──────────────────────────────────────────────
# フォールド定義（半年刻み、2021 〜 2024）
# Fold N: train < split_date, test in [split_date, split_end)
# ──────────────────────────────────────────────
FOLDS = [
    # (fold_id, split_date,    split_end,     description)
    (1, '2021-01-01', '2021-07-01', 'Pre-Alpha era'),
    (2, '2021-07-01', '2022-01-01', 'Delta→Omicron transition'),
    (3, '2022-01-01', '2022-07-01', 'Omicron early (BA.1/BA.2)'),
    (4, '2022-07-01', '2023-01-01', 'Omicron variants (BA.4/BA.5/BQ.1)'),
    (5, '2023-01-01', '2023-07-01', 'XBB era'),
    (6, '2023-07-01', '2024-01-01', 'JN.1 transition'),
    (7, '2024-01-01', None,         'Most recent (no upper bound)'),
]


def main():
    parser = argparse.ArgumentParser(description='Semi-annual walk-forward validation')
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='実行するフォールド番号（省略時: config.WALK_FORWARD_FOLDS or 全フォールド）')
    args = parser.parse_args()

    from transformer_260625 import config
    from transformer_260625.main import _run_entry

    config.SPLIT_MODE = 'walk_forward'
    if args.folds:
        config.WALK_FORWARD_FOLDS = args.folds

    _run_entry()


if __name__ == '__main__':
    main()
