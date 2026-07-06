"""マルチシード実験エントリポイント。

同一構成を複数シードで学習＋評価し、主要指標を mean±std に集計する。
single run の差が真の改善かノイズかを区別するために使う。

実行ロジック（シードループ・集計）は main.run_multi_seed() に集約。

Usage:
    # config.MULTI_SEED_LIST（デフォルト [42, 1, 7]）を使う
    python -m transformer_260707.scripts.eval.multi_seed
    # シードを明示指定
    python -m transformer_260707.scripts.eval.multi_seed --seeds 42 1 7 123

出力:
    outputs/<...>/results/multi_seed/<timestamp>/multi_seed_summary.json
      - aggregate : 各指標の mean / std / n / values
      - per_seed  : 各シードの run_summary
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description='Multi-seed training/eval harness')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='使用するシード（省略時は config.MULTI_SEED_LIST）')
    args = parser.parse_args()

    from transformer_260707 import config
    from transformer_260707.main import _run_entry

    config.MULTI_SEED = True
    if args.seeds:
        config.MULTI_SEED_LIST = args.seeds

    _run_entry()


if __name__ == '__main__':
    main()
