# --- transformer_260817/scripts/eval/build_knn_datastore.py ---
"""kNN検索拡張出力（USE_KNN_OUTPUT）用のデータストアを構築して保存する。

学習済みcheckpoint（walk_forwardのfold）に対し、そのfoldのtrain split の代表サンプル
（共起グループ単位）のエンコーダ表現と位置分布を保存する。モデルの再学習は不要
（順伝播のみ）。規模が大きいので --max_groups で無作為抽出するパイロットから始めること
（fold_3 の train 全体は約184万群＝前方パスのみで約2.7時間の見積もり）。

注意: データストアは fold ごとに別物（そのfoldのtrainで作る）。出力先は config.KNN_DATASTORE_PATH
（既定 cache/knn_datastore_*.npy）で、--datastore_path で上書きできる。評価時は同じパスを
config.KNN_DATASTORE_PATH に設定し、USE_KNN_OUTPUT=True にする。

Usage:
  python -m transformer_260817.scripts.eval.build_knn_datastore \\
      --checkpoint outputs/transformer_260723/results/walk_forward/<ts>/fold_3/<ts2>/models/best_model.pth \\
      --max_groups 300000 --datastore_path cache/knn_datastore_fold3
"""
import argparse
import random
import re

from transformer_260817 import config
from transformer_260817.db.connection import get_db_path
from transformer_260817.scripts.analysis.xai import _xai_common as X
from transformer_260817.utils.knn_output import build_datastore
from transformer_260817.utils.logging import force_print


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--max_groups', type=int, default=300000,
                    help='データストアに入れる共起グループ数の上限（train splitから無作為抽出）')
    ap.add_argument('--datastore_path', default=None, help='保存先のベース（省略時 config.KNN_DATASTORE_PATH）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force_cpu', action='store_true')
    args = ap.parse_args()

    m = re.search(r'fold_(\d+)', args.checkpoint)
    fold_id = int(m.group(1)) if m else None
    db_path = get_db_path()
    if fold_id is not None:
        # train split(0) を使うため、test専用の軽量版 assign_fold_test_window ではなく
        # assign_wf_splits でそのfoldの train/valid/test を全て割り当てる
        # （軽量版は test 以外を -1 に戻すため train が空になる。2026-10-07に空データストアで発覚）。
        # load_model_and_loader がDataLoader構築時にsplit列を読むため、構築前に割り当てる。
        from transformer_260817.db.connection import connect_db
        from transformer_260817.db.queries import assign_wf_splits
        train_start, split_date, split_end = X.get_fold_windows()[fold_id]
        config.WALK_FORWARD_TRAIN_START = train_start
        config.TEMPORAL_SPLIT_DATE = split_date
        config.TEMPORAL_SPLIT_TEST_END = split_end
        con = connect_db(db_path)
        assign_wf_splits(con)
        con.close()
    model, loader, device = X.load_model_and_loader(args.checkpoint, split='train',
                                                    force_cpu=args.force_cpu)
    if len(loader.dataset.sample_ids) == 0:
        raise SystemExit("[ERROR] train split が0件です（split_type_wf の割り当てを確認してください）")

    ds = loader.dataset
    if args.max_groups is not None and len(ds.sample_ids) > args.max_groups:
        rng = random.Random(args.seed)
        ds.sample_ids = sorted(rng.sample(ds.sample_ids, args.max_groups))
        ds._length = len(ds.sample_ids)
        force_print(f"[INFO] train群を無作為抽出: {args.max_groups:,} 群 (seed={args.seed})")

    if args.datastore_path:
        config.KNN_DATASTORE_PATH = args.datastore_path
    store = build_datastore(model, loader, device, max_groups=args.max_groups)
    if len(store) == 0:
        raise SystemExit("[ERROR] データストアが0件のため保存しません（ラベル/splitを確認してください）")
    store.save()


if __name__ == '__main__':
    main()
