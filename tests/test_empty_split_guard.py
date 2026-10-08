"""A1: 空splitで学習・評価が「成功」しないこと。

再現した問題（2026-10-08）: trainが0件でも、DataLoaderは0バッチ・train_one_epochは例外なく loss=0 を返し、
割当が壊れたまま20時間の学習が進みうる。
"""
import duckdb
import pytest
import torch

from transformer.db.dataset import EmptySplitError, create_db_dataloader
from transformer.train import train_one_epoch


def _clear_split(path, split_value):
    con = duckdb.connect(path)
    con.execute(f"UPDATE samples SET split_type_wf = -1 WHERE split_type_wf = {split_value}")
    con.close()


def test_empty_train_split_raises(synthetic_db):
    _clear_split(synthetic_db, 0)
    with pytest.raises(EmptySplitError, match='0件'):
        create_db_dataloader(synthetic_db, 0, 4, max_cooccurrence=20, num_workers_override=0)


def test_empty_valid_split_raises_too(synthetic_db):
    # 合成DBには valid(1) が元々無い（assign_wf_splitsを通していないため）
    with pytest.raises(EmptySplitError):
        create_db_dataloader(synthetic_db, 1, 4, max_cooccurrence=20, num_workers_override=0)


def test_allow_empty_opt_in(synthetic_db):
    _clear_split(synthetic_db, 0)
    loader = create_db_dataloader(synthetic_db, 0, 4, max_cooccurrence=20, num_workers_override=0,
                                  allow_empty=True)
    assert len(loader.dataset) == 0 and sum(1 for _ in loader) == 0


def test_nonempty_split_is_unaffected(synthetic_db):
    loader = create_db_dataloader(synthetic_db, 0, 4, max_cooccurrence=20, num_workers_override=0)
    assert len(loader.dataset) == 3


def test_error_message_points_to_the_known_pitfall(synthetic_db):
    _clear_split(synthetic_db, 0)
    with pytest.raises(EmptySplitError, match='assign_fold_test_window'):
        create_db_dataloader(synthetic_db, 0, 4, max_cooccurrence=20, num_workers_override=0)


def test_train_one_epoch_raises_when_no_batch_processed(synthetic_db):
    _clear_split(synthetic_db, 0)
    loader = create_db_dataloader(synthetic_db, 0, 4, max_cooccurrence=20, num_workers_override=0,
                                  allow_empty=True)
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(RuntimeError, match='1バッチも処理'):
        train_one_epoch(model, loader, opt, loss_fn=None, loss_wrapper=None)
