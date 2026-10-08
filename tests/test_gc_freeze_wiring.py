"""A10: create_db_dataloader が gc.freeze() を呼ぶこと（COW崩壊対策の配線の回帰防止、2026-10-05）。"""
import gc

import torch
from torch.utils.data import IterableDataset

from transformer.db import dataset as ds_mod


class _DummyDataset(IterableDataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        return 1                      # 実クラス DBIterableDataset と同様に件数を持つ（空splitガード対象）

    def __iter__(self):
        return iter([])


def test_create_db_dataloader_freezes_gc(monkeypatch, cfg):
    monkeypatch.setattr(ds_mod, 'DBIterableDataset', _DummyDataset)
    cfg(DEVICE='cpu', DATALOADER_PIN_MEMORY=False)
    gc.unfreeze()
    assert gc.get_freeze_count() == 0
    try:
        ds_mod.create_db_dataloader('dummy.duckdb', 0, 4, num_workers_override=0)
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()
