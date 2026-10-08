"""A7: save_config_copy（実行時overrideがスナップショットに反映されること）。
過去バグ: config.py をそのままコピーしており、walk_forward等の実行時上書きが消えていた（2026-07-29）。"""
import os

from transformer_261008 import config
from transformer_261008.utils.io import save_config_copy


def _load(path):
    ns = {}
    with open(path) as f:
        exec(f.read(), ns)
    return ns


def test_runtime_override_is_reflected(tmp_path, cfg):
    cfg(USE_POINT_IN_TIME_FREQ=True, TEMPORAL_SPLIT_DATE='2099-01-01')
    save_config_copy(str(tmp_path))
    snap = _load(os.path.join(tmp_path, 'models', 'config_snapshot.py'))
    assert snap['USE_POINT_IN_TIME_FREQ'] is True
    assert snap['TEMPORAL_SPLIT_DATE'] == '2099-01-01'


def test_static_default_differs_from_override(tmp_path, cfg):
    """静的な config.py の既定は False（上書きしなければFalseのまま）。"""
    original = config.USE_POINT_IN_TIME_FREQ
    cfg(USE_POINT_IN_TIME_FREQ=not original)
    save_config_copy(str(tmp_path))
    snap = _load(os.path.join(tmp_path, 'models', 'config_snapshot.py'))
    assert snap['USE_POINT_IN_TIME_FREQ'] is (not original)


def test_non_serializable_values_are_excluded(tmp_path, cfg):
    cfg(_dummy_obj=object())            # 先頭'_'は元々除外
    cfg(DUMMY_OBJECT=object(), DUMMY_FUNC=lambda: 1)
    save_config_copy(str(tmp_path))
    text = open(os.path.join(tmp_path, 'models', 'config_snapshot.py')).read()
    assert 'DUMMY_OBJECT' not in text and 'DUMMY_FUNC' not in text
    assert 'import torch' not in text   # モジュールは書き出さない


def test_snapshot_is_valid_python_and_has_core_keys(tmp_path):
    save_config_copy(str(tmp_path))
    snap = _load(os.path.join(tmp_path, 'models', 'config_snapshot.py'))
    for key in ('FEATURE_DIM', 'N_LAYERS', 'MAX_SEQ_LEN', 'USE_HIERARCHICAL_PREDICTION'):
        assert key in snap
