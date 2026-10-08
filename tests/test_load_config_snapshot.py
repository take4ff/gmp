"""A12: load_config_snapshot（旧checkpointのスナップショット互換）。

過去の問題: USE_REGION_CONDITIONED_POSITION が既定Trueになった後、このフラグが存在しない時代の
checkpointを読むと state_dict のキー不足（region_hier_scale, pos_region_map）で失敗した（2026-10-06）。
"""
from transformer.scripts.analysis.xai import _xai_common as X


def _write(tmp_path, body):
    (tmp_path / 'config_snapshot.py').write_text(body)
    return str(tmp_path)


def test_flag_absent_in_old_snapshot_is_turned_off(tmp_path, cfg):
    cfg(USE_REGION_CONDITIONED_POSITION=True)
    X.load_config_snapshot(_write(tmp_path, "FEATURE_DIM = 256\nN_LAYERS = 4\n"))
    from transformer import config
    assert config.USE_REGION_CONDITIONED_POSITION is False


def test_substitution_head_flag_absent_in_old_snapshot_is_turned_off(tmp_path, cfg):
    """base_after/aa_afterヘッド（パラメータ追加）が無い時代のcheckpointも読めるようにFalseへ戻す。"""
    cfg(USE_SUBSTITUTION_HEAD=True)
    X.load_config_snapshot(_write(tmp_path, "FEATURE_DIM = 256\n"))
    from transformer import config
    assert config.USE_SUBSTITUTION_HEAD is False


def test_flag_present_in_snapshot_is_respected(tmp_path, cfg):
    from transformer import config
    cfg(USE_REGION_CONDITIONED_POSITION=False)
    X.load_config_snapshot(_write(tmp_path, "USE_REGION_CONDITIONED_POSITION = True\n"))
    assert config.USE_REGION_CONDITIONED_POSITION is True


def test_other_values_are_overridden_from_snapshot(tmp_path, cfg):
    from transformer import config
    cfg(N_LAYERS=4)
    X.load_config_snapshot(_write(tmp_path, "N_LAYERS = 7\n"))
    assert config.N_LAYERS == 7


def test_missing_snapshot_file_leaves_config_untouched_if_lenient(tmp_path, cfg):
    from transformer import config
    cfg(USE_REGION_CONDITIONED_POSITION=True, STRICT_FALLBACKS=False)
    X.load_config_snapshot(str(tmp_path))          # ファイル無し
    assert config.USE_REGION_CONDITIONED_POSITION is True
