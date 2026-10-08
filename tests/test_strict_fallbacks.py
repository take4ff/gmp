"""C5: 「フラグはONなのに必要なリソースが無い」ときに、警告だけで続行せず止まること（STRICT_FALLBACKS=True、既定）。

いずれも従来は警告のみで続行し、結果は「もっともらしい値」のまま静かに劣化した:
事前学習checkpoint無し→ランダム初期化 / ホモプラシーCSV無し→機能が無効化 / point-in-time頻度表無し→
codon_freqが0（リーク対策の設計が黙って崩れる）/ kNNデータストア無し / config_snapshot無し→現行configで続行 /
入力CSV・データファイルが読めない→サンプルが欠落。
"""
import pytest

from transformer_261008 import config, preprocess
from transformer_261008.db import dataset as ds
from transformer_261008.model import HierarchicalTransformer
from transformer_261008.scripts.analysis.xai import _xai_common as X
from transformer_261008.utils.knn_output import KNNOutput
from transformer_261008.utils.logging import FallbackError, fallback_or_raise


def test_default_is_strict():
    assert config.STRICT_FALLBACKS is True


def test_helper_raises_when_strict_and_warns_when_lenient(capsys):
    with pytest.raises(FallbackError, match='欠けています'):
        fallback_or_raise('リソースが欠けています', strict=True)
    fallback_or_raise('リソースが欠けています', strict=False)
    assert '[WARNING] リソースが欠けています' in capsys.readouterr().out


def test_helper_follows_config(cfg):
    cfg(STRICT_FALLBACKS=True)
    with pytest.raises(FallbackError):
        fallback_or_raise('x')
    cfg(STRICT_FALLBACKS=False)
    fallback_or_raise('x')


def test_missing_pretrain_checkpoint_stops_instead_of_random_init(tmp_path, cfg):
    from transformer_261008.main import build_components
    cfg(USE_PRETRAINING=True, OUTPUT_DIR=str(tmp_path) + '/', WF_PREV_FOLD_CHECKPOINT=None,
        SPLIT_MODE='walk_forward', DEVICE='cpu', PRETRAINING_MODE='mlm')
    with pytest.raises(FallbackError, match='事前学習checkpoint'):
        build_components(None)
    cfg(STRICT_FALLBACKS=False)
    build_components(None)                                       # 緩和時は従来通りランダム初期化で続行


def test_missing_or_broken_homoplasy_csv_stops(tmp_path, cfg):
    cfg(HOMOPLASY_CSV=str(tmp_path / 'nope.csv'))
    with pytest.raises(FallbackError, match='ホモプラシーCSV'):
        HierarchicalTransformer._load_homoplasy_bias(None, 'USE_HOMOPLASY_PRIOR')
    bad = tmp_path / 'bad.csv'
    bad.write_text('wrong,columns\n1,2\n')
    cfg(HOMOPLASY_CSV=str(bad))
    with pytest.raises(FallbackError, match='読み込みに失敗'):
        HierarchicalTransformer._load_homoplasy_bias(None, 'USE_HOMOPLASY_PRIOR')


def test_missing_point_in_time_table_stops(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(ds, '_POINT_IN_TIME_FREQ_CACHE', {})
    cfg(TEMPORAL_SPLIT_DATE='1999-01-01', POINT_IN_TIME_FREQ_DIR=str(tmp_path))
    with pytest.raises(FallbackError, match='point-in-time'):
        ds._get_point_in_time_freq_dict()
    assert '1999-01-01' not in ds._POINT_IN_TIME_FREQ_CACHE      # 失敗を空dictとしてキャッシュしない
    cfg(STRICT_FALLBACKS=False)
    assert ds._get_point_in_time_freq_dict() == {}               # 緩和時は従来の挙動（空）


def test_existing_point_in_time_tables_cover_every_fold_date():
    """実ファイルの存在確認: 全foldのsplit_dateの表がある（厳格化で学習が止まらない前提）。"""
    import os
    from transformer_261008.scripts.eval.walk_forward import FOLDS
    missing = [sd for _, _, sd, _, _ in FOLDS
               if not os.path.exists(os.path.join(config.POINT_IN_TIME_FREQ_DIR, f'{sd}.csv'))]
    if not os.path.isdir(config.POINT_IN_TIME_FREQ_DIR):
        pytest.skip('reference/ が無い環境')
    assert missing == []


def test_missing_knn_datastore_stops(tmp_path, cfg):
    cfg(KNN_DATASTORE_PATH=str(tmp_path / 'nope'))
    with pytest.raises(FallbackError, match='kNN datastore'):
        KNNOutput.load(10)


def test_missing_config_snapshot_stops(tmp_path):
    with pytest.raises(FallbackError, match='config_snapshot'):
        X.load_config_snapshot(str(tmp_path))


def test_missing_sequences_csv_stops(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(preprocess, 'get_all_sequences_csv_paths', lambda: [str(tmp_path / 'nope.csv')])
    with pytest.raises(FallbackError, match='Sequences CSV'):
        preprocess.load_combined_sequences_df()
    with pytest.raises(FallbackError, match='Sequences CSV'):
        preprocess.load_release_dates()
    cfg(STRICT_FALLBACKS=False)
    assert preprocess.load_release_dates() == {}


def _strain_dir_with_unreadable_file(tmp_path):
    d = tmp_path / 'usher' / 'X'
    d.mkdir(parents=True)
    (d / 'mutation_paths.tsv').mkdir()                          # ファイルでなくディレクトリ → open が失敗する
    return tmp_path / 'usher'


def test_unreadable_data_file_stops_instead_of_silently_dropping(tmp_path, cfg, monkeypatch):
    from test_feature_golden import DISSIM, FREQ, PAM, make_state
    base = _strain_dir_with_unreadable_file(tmp_path)
    monkeypatch.setattr(preprocess, 'get_all_data_base_dirs', lambda: [str(base)])
    cfg(INCREMENTAL_CACHE_DIR=str(tmp_path / 'cache'))
    with pytest.raises(FallbackError, match='データファイルの処理に失敗'):
        preprocess.process_strain_features_core_chunked('X', make_state(), FREQ, DISSIM, PAM, {}, 'h')
    cfg(STRICT_FALLBACKS=False)
    paths, stats = preprocess.process_strain_features_core_chunked('X', make_state(), FREQ, DISSIM, PAM, {}, 'h')
    assert stats['valid_samples'] == 0                           # 緩和時は従来通り除外して続行
