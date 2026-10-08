"""B2: 前処理の中核 process_strain_features_core_chunked（usher出力 → 特徴量・ラベル）を極小データで検証する。

検証: ラベル(region/position/aa_pos/codon_pos/synonymous)が最終ステップから正しく作られる、
raw_pathが切り詰められない（RAW_PATH_TRUNCATE_LEN=None）/設定時は末尾が残る、除外統計、
入力側特徴量が最終ステップ(ターゲット)を含まない。
"""
import os

import pytest

from transformer import config, preprocess
from transformer.db import feature as F
from transformer.utils.io import load_batch_cache  # noqa: F401  (importできること)
from test_feature_golden import DISSIM, FREQ, PAM, make_state

NSP1 = config.PROTEIN_VOCABS['nsp1']


def _write(base, strain, rows):
    d = base / strain
    d.mkdir(parents=True)
    (d / 'mutation_paths.tsv').write_text('name\tcol2\tpath\n' + '\n'.join(rows) + '\n')


def _run(tmp_path, monkeypatch, rows, **cfg_kw):
    for k, v in cfg_kw.items():
        monkeypatch.setattr(config, k, v, raising=False)
    monkeypatch.setattr(config, 'INCREMENTAL_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(preprocess, 'get_all_data_base_dirs', lambda: [str(tmp_path / 'usher')])
    _write(tmp_path / 'usher', 'X', rows)
    chunk_paths, stats = preprocess.process_strain_features_core_chunked(
        'X', make_state(), FREQ, DISSIM, PAM, {}, 'testhash')
    samples = []
    for p in chunk_paths:
        samples.extend(preprocess.load_strain_cache(p))
    return samples, stats


def test_labels_come_from_the_last_step_and_inputs_exclude_it(tmp_path, monkeypatch):
    samples, stats = _run(tmp_path, monkeypatch, ['S1\tx\tA1T>T2C>G3C'], MAX_CO_OCCURRENCE=5)
    (name, raw, plen, max_co, feats_x, y), = samples
    assert (name, raw, plen, max_co) == ('S1', 'A1T>T2C>G3C', 3, 1)
    assert len(feats_x) == 2                                  # 入力は最終ステップを除く2ステップ
    # 最終ステップ G3C: 直前までに A1T,T2C が適用され codon=TCG(Ser)。G3C → TCC(Ser) = 同義
    assert y == [(NSP1, 3, 1, 3, 1)]                          # (region, position, aa_pos, codon_pos, is_synonymous)
    assert stats['valid_samples'] == 1 and stats['total_raw_lines'] == 1


def test_cooccurring_last_step_yields_multiple_targets(tmp_path, monkeypatch):
    samples, _ = _run(tmp_path, monkeypatch, ['S1\tx\tA1T>T6A,T9A'], MAX_CO_OCCURRENCE=5)
    (_, _, _, max_co, _, y), = samples
    assert max_co == 2 and [t[1] for t in y] == [6, 9]


def test_long_raw_path_is_not_truncated_when_config_is_none(tmp_path, monkeypatch):
    path = '>'.join(['A1T', 'T1A'] * 150)                     # 1,199文字（>500）
    assert len(path) > 500
    samples, _ = _run(tmp_path, monkeypatch, [f'S1\tx\t{path}'], MAX_CO_OCCURRENCE=5, RAW_PATH_TRUNCATE_LEN=None)
    assert samples[0][1] == path                              # 切り詰めなし（2026-07-10のバグの回帰防止）


def test_truncation_keeps_the_tail_when_configured(tmp_path, monkeypatch):
    path = '>'.join(['A1T', 'T1A'] * 20)
    samples, _ = _run(tmp_path, monkeypatch, [f'S1\tx\t{path}'], MAX_CO_OCCURRENCE=5, RAW_PATH_TRUNCATE_LEN=30)
    assert samples[0][1] == path[-30:]                        # 直近(末尾)の履歴を保持
    # ラベルは切り詰め前の完全な履歴から計算される（長さに依らず最終ステップ T1A）
    assert samples[0][5][0][1] == 1


def test_exclusion_statistics(tmp_path, monkeypatch):
    rows = ['ok\tx\tA1T>T2C',                 # 有効
            'toomany\tx\tA1T>T2C,G3C,T6A',   # 共起数 3 > MAX_CO_OCCURRENCE=2
            'short\tx\tA1T',                 # 1ステップのみ（TARGET_LEN以下）
            'bad\tx\tXYZ>A1T',               # 特徴量生成エラー
            'fmt\tonlytwo']                  # 列不足
    samples, st = _run(tmp_path, monkeypatch, rows, MAX_CO_OCCURRENCE=2)
    assert [s[0] for s in samples] == ['ok']
    assert st['valid_samples'] == 1 and st['excluded_cooccur'] == 1 and st['excluded_short_path'] == 1
    assert st['excluded_feature_gen_error'] == 1 and st['excluded_format'] == 1 and st['total_raw_lines'] == 5


def test_each_sample_starts_from_the_reference_genome(tmp_path, monkeypatch):
    """前のサンプルの変異が次のサンプルへ持ち越されない（共有codon_stateの復元）。"""
    rows = ['a\tx\tA1T>T2C', 'b\tx\tA1T>T2C']
    samples, _ = _run(tmp_path, monkeypatch, rows, MAX_CO_OCCURRENCE=5)
    assert samples[0][4] == samples[1][4] and samples[0][5] == samples[1][5]
