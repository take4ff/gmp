"""共通fixture。configはモジュールグローバルなので、書き換えは必ず monkeypatch 経由で行う
（テスト終了時に自動で元へ戻る）。さらに autouse の _restore_config が、setattr で直接書き換える
コード（例: load_config_snapshot）による漏れも全テスト後に復元する。"""
import pickle

import duckdb
import pytest

from transformer_260817 import config
from transformer_260817.db.connection import init_db


@pytest.fixture(autouse=True)
def _restore_config():
    before = dict(vars(config))
    yield
    for k in list(vars(config)):
        if k not in before:
            delattr(config, k)
    for k, v in before.items():
        if getattr(config, k, object()) is not v:
            setattr(config, k, v)


@pytest.fixture
def cfg(monkeypatch):
    """config を安全に書き換えるためのヘルパ: cfg(NAME=value, ...)"""
    def _set(**kwargs):
        for k, v in kwargs.items():
            monkeypatch.setattr(config, k, v, raising=False)
        return config
    return _set


# ---- 合成DuckDB（極小）-------------------------------------------------------
# (sample_id, strain_id, raw_path, collection_date, split_type_wf, targets[(region,pos,aa,codon,syn)])
# グループ（同一 input_path_str）: G1={1,2,3}, G2={4,5}, G3={6}; test: G4={7}, G5={8}
SYNTH_SAMPLES = [
    (1, 1, 'A10T>C20G>T30A',       '2021-03-01', 0, [(1, 100, 10, 1, 0)]),
    (2, 1, 'A10T>C20G>G40C,T50A',  '2021-03-15', 0, [(2, 200, 20, 2, 1), (3, 210, 21, 3, 0)]),
    (3, 1, 'A10T>C20G>A60G',       '2021-04-01', 0, [(4, 300, 30, 1, 1)]),
    (4, 2, 'G1T>A2C>C3G',          '2021-04-10', 0, [(5, 400, 40, 2, 0)]),
    (5, 2, 'G1T>A2C>T4A',          '2021-05-01', 0, [(6, 500, 50, 3, 1)]),
    (6, 2, 'C5A>G6T>A7C',          '2021-05-20', 0, [(7, 600, 60, 1, 0)]),
    (7, 1, 'G1T>A2C>C3G',          '2021-08-01', 2, [(5, 400, 40, 2, 0)]),   # train(4)と完全一致のraw_path
    (8, 2, 'T9C>A8G>C7T',          '2021-09-01', 2, [(8, 700, 70, 1, 1)]),
]
SYNTH_STRAINS = [(1, 'A', 5.0), (2, 'B', 7.0)]


def build_synthetic_db(path, samples=None):
    """init_db と同じスキーマの極小DuckDBを作る。特徴量は各タイムステップ1変異（値は小さい整数）。"""
    samples = SYNTH_SAMPLES if samples is None else samples
    init_db(path)
    con = duckdb.connect(path)
    for sid, name, strength in SYNTH_STRAINS:
        con.execute("INSERT INTO strains (strain_id, strain_name, strain_name_ncbi, strain_name_usher,"
                    " strength_score_ncbi, strength_score_usher, strength_score) VALUES (?,?,?,?,?,?,?)",
                    [sid, name, name, name, strength, strength, strength])
    fid = 0
    for i, (sid, strain, raw, date, split, targets) in enumerate(samples):
        steps = raw.split('>')
        max_co = max(len(s.split(',')) for s in steps)
        con.execute("INSERT INTO samples (sample_id, strain_id, raw_path, path_length, max_cooccurrence,"
                    " split_type, split_type_date, split_type_wf, strength_score, collection_date)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [sid, strain, raw, len(steps), max_co, split, split, split, 5.0, date])
        for ts in range(len(steps) - 1):                      # 入力側タイムステップ（末尾はターゲット）
            for co in range(len(steps[ts].split(','))):
                fid += 1
                cat = [(sid + ts + co + k) % 2 for k in range(config.NUM_FEATURE_STRING)]
                num = [0.1 * ((sid + k) % 5) for k in range(config.NUM_CHEM_FEATURES)]
                con.execute("INSERT INTO features VALUES (?,?,?,?,?,?)",
                            [fid, sid, ts, co, pickle.dumps(cat), pickle.dumps(num)])
        con.execute("INSERT INTO labels VALUES (?,?,?)", [sid, sid, pickle.dumps(targets)])
    con.close()
    return path


@pytest.fixture
def synthetic_db(tmp_path, cfg):
    """合成DBのパスを返し、config を合成DB向けに設定する（DB_FILE・split列・サンプリング等）。"""
    path = str(tmp_path / 'synth.duckdb')
    build_synthetic_db(path)
    cfg(DB_FILE=path, SPLIT_MODE='walk_forward', USE_SUBSTITUTION_HEAD=False, STRENGTH_SOURCE='usher',
        USE_UNIQUE_FILTER=False, USE_POINT_IN_TIME_FREQ=False, DEVICE='cpu', DATALOADER_PIN_MEMORY=False,
        NUM_DATALOADER_WORKERS=0, MAX_GROUP_MEMBERS_FOR_CACHE=None, USE_TRAIN_ENTROPY_FILTER=False,
        USE_TRAIN_STRENGTH_FILTER=False, MAX_STRAIN_NUM=10000, SAMPLING_MODE='proportional')
    return path
