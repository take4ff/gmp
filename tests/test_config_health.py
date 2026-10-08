"""B4 + B6: 設定（config.py）の健全性。

- 名前の誤り: コードが参照する config.X / getattr(config,'X') が config に存在する
  （getattr のデフォルト値付き参照はタイプミスを黙って無視する。274か所）。
- 未配線フラグ: 定義されているのにコードから一度も参照されないフラグは許可リストで管理する
  （新しい空フラグを入れたら気付けるように）。
- ドキュメント: CLAUDE.md の既定値表が config と一致する。
- 既定値の互換: 学習可能パラメータを追加するフラグが既定Trueなら、旧checkpoint互換の登録が必須。
"""
import glob
import os
import re

import pytest
import torch

from transformer_261008 import config
from transformer_261008.model import HierarchicalTransformer
from transformer_261008.scripts.analysis.xai import _xai_common as X

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, 'transformer_261008')


def _py_sources():
    out = {}
    for f in glob.glob(os.path.join(PKG, '**', '*.py'), recursive=True):
        out[f] = open(f, encoding='utf-8').read()
    return out


SOURCES = _py_sources()
OTHER_SRC = '\n'.join(s for f, s in SOURCES.items() if not f.endswith('/config.py'))
DEFINED = set(re.findall(r'^([A-Z][A-Z0-9_]+)\s*=', SOURCES[os.path.join(PKG, 'config.py')], flags=re.M))

# 定義済みだがコードから参照されていない既知のフラグ（未実装スタブ・旧設定の残骸）。
# 新たに増えたらこのテストが失敗する。配線するか、意図した未実装なら理由を添えてここに追加すること。
UNREFERENCED_ALLOWLIST = {
    'ABSTENTION_THRESHOLD', 'BRANCH_SELECTOR_EMBED', 'CACHE_MAX_SIZE', 'ENABLE_LRU_CACHE',
    'ENSEMBLE_N_MODELS', 'ESM2_MODEL_NAME', 'EVESCAPE_CSV', 'HIDDEN_DIM', 'MIN_SEQ_LEN',
    'MODEL_SAVE_DIR', 'STRUCTURE_CSV', 'USE_BRANCH_DISAGGREGATION',
}


def test_names_referenced_in_code_exist_in_config():
    """config.X（属性参照）と getattr(config, 'X', ...) の X が config.py に定義されている。"""
    refs = {}
    for f, src in SOURCES.items():
        if f.endswith('/config.py'):
            continue
        for m in re.finditer(r"getattr\(\s*config\s*,\s*['\"]([A-Z][A-Z0-9_]+)['\"]", src):
            refs.setdefault(m.group(1), f)
        for m in re.finditer(r"\bconfig\.([A-Z][A-Z0-9_]+)\b", src):
            refs.setdefault(m.group(1), f)
    # 実行時に動的に設定される値（walk_forwardが設定する等）は config.py に無くてもよい
    runtime_set = {'WF_PREV_FOLD_CHECKPOINT', 'WF_PREV_CHECKPOINT_OVERRIDE', 'WALK_FORWARD_TRAIN_START',
                   'TEMPORAL_SPLIT_TEST_END', 'TEMPORAL_SPLIT_DATE', 'KNN_DATASTORE_PATH', 'EVAL_X_AXIS'}
    unknown = {n: os.path.relpath(f, REPO) for n, f in refs.items() if n not in DEFINED and n not in runtime_set}
    assert unknown == {}, f"config に定義が無い名前（タイプミスの可能性）: {unknown}"


def test_unreferenced_flags_are_exactly_the_known_stubs():
    unreferenced = {n for n in DEFINED if not re.search(r'\b' + n + r'\b', OTHER_SRC)
                    and not re.search(r"['\"]" + n + r"['\"]", OTHER_SRC)}
    new = unreferenced - UNREFERENCED_ALLOWLIST
    resolved = UNREFERENCED_ALLOWLIST - unreferenced
    assert new == set(), f"新たに未配線のフラグ（配線するか許可リストへ）: {sorted(new)}"
    assert resolved == set(), f"許可リストにあるが既に配線済み（リストから外す）: {sorted(resolved)}"


def test_claude_md_default_table_matches_config():
    path = os.path.join(REPO, 'CLAUDE.md')
    if not os.path.exists(path):
        pytest.skip('CLAUDE.md が無い（.gitignore対象のためCI等では無い）')
    text = open(path, encoding='utf-8').read()
    section = text.split('### 実験切り替え', 1)[1].split('###', 1)[0]
    rows = re.findall(r"\|\s*`([A-Z0-9_]+)`\s*\|\s*`([^`]*)`\s*\|", section)
    assert rows, '実験切り替えの表が読めない'
    mismatch = {}
    for name, doc in rows:
        actual = getattr(config, name)
        shown = repr(actual) if isinstance(actual, str) else str(actual)
        if shown != doc:
            mismatch[name] = (doc, shown)
    assert mismatch == {}, f"CLAUDE.md の既定値表が config と食い違い (表, 実際): {mismatch}"


def _param_keys(**flags):
    saved = {k: getattr(config, k) for k in flags}
    try:
        for k, v in flags.items():
            setattr(config, k, v)
        config.DEVICE = 'cpu'
        torch.manual_seed(0)
        m = HierarchicalTransformer()
        return set(m.state_dict().keys())
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def test_default_true_flags_that_add_parameters_have_old_checkpoint_compat():
    """既定Trueのフラグが学習可能パラメータ/バッファを追加するなら、旧checkpoint（フラグが無い時代）を
    読めるよう _OFF_IF_ABSENT_FROM_SNAPSHOT に登録されていること。
    過去の問題: USE_REGION_CONDITIONED_POSITION が既定Trueになり、旧checkpointが
    'Missing key(s): region_hier_scale, pos_region_map' で読めなくなった（2026-10-06）。"""
    model_src = open(os.path.join(PKG, 'model.py'), encoding='utf-8').read()
    flags = sorted(n for n in dir(config) if n.startswith('USE_') and getattr(config, n) is True
                   and re.search(r'\b' + n + r'\b', model_src))
    base = _param_keys()
    adds_params = [f for f in flags if _param_keys(**{f: False}) != base]
    missing = [f for f in adds_params if f not in X._OFF_IF_ABSENT_FROM_SNAPSHOT]
    assert missing == [], (f"既定Trueでパラメータを追加するが旧checkpoint互換が未登録: {missing}。"
                          f"_xai_common._OFF_IF_ABSENT_FROM_SNAPSHOT に追加すること")
    assert 'USE_REGION_CONDITIONED_POSITION' in adds_params          # 検出ロジック自体の確認
