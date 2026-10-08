"""パッケージ名は固定の `transformer`（2026-10-08に transformer_261008 から改名）。

版は日付付きディレクトリではなく gitタグ＋provenance.json で管理する。旧名（日付付き）への参照が
コード・テスト・petra に残っていないことを検証する（改名や、旧版からのコピー時の取りこぼし防止）。
"""
import os
import re

from transformer import config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATED = re.compile(r'transformer_\d{6}')
IMPORT_OR_RUN = re.compile(r'(^\s*(from|import)\s+transformer_\d{6})|(-m\s+transformer_\d{6})', re.M)


def _sources():
    for top in ('transformer', 'tests', 'petra'):
        for root, dirs, files in os.walk(os.path.join(REPO, top)):
            dirs[:] = [d for d in dirs if d not in ('__pycache__', 'PETra', 'golden')]
            for f in files:
                if f.endswith(('.py', '.ini', '.sh', '.cfg')):
                    yield os.path.join(root, f)


def test_package_directory_and_output_dir_use_the_fixed_name():
    assert os.path.isdir(os.path.join(REPO, 'transformer'))
    assert not [d for d in os.listdir(REPO) if re.fullmatch(r'transformer_\d{6}', d)]   # 日付付きディレクトリを作らない
    assert config.OUTPUT_DIR == 'outputs/transformer/'


def test_no_import_or_module_run_of_a_dated_package():
    offenders = []
    for path in _sources():
        if path.endswith('test_package_name.py'):
            continue
        for m in IMPORT_OR_RUN.finditer(open(path, encoding='utf-8').read()):
            offenders.append((os.path.relpath(path, REPO), m.group(0).strip()))
    assert offenders == [], offenders


def test_no_hardcoded_dated_output_paths_in_code():
    offenders = []
    for path in _sources():
        if path.endswith('test_package_name.py'):
            continue
        for i, line in enumerate(open(path, encoding='utf-8').read().split('\n'), 1):
            if re.search(r'outputs/transformer_\d{6}', line) and not line.lstrip().startswith('#'):
                offenders.append((os.path.relpath(path, REPO), i))
    # 旧runの結果を参照する例・既定値は許容しない（出力は outputs/transformer/ へ）。許容する場合は理由を添えて除外すること
    assert offenders == [], offenders
