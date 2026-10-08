"""B3: 全モジュールがimportでき、importに副作用が無いこと。

AIによる一括置換・リファクタ・バージョンコピーの後に、めったに実行しないスクリプトの壊れ
（ImportError・未定義名・importで解析が走る等）を検知する。1プロセスで約1.5秒。
"""
import ast
import importlib
import io
import contextlib
import os
import pkgutil

import transformer_261008 as pkg

PKG_DIR = os.path.dirname(pkg.__file__)


def _modules():
    return [m.name for m in pkgutil.walk_packages(pkg.__path__, 'transformer_261008.')
            if not m.name.endswith('.__main__')]


def test_every_module_imports():
    bad = []
    for name in _modules():
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                importlib.import_module(name)
        except BaseException as e:               # SystemExit等も検知
            bad.append((name, type(e).__name__, str(e)[:100]))
    assert bad == [], bad


def test_scripts_with_main_have_a_main_guard():
    """def main を持つスクリプトは `if __name__ == '__main__'` ガードを持つ（importで実行されない）。"""
    missing = []
    for root, _, files in os.walk(os.path.join(PKG_DIR, 'scripts')):
        for f in files:
            if not f.endswith('.py') or f == '__init__.py':
                continue
            src = open(os.path.join(root, f), encoding='utf-8').read()
            tree = ast.parse(src)
            has_main = any(isinstance(n, ast.FunctionDef) and n.name == 'main' for n in tree.body)
            if has_main and '__main__' not in src:
                missing.append(os.path.relpath(os.path.join(root, f), PKG_DIR))
    assert missing == [], missing


def test_no_module_level_executable_statements_in_scripts():
    """scripts/ の各ファイルのモジュール直下に、import・定義・定数代入・docstring以外の実行文を置かない
    （例: 旧 mutation_freq.py は直下で全データを読む解析を実行しimportで固まった）。"""
    allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
               ast.Assign, ast.AnnAssign, ast.If, ast.Try, ast.AugAssign)
    offenders = []
    for root, _, files in os.walk(os.path.join(PKG_DIR, 'scripts')):
        for f in files:
            if not f.endswith('.py'):
                continue
            tree = ast.parse(open(os.path.join(root, f), encoding='utf-8').read())
            for n in tree.body:
                is_doc = isinstance(n, ast.Expr) and isinstance(getattr(n, 'value', None), ast.Constant)
                # matplotlib.use('Agg')（バックエンド指定）はimport時の定型で、副作用は無い
                is_mpl_use = (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                              and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'use'
                              and getattr(n.value.func.value, 'id', None) == 'matplotlib')
                if not (isinstance(n, allowed) or is_doc or is_mpl_use):
                    offenders.append((os.path.relpath(os.path.join(root, f), PKG_DIR), n.lineno, type(n).__name__))
    assert offenders == [], offenders
