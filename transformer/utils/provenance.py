# --- utils/provenance.py ---
"""実行ごとの来歴（どのコード・環境・DBで出た結果か）を run 出力に保存する。

config_snapshot.py は設定値のみで、コードの版（git）・環境・DBの状態は残らなかった。
結果と一緒に provenance.json（コミットhash・dirty状態・環境・DBのfingerprint）を保存し、
未コミットの変更があれば code_changes.patch に追跡ファイルのdiffを保存する。
失敗しても学習は止めない（警告のみ）。

注意: `git diff HEAD` は未追跡ファイルを含まない。provenance.json の untracked に一覧を残すが、
再現性のため**実行前にコミットしておく**こと（dirty のままだとコミットhashだけでは再現できない）。
"""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

from . import logging as _log

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(args, cwd):
    r = subprocess.run(['git'] + args, cwd=cwd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"git {' '.join(args)} failed")
    return r.stdout


def _git_info(repo_dir):
    info = {'repo_dir': repo_dir}
    try:
        info['commit'] = _git(['rev-parse', 'HEAD'], repo_dir).strip()
        info['branch'] = _git(['rev-parse', '--abbrev-ref', 'HEAD'], repo_dir).strip()
        # 版の識別: 直近のタグ（無ければ短縮hash）。実験の区切りはgitタグで管理する（日付付きディレクトリは作らない）
        info['describe'] = _git(['describe', '--tags', '--always', '--dirty'], repo_dir).strip()
        lines = [l for l in _git(['status', '--porcelain'], repo_dir).split('\n') if l.strip()]
        info['modified'] = [l[3:] for l in lines if not l.startswith('??')][:200]
        info['untracked'] = [l[3:] for l in lines if l.startswith('??')][:200]
        info['dirty'] = bool(lines)
    except Exception as e:                                   # gitが無い・リポジトリ外 等
        info['error'] = str(e)[:200]
    return info


def collect_provenance(repo_dir=None, db_path=None):
    """来歴情報の dict を返す（副作用なし）。"""
    repo_dir = repo_dir or REPO_ROOT
    out = {'timestamp': datetime.now().isoformat(timespec='seconds'), 'git': _git_info(repo_dir),
           'argv': sys.argv, 'python': sys.version.split()[0], 'platform': platform.platform(),
           'host': platform.node()}
    for mod in ('torch', 'numpy', 'duckdb', 'pandas'):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        import torch
        out['cuda_device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out['cuda_device'] = None
    if db_path is None:
        try:
            from ..db.connection import get_db_path
            db_path = get_db_path()
        except Exception:
            db_path = None
    if db_path and os.path.exists(db_path):
        st = os.stat(db_path)
        out['db'] = {'path': db_path, 'size_bytes': st.st_size,
                     'mtime': datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')}
    return out


def save_run_provenance(output_dir, repo_dir=None, db_path=None, max_patch_bytes=2_000_000):
    """output_dir/provenance.json（と dirty なら code_changes.patch）を保存する。失敗は警告のみ。"""
    try:
        repo_dir = repo_dir or REPO_ROOT
        info = collect_provenance(repo_dir, db_path)
        os.makedirs(output_dir, exist_ok=True)
        if info['git'].get('dirty'):
            patch = _git(['diff', 'HEAD'], repo_dir)
            if len(patch) <= max_patch_bytes:
                with open(os.path.join(output_dir, 'code_changes.patch'), 'w', encoding='utf-8') as f:
                    f.write(patch)
                info['git']['patch_file'] = 'code_changes.patch'
            else:
                info['git']['patch_file'] = f'(skipped: {len(patch)} bytes > {max_patch_bytes})'
        with open(os.path.join(output_dir, 'provenance.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        g = info['git']
        _log.force_print(f"[INFO] provenance saved: commit={str(g.get('commit'))[:8]} "
                         f"dirty={g.get('dirty')} untracked={len(g.get('untracked', []))}")
        return info
    except Exception as e:
        _log.force_print(f"[WARNING] Failed to save provenance: {e}")
        return None
