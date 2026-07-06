# --- scripts/preprocess/generate_homoplasy_prior.py ---
# 提案10（ホモプラシー事前分布）用の site_recurrence.csv を生成する。
#
# ホモプラシー = 系統樹上で同一ゲノム座標に独立に繰り返し発生する変異。
# UShER の MAT ツリー（.pb）から matUtils で「各ノード（枝）に割り当てられた変異」を
# 取り出し、座標ごとの独立発生回数（＝枝の本数）を数えて recurrence_count とする。
#
# 生成物: reference/homoplasy/site_recurrence.csv  （列: position_id,recurrence_count）
#   - position_id はゲノム塩基座標そのもの。model の Position トークン ID と一致するため
#     変換不要（db/feature.py: position_id = int(mutation[1:-1])）。
#   - recurrence_count は生の枝本数。model 側は log(1+count) に変換して使う
#     （model.py:_load_homoplasy_bias）ため、ここでは対数化しない。
#
# 2 段構成:
#   Stage 1: matUtils extract -A で全ノードの枝変異を書き出す（usher 環境の matUtils を使用）
#   Stage 2: そのテキストを走査して座標別に集計し CSV 出力（環境非依存・pure python）
#
# 使い方:
#   # 通常（matUtils を自動実行。matUtils が PATH に無ければ conda run -n usher で呼ぶ）
#   conda activate usher   # もしくは gvp25-05 のまま（--conda-env usher で委譲）
#   python -m transformer_260707.scripts.preprocess.generate_homoplasy_prior
#
#   # 既に matUtils extract -A の出力がある場合は再走査を省略
#   python -m transformer_260707.scripts.preprocess.generate_homoplasy_prior \
#       --node-mutations /path/to/all_node_mutations.txt

import argparse
import os
import re
import shutil
import subprocess
import sys

from ... import config

_MUT_RE = re.compile(r'^([ACGT])(\d+)([ACGT])$')


def _default_pb():
    return os.path.join('reference', 'phylogeny', 'public-2024-10-17.all.masked.pb')


def _run_matutils_all_paths(pb_path, out_txt, conda_env):
    """matUtils extract -A を実行して全ノードの枝変異テキストを生成する。"""
    matutils = shutil.which('matUtils')
    out_dir = os.path.dirname(os.path.abspath(out_txt)) or '.'
    out_name = os.path.basename(out_txt)
    os.makedirs(out_dir, exist_ok=True)

    if matutils is not None:
        cmd = [matutils, 'extract', '-i', pb_path, '-d', out_dir, '-A', out_name]
    else:
        # matUtils が現在の環境に無い場合、conda run で usher 環境へ委譲する。
        print(f"[INFO] matUtils not on PATH; delegating to 'conda run -n {conda_env}'")
        cmd = ['conda', 'run', '-n', conda_env, 'matUtils',
               'extract', '-i', pb_path, '-d', out_dir, '-A', out_name]

    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if not os.path.exists(out_txt):
        raise FileNotFoundError(f"matUtils did not produce expected output: {out_txt}")
    return out_txt


def _count_recurrence(node_mutations_txt, max_position):
    """全ノード枝変異テキストを走査し、座標別の独立発生回数を数える。

    各行は 'NAME: MUT1,MUT2,...'（変異なしのノードは 'NAME: '）。
    変異は SNP のみ（[ACGT]<pos>[ACGT]）。ref/alt が N 等の非 ACGT は除外する。
    """
    counts = {}
    n_lines = n_events = n_skipped = 0
    with open(node_mutations_txt, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            n_lines += 1
            idx = line.find(': ')
            if idx < 0:
                continue
            payload = line[idx + 2:].strip()
            if not payload:
                continue
            for tok in payload.split(','):
                m = _MUT_RE.match(tok.strip())
                if not m:
                    n_skipped += 1
                    continue
                pos = int(m.group(2))
                if 1 <= pos < max_position:
                    counts[pos] = counts.get(pos, 0) + 1
                    n_events += 1
                else:
                    n_skipped += 1
    print(f"[INFO] Scanned {n_lines:,} nodes; counted {n_events:,} branch mutation events "
          f"across {len(counts):,} sites (skipped {n_skipped:,} non-SNP/out-of-range tokens)")
    return counts


def _write_csv(counts, out_csv, min_count):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or '.', exist_ok=True)
    rows = sorted((p, c) for p, c in counts.items() if c >= min_count)
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('position_id,recurrence_count\n')
        for pos, c in rows:
            f.write(f'{pos},{c}\n')
    print(f"[INFO] Wrote {len(rows):,} sites (min_count={min_count}) -> {out_csv}")
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("[INFO] Top recurrent sites (position: count):")
    for pos, c in top:
        print(f"        {pos}: {c:,}")


def main():
    ap = argparse.ArgumentParser(description="Generate homoplasy site_recurrence.csv from a UShER MAT.")
    ap.add_argument('--pb', default=_default_pb(),
                    help="Input MAT .pb (default: reference/phylogeny/public-2024-10-17.all.masked.pb)")
    ap.add_argument('--node-mutations', default=None,
                    help="Pre-generated 'matUtils extract -A' text. If given, Stage 1 is skipped.")
    ap.add_argument('--out', default=getattr(config, 'HOMOPLASY_CSV', 'reference/homoplasy/site_recurrence.csv'),
                    help="Output CSV path (default: config.HOMOPLASY_CSV)")
    ap.add_argument('--conda-env', default='usher',
                    help="Conda env with matUtils, used only if matUtils is not on PATH (default: usher)")
    ap.add_argument('--min-count', type=int, default=1,
                    help="Only write sites with recurrence_count >= this (default: 1)")
    ap.add_argument('--keep-node-mutations', action='store_true',
                    help="Keep the intermediate node-mutations text (otherwise it is removed if auto-generated)")
    args = ap.parse_args()

    max_position = getattr(config, 'VOCAB_SIZE_POSITION', 30006)

    auto_generated = False
    node_txt = args.node_mutations
    if node_txt is None:
        if not os.path.exists(args.pb):
            print(f"[ERROR] MAT file not found: {args.pb}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        node_txt = os.path.join(config.CACHE_DIR, 'all_node_mutations.txt')
        _run_matutils_all_paths(args.pb, node_txt, args.conda_env)
        auto_generated = True
    elif not os.path.exists(node_txt):
        print(f"[ERROR] --node-mutations file not found: {node_txt}", file=sys.stderr)
        sys.exit(1)

    counts = _count_recurrence(node_txt, max_position)
    _write_csv(counts, args.out, args.min_count)

    if auto_generated and not args.keep_node_mutations:
        try:
            os.remove(node_txt)
            print(f"[INFO] Removed intermediate {node_txt}")
        except OSError:
            pass


if __name__ == '__main__':
    main()
