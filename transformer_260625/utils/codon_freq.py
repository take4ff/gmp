"""
utils/codon_freq.py
===================
コドン別変異頻度の計算ユーティリティ

【集計方式】
  タイムステップ重複除外方式（table_set.csv と同じ哲学）をコドン遷移レベルに拡張。

  各タイムステップ t について、全サンプルの t 番目のステップから
  ユニークな (codon_before, codon_after, region) の集合を作り、
  全タイムステップの集合を合算する。

  → 同一の系統枝で発生したコドン遷移は1回のみカウントされる
  → 異なるタイムステップで独立に発生した場合のみ2以上にカウント
  （系統樹上の独立発生イベント数の近似）

【データソース】
  /mnt/ssd1/home3/aiba/usher_output/<strain>/<num>/mutation_paths.tsv
  - 1行: name \\t length \\t mutation_path
  - mutation_path: "C241T>A23403G>C3037T,G204A>T19524C" （>でステップ区切り、,で共起変異）

【使い方】
  from transformer_260625.utils.codon_freq import compute_codon_freq, save_codon_freq

  freq = compute_codon_freq(
      usher_dir="/mnt/ssd1/home3/aiba/usher_output",
      codon_csv="reference/codon/codon_mutation4.csv",
      max_cooccur=10,
  )
  save_codon_freq(freq, "outputs/codon_freq/codon_freq.csv")

  nohup conda run -n gvp25-05 python3 -m transformer_260625.utils.codon_freq \
  --usher_dir /mnt/ssd1/home3/aiba/usher_output \
  --codon_csv reference/codon/codon_mutation4.csv \
  --output outputs/codon_freq/codon_freq.csv > nohup_codon.out &
"""

import os
import csv
import time
import math
from collections import defaultdict, Counter
from copy import deepcopy

import pandas as pd

# ローカルモジュール
from ..utils.logging import force_print


# ==============================================================
# DNA → アミノ酸変換テーブル（legacy/dataset.py と共通）
# ==============================================================
DNA2Protein = {
    'TTT': 'F', 'TCT': 'S', 'TAT': 'Y', 'TGT': 'C',
    'TTC': 'F', 'TCC': 'S', 'TAC': 'Y', 'TGC': 'C',
    'TTA': 'L', 'TCA': 'S', 'TAA': '*', 'TGA': '*',
    'TTG': 'L', 'TCG': 'S', 'TAG': '*', 'TGG': 'W',
    'CTT': 'L', 'CCT': 'P', 'CAT': 'H', 'CGT': 'R',
    'CTC': 'L', 'CCC': 'P', 'CAC': 'H', 'CGC': 'R',
    'CTA': 'L', 'CCA': 'P', 'CAA': 'Q', 'CGA': 'R',
    'CTG': 'L', 'CCG': 'P', 'CAG': 'Q', 'CGG': 'R',
    'ATT': 'I', 'ACT': 'T', 'AAT': 'N', 'AGT': 'S',
    'ATC': 'I', 'ACC': 'T', 'AAC': 'N', 'AGC': 'S',
    'ATA': 'I', 'ACA': 'T', 'AAA': 'K', 'AGA': 'R',
    'ATG': 'M', 'ACG': 'T', 'AAG': 'K', 'AGG': 'R',
    'GTT': 'V', 'GCT': 'A', 'GAT': 'D', 'GGT': 'G',
    'GTC': 'V', 'GCC': 'A', 'GAC': 'D', 'GGC': 'G',
    'GTA': 'V', 'GCA': 'A', 'GAA': 'E', 'GGA': 'G',
    'GTG': 'V', 'GCG': 'A', 'GAG': 'E', 'GGG': 'G',
    'nnn': 'n',
}


# ==============================================================
# 1. コドン参照データのロード
# ==============================================================

def load_codon_template(codon_csv: str) -> dict:
    """
    codon_mutation4.csv を読み込み、legacy/dataset.py と同形式の
    codon_state テンプレートを返す。

    Returns:
        {
            'base':     [str, ...],   # 各塩基位置の塩基 (0-indexed)
            'protein':  [str, ...],   # タンパク質領域名
            'aa_pos':   [int, ...],   # アミノ酸配列位置
            'codon':    [str, ...],   # コドン
            'codon_pos':[int, ...],   # コドン内位置 (1, 2, 3)
        }
    """
    df = pd.read_csv(codon_csv)
    return {
        'base':      df['base'].tolist(),
        'protein':   df['protein'].tolist(),
        'aa_pos':    df['protein_pos'].tolist(),
        'codon':     df['codon'].tolist(),
        'codon_pos': df['codon_pos'].tolist(),
    }


# ==============================================================
# 2. 1変異のコドン遷移を計算（codon_state を更新）
# ==============================================================

def apply_mutation(mutation: str, codon_state: dict):
    """
    1つの変異文字列（例: "C241T"）を codon_state に適用し、
    コドン遷移情報を返す。codon_state はインプレース更新される。

    Returns:
        (codon_before, codon_after, region, aa_before, aa_after)
        変異の base が不一致の場合は None を返す。
    """
    base_pos = int(mutation[1:-1])
    bef      = mutation[0]
    aft      = mutation[-1]
    idx      = base_pos - 1  # 0-indexed

    if idx < 0 or idx >= len(codon_state['base']):
        return None  # 範囲外

    base      = str(codon_state['base'][idx])
    protein   = str(codon_state['protein'][idx])
    codon     = str(codon_state['codon'][idx])
    codon_pos = int(codon_state['codon_pos'][idx])

    new_codon = codon

    if bef == base and codon != 'none':
        # 塩基更新
        codon_state['base'][idx] = aft

        # コドン更新
        if 1 <= codon_pos <= 3:
            c_list = list(codon)
            c_list[codon_pos - 1] = aft
            new_codon = ''.join(c_list)

            start_idx = idx - (codon_pos - 1)
            limit = len(codon_state['codon'])
            for k in range(3):
                curr = start_idx + k
                if 0 <= curr < limit:
                    codon_state['codon'][curr] = new_codon

    codon_bef = codon     if codon     != 'none' else 'nnn'
    codon_aft = new_codon if new_codon != 'none' else 'nnn'

    aa_bef = DNA2Protein.get(codon_bef, 'X')
    aa_aft = DNA2Protein.get(codon_aft, 'X')

    return (codon_bef, codon_aft, protein, aa_bef, aa_aft)


# ==============================================================
# 3. mutation_paths.tsv の読み込み
# ==============================================================

def iter_mutation_paths(usher_dir: str, max_cooccur: int = 10):
    """
    usher_output/ 以下の全 mutation_paths.tsv を走査し、
    各パスを (strain, path_list) として yield する。

    path_list: ['C241T', 'A23403G,C3037T', 'T19524C', ...]
      - '>' で分割されたステップのリスト
      - 共起変異はカンマ区切りのまま1ステップとして保持
    """
    strains = sorted(
        d for d in os.listdir(usher_dir)
        if not d.startswith('.')
    )
    force_print(f"[INFO] 系統数: {len(strains)}")

    total = 0
    skipped = 0

    for strain in strains:
        strain_dir = os.path.join(usher_dir, strain)
        if not os.path.isdir(strain_dir):
            continue

        # サブディレクトリ（0/, 1/, ...）を列挙
        tsv_paths = []
        direct_tsv = os.path.join(strain_dir, 'mutation_paths.tsv')
        if os.path.exists(direct_tsv):
            tsv_paths.append(direct_tsv)
        else:
            for sub in sorted(os.listdir(strain_dir)):
                if sub.isdigit():
                    p = os.path.join(strain_dir, sub, 'mutation_paths.tsv')
                    if os.path.exists(p):
                        tsv_paths.append(p)

        for tsv_path in tsv_paths:
            try:
                with open(tsv_path, 'r', encoding='utf-8_sig') as f:
                    reader = csv.reader(f, delimiter='\t')
                    next(reader, None)  # ヘッダースキップ
                    for row in reader:
                        if len(row) < 3:
                            continue
                        path_str = row[2].strip()
                        if not path_str:
                            continue

                        path_list = path_str.split('>')

                        # 共起数フィルタ
                        if max_cooccur is not None:
                            if any(len(step.split(',')) > max_cooccur for step in path_list):
                                skipped += 1
                                continue

                        total += 1
                        yield strain, path_list

            except Exception as e:
                force_print(f"[WARNING] {tsv_path}: {e}")

    force_print(f"[INFO] 読み込み完了: {total} サンプル (共起数超過除外: {skipped})")


def _count_total_strains(usher_dir: str) -> int:
    """usher_output/ 内の系統（ディレクトリ）数を返す"""
    return sum(
        1 for d in os.listdir(usher_dir)
        if not d.startswith('.') and os.path.isdir(os.path.join(usher_dir, d))
    )


# ==============================================================
# 4. コドン遷移頻度の集計（タイムステップ重複除外）
# ==============================================================

def compute_codon_freq(
    usher_dir: str,
    codon_csv:  str,
    max_cooccur: int = 10,
    strain_log_interval: int = 50,
    sample_log_interval: int = 100_000,
) -> Counter:
    """
    全サンプルのパスを走査し、タイムステップ重複除外方式で
    コドン遷移頻度を計算する。

    集計キー:
        (codon_before, codon_after, region, aa_before, aa_after)

    Args:
        strain_log_interval: 何系統ごとに進捗を表示するか
        sample_log_interval: 何サンプルごとに進捗を表示するか

    Returns:
        Counter: {(codon_before, codon_after, region, aa_before, aa_after): count}
    """
    force_print("[INFO] コドン参照データを読み込み中...")
    template = load_codon_template(codon_csv)

    total_strains = _count_total_strains(usher_dir)
    force_print(f"[INFO] パスの走査・集計を開始... (系統数: {total_strains})")
    start = time.time()

    # timestep_events[t] = set of (codon_bef, codon_aft, region, aa_bef, aa_aft)
    timestep_events: dict[int, set] = defaultdict(set)

    n_samples   = 0
    n_strains   = 0
    prev_strain = None

    for strain, path_list in iter_mutation_paths(usher_dir, max_cooccur):

        # --- 系統が切り替わったら系統カウントを進める ---
        if strain != prev_strain:
            n_strains  += 1
            prev_strain = strain

            if n_strains % strain_log_interval == 0 or n_strains == 1:
                elapsed = time.time() - start
                rate    = n_strains / elapsed if elapsed > 0 else 0
                eta_s   = (total_strains - n_strains) / rate if rate > 0 else 0
                eta_str = _fmt_time(eta_s)
                force_print(
                    f"[進捗] 系統 {n_strains:>4}/{total_strains}  "
                    f"サンプル {n_samples:>9,}  "
                    f"経過 {_fmt_time(elapsed)}  "
                    f"残り {eta_str}  "
                    f"({rate:.1f} 系統/s)  "
                    f"現系統: {strain}"
                )

        # --- コドン遷移の集計 ---
        local_state = {
            'base':      template['base'][:],
            'protein':   template['protein'],
            'aa_pos':    template['aa_pos'],
            'codon':     template['codon'][:],
            'codon_pos': template['codon_pos'],
        }

        for t, step in enumerate(path_list):
            for mutation in step.split(','):
                mutation = mutation.strip()
                if not mutation:
                    continue
                result = apply_mutation(mutation, local_state)
                if result is None:
                    continue
                codon_bef, codon_aft, region, aa_bef, aa_aft = result
                timestep_events[t].add((codon_bef, codon_aft, region, aa_bef, aa_aft))

        n_samples += 1
        if n_samples % sample_log_interval == 0:
            elapsed = time.time() - start
            rate_s  = n_samples / elapsed if elapsed > 0 else 0
            force_print(
                f"[進捗]   └ {n_samples:,} サンプル処理済み "
                f"({rate_s:,.0f} サンプル/s)  "
                f"タイムステップ種類数: {len(timestep_events)}"
            )

    # --- 最終進捗表示 ---
    elapsed = time.time() - start
    force_print(
        f"[進捗] 系統 {n_strains}/{total_strains} 完了  "
        f"総サンプル {n_samples:,}  経過 {_fmt_time(elapsed)}"
    )

    # --- タイムステップ別 set を合算 ---
    force_print("[INFO] タイムステップ別集合を合算中...")
    freq: Counter = Counter()
    for t_set in timestep_events.values():
        for key in t_set:
            freq[key] += 1

    elapsed = time.time() - start
    force_print(
        f"[INFO] 集計完了: {n_samples:,} サンプル, "
        f"{len(freq):,} ユニークコドン遷移, {_fmt_time(elapsed)}"
    )
    return freq


def _fmt_time(seconds: float) -> str:
    """秒数を HH:MM:SS 形式の文字列に変換する"""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"


# ==============================================================
# 5. 結果の保存・読み込み
# ==============================================================

def save_codon_freq(freq: Counter, output_path: str) -> None:
    """
    集計結果を CSV に保存する。

    カラム: codon_before, codon_after, region, aa_before, aa_after, count
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for (cb, ca, region, ab, aa), count in freq.most_common():
        rows.append({
            'codon_before': cb,
            'codon_after':  ca,
            'region':       region,
            'aa_before':    ab,
            'aa_after':     aa,
            'count':        count,
        })

    df = pd.DataFrame(rows, columns=[
        'codon_before', 'codon_after', 'region', 'aa_before', 'aa_after', 'count'
    ])
    df.to_csv(output_path, index=False)
    force_print(f"[INFO] 保存完了: {output_path} ({len(df)} 行)")


def load_codon_freq(csv_path: str) -> dict:
    """
    save_codon_freq で保存した CSV を読み込み、
    { (codon_before, codon_after, region): count } の辞書を返す。

    特徴量生成での参照用。
    """
    df = pd.read_csv(csv_path)
    freq = {}
    for row in df.itertuples(index=False):
        key = (row.codon_before, row.codon_after, row.region)
        freq[key] = int(row.count)
    return freq


# ==============================================================
# 6. エントリーポイント（単体実行用）
# ==============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='コドン別変異頻度を計算して CSV に保存する'
    )
    parser.add_argument(
        '--usher_dir',
        default='/mnt/ssd1/home3/aiba/usher_output',
        help='usher_output ディレクトリのパス',
    )
    parser.add_argument(
        '--codon_csv',
        default='reference/codon/codon_mutation4.csv',
        help='codon_mutation4.csv のパス（gmp/ からの相対パス）',
    )
    parser.add_argument(
        '--output',
        default='outputs/codon_freq/codon_freq.csv',
        help='出力 CSV のパス',
    )
    parser.add_argument(
        '--max_cooccur',
        type=int,
        default=10,
        help='共起変異数の上限（これを超えるステップを含むパスは除外）',
    )
    args = parser.parse_args()

    freq = compute_codon_freq(
        usher_dir=args.usher_dir,
        codon_csv=args.codon_csv,
        max_cooccur=args.max_cooccur,
    )
    save_codon_freq(freq, args.output)
