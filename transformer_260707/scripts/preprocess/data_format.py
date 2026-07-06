# --- transformer_260707/scripts/data_format.py ---
# UShER 出力の mutation-paths.txt を mutation_paths.tsv に変換する前処理スクリプト。
# preprocess.py より先に実行する（1回限りの変換処理）。
# Usage: python -m transformer_260707.scripts.data_format
import os
import re
import pandas as pd
from transformer_260707 import config


def get_usher_output_dir():
    """config.DATA_BASE_DIR を絶対パスに解決する。"""
    return os.path.abspath(config.DATA_BASE_DIR)


def process_mutation_paths_safe(strain_dir, strain):
    """
    mutation-paths.txt を処理し、複数の TSV ファイルを生成する。
    """
    def handle_mutation_paths(file_path, output_prefix):
        try:
            print(f"Processing file: {file_path}")
            df = pd.read_csv(file_path, sep='\t', header=None)
            df = df.set_axis(['name', 'mutation_path'], axis=1)
            df = df.dropna(subset=['mutation_path'])
            print(f"  Data rows after NaN removal: {len(df)}")

            clades_file = os.path.join(os.path.dirname(file_path), 'clades.txt')
            if not os.path.exists(clades_file):
                print(f"  Warning: clades.txt not found at {clades_file}")
                return

            with open(clades_file, 'r', encoding='utf-8', errors='replace') as f:
                datalist = f.readlines()
            clades = [data.split('\t')[2].rstrip() for data in datalist if len(data.split('\t')) > 2]

            if len(df) != len(clades):
                print(f"  Warning: Data length ({len(df)}) != Clades length ({len(clades)})")
                min_len = min(len(df), len(clades))
                df = df.head(min_len)
                clades = clades[:min_len]
                print(f"  Truncated to {min_len} rows")

            df_mutation = df.loc[:, 'mutation_path']
            name = df.loc[:, 'name']

            mutation_paths = []
            for path in df_mutation.values.tolist():
                if pd.isna(path) or path == '' or not isinstance(path, str):
                    mutation_paths.append([])
                else:
                    mutation_paths.append(path.split(' '))

            for path in mutation_paths:
                if path:
                    path[:] = [re.sub('.*:', '', mutation) for mutation in path]
                    if path:
                        path.pop(-1)

            mutation_paths_strain = []
            name_strain = []
            mutation_paths_other = []
            name_other = []

            for i, clade in enumerate(clades):
                if i < len(mutation_paths) and i < len(name):
                    if clade == strain:
                        mutation_paths_strain.append(mutation_paths[i])
                        name_strain.append(name.iloc[i])
                    else:
                        mutation_paths_other.append(mutation_paths[i])
                        name_other.append(name.iloc[i])

            def write_tsv(file_name, names, paths):
                try:
                    with open(file_name, 'w') as f:
                        f.write("name\tlength(「>」separate)\tmutation_path\n")
                        for n, path in zip(names, paths):
                            temp = '>'.join(path) if path else ''
                            f.write(f"{n}\t{len(path)}\t{temp}\n")
                    print(f"  Saved: {file_name}")
                except Exception as e:
                    print(f"  Error writing {file_name}: {e}")

            write_tsv(output_prefix + 'mutation_paths.tsv', name, mutation_paths)
            write_tsv(output_prefix + f'mutation_paths_{strain}.tsv', name_strain, mutation_paths_strain)
            write_tsv(output_prefix + 'mutation_paths_other.tsv', name_other, mutation_paths_other)

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

    try:
        print(f"\n=== Processing strain: {strain} ===")

        if not os.path.exists(strain_dir):
            print(f"  Directory not found: {strain_dir}")
            return

        try:
            numeric_dirs = [d for d in os.listdir(strain_dir)
                            if os.path.isdir(os.path.join(strain_dir, d)) and d.isdigit()]
            numeric_dirs = sorted(set(numeric_dirs), key=int)
        except Exception as e:
            print(f"  Error reading directory {strain_dir}: {e}")
            return

        if numeric_dirs:
            print(f"  Found numeric subdirs: {numeric_dirs}")
            for numeric_dir in numeric_dirs:
                file_path = os.path.join(strain_dir, numeric_dir, 'mutation-paths.txt')
                if os.path.exists(file_path):
                    handle_mutation_paths(file_path, os.path.join(strain_dir, numeric_dir) + '/')
                else:
                    print(f"  File not found: {file_path}")
        else:
            file_path = os.path.join(strain_dir, 'mutation-paths.txt')
            if os.path.exists(file_path):
                handle_mutation_paths(file_path, strain_dir + '/')
            else:
                print(f"  File not found: {file_path}")

    except Exception as e:
        print(f"Error processing strain {strain}: {e}")


def main():
    import time

    usher_output_dir = get_usher_output_dir()

    if not os.path.exists(usher_output_dir):
        print(f"Directory not found: {usher_output_dir}")
        return

    usher_folders = sorted(
        item for item in os.listdir(usher_output_dir)
        if os.path.isdir(os.path.join(usher_output_dir, item))
    )
    print(f"usher_output フォルダ数: {len(usher_folders)}  ({usher_output_dir})")

    print(f"\n=== Processing all {len(usher_folders)} strains ===")
    start_time = time.time()
    success_count = 0
    error_count = 0

    for i, strain in enumerate(usher_folders):
        try:
            print(f"\n[{i+1}/{len(usher_folders)}] Processing: {strain}")
            strain_dir = os.path.join(usher_output_dir, strain)
            process_mutation_paths_safe(strain_dir, strain)
            success_count += 1
        except Exception as e:
            print(f"  Fatal error for {strain}: {e}")
            error_count += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            pct = (i + 1) / len(usher_folders) * 100
            print(f"\n--- Progress: {i+1}/{len(usher_folders)} ({pct:.1f}%) "
                  f"| Elapsed: {elapsed:.1f}s | Success: {success_count} Errors: {error_count} ---")

    total_time = time.time() - start_time
    print(f"\n=== Complete === Total: {total_time:.1f}s | "
          f"Success: {success_count} | Errors: {error_count} | "
          f"Rate: {success_count / max(len(usher_folders), 1) * 100:.1f}%")


if __name__ == '__main__':
    main()
