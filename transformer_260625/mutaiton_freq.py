# copy:jupyter_notebook/20250620_table_heatmap.ipynb
# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors
from matplotlib import cm
import cv2

import glob
import re

# %%
def count_numeric_subfolders(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"エラー: 指定されたパスが見つかりません: {folder_path}")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"エラー: 指定されたパスはフォルダではありません: {folder_path}")
    numeric_folder_count = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and item.isdigit():
            numeric_folder_count += 1
    return numeric_folder_count

# %%
def import_mutation_paths(base_dir, strain):
    # ホームディレクトリを展開
    base_dir = os.path.expanduser(base_dir)
    strain_dir = os.path.join(base_dir, strain)

    # strain直下のファイルパスを確認
    file_paths = []
    file_path = os.path.join(strain_dir, f"mutation_paths.tsv")
    #file_path = os.path.join(strain_dir, f"mutation_paths_{strain}.tsv")
    if os.path.exists(file_path):
        file_paths.append(file_path)
    
    # strain/numサブディレクトリを探索
    else:
        if os.path.exists(strain_dir) and os.path.isdir(strain_dir):
            num_dirs = [d for d in os.listdir(strain_dir) if d.isdigit()]
            num_dirs.sort(key=int)  # 数字順にソート

            for num in num_dirs:
                file_path = os.path.join(strain_dir, num, f"mutation_paths.tsv")
                #file_path = os.path.join(strain_dir, num, f"mutation_paths_{strain}.tsv")
                if os.path.exists(file_path):
                    file_paths.append(file_path)

    if not file_paths:
        raise FileNotFoundError(f"mutation_paths_{strain}.tsvが{strain_dir}内に見つかりませんでした。")

    return file_paths


# %%
#変異パターンに対応する表中のセルのインクリメント（position step指定可能）
def count_mutation(df,mutation,step):

    bef = mutation[0]
    aft = mutation[-1]
    pos = int(int(mutation[1:-1]) /step + 0.5)
    pattern = f"{bef}->{aft}"

    df.loc[pos-1, pattern] += 1

# 変異パターンの表を作成
def mutation_table(paths,step=1):
    columns = ["position","A->T","A->G","A->C","T->A","T->G","T->C","G->A","G->T","G->C","C->A","C->T","C->G"]
    df = pd.DataFrame(np.zeros((int(30000/step),len(columns)),dtype=int), columns=columns)
    df["position"] = range(1, 30001, step)

    for path in paths:
        for mutations in path:
            for mutation in mutations.split(','):
                count_mutation(df, mutation,step)
    df_all = pd.DataFrame(np.zeros((int(30000/step), 2), dtype=int), columns=["position","all"])
    df_all["position"] = range(1, 30001, step)
    df_all["all"] = df.iloc[:, 1:].sum(axis=1)
    return df,df_all

# タイムステップごとの変異パターンの表を作成（position step指定可能）
def mutation_table_by_timestep(paths,time_step_max, step=1):
    df_timestep = []
    for i in range(time_step_max):
        columns = ["position","A->T","A->G","A->C","T->A","T->G","T->C","G->A","G->T","G->C","C->A","C->T","C->G"]
        df = pd.DataFrame(np.zeros((int(30000/step),len(columns)),dtype=int), columns=columns)
        df["position"] = range(1, 30001, step)

        for path in paths:
            if i >= len(path):
                continue
            for mutation in path[i].split(','):
                count_mutation(df, mutation, step)
        df_timestep.append(df)

    return df_timestep

# %%
# DataFrameをCSVファイルに保存
def save_df(df, output_dir = 'tables', output_file_name = 'mutation_table.csv'):

    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, output_file_name)
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

# タイムステップごとのDataFrameをCSVファイルに保存
def save_timestep_df(timestep_dataframes, output_dir = 'timestep_tables'):

    os.makedirs(output_dir, exist_ok=True)

    for i, df in enumerate(timestep_dataframes):
        file_path = os.path.join(output_dir, f"timestep_{i + 1}.csv")
        df.to_csv(file_path, index=False)
        #print(f"Saved: {file_path}")

# タイムステップごとのヒートマップを作成・保存
def save_heatmaps(df_timestep, step, output_dir = 'timestep_heatmaps'):
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])  # 0番目を白に
    white_viridis = mcolors.ListedColormap(newcolors)

    os.makedirs(output_dir, exist_ok=True)

    columns = ["position","A->T","A->G","A->C","T->A","T->G","T->C","G->A","G->T","G->C","C->A","C->T","C->G"]
    x_labels = columns[1:]  # positionを除外

    for i, df in enumerate(df_timestep):
        data = df.iloc[:, 1:]
        norm_data = (data - data.min().min()) / (data.max().max() - data.min().min())
        plt.figure(figsize=(10, 6))
        plt.imshow(norm_data, aspect='auto', cmap=white_viridis, interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(label='Normalized Value')
        plt.title(f'Timestep {i+1}')
        plt.xlabel('Mutation Type')
        plt.ylabel(f'position')
        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(0, len(df) + 1, step // 2), labels=(np.arange(0, len(df) + 1, step // 2) * step).astype(int))
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mutation_heatmap_timestep_{i+1}.png')
        plt.close()
    print('ヒートマップを保存しました。')

def save_heatmap(df, step, output_dir = 'timestep_heatmaps'):
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])  # 0番目を白に
    white_viridis = mcolors.ListedColormap(newcolors)

    os.makedirs(output_dir, exist_ok=True)

    columns = ["position","A->T","A->G","A->C","T->A","T->G","T->C","G->A","G->T","G->C","C->A","C->T","C->G"]
    x_labels = columns[1:]  # positionを除外

    data = df.iloc[:, 1:]
    norm_data = (data - data.min().min()) / (data.max().max() - data.min().min())
    plt.figure(figsize=(10, 6))
    plt.imshow(norm_data, aspect='auto', cmap=white_viridis, interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Normalized Value')
    plt.xlabel('Mutation Type')
    plt.ylabel(f'position')
    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(0, len(df) + 1, step // 2), labels=(np.arange(0, len(df) + 1, step // 2) * step).astype(int))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mutation_heatmap.png')
    plt.close()
    print('ヒートマップを保存しました。')

# タイムステップごとのヒートマップ画像から動画を作成
def create_video_from_heatmaps(image_folder, output_video_path="timestep_heatmaps/mutation_heatmaps_video.mp4", fps=2):
    # Get list of image files sorted by timestep
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") and "heatmap_timestep" in img]
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by timestep

    # Read the first image to get dimensions
    if not images:
        raise ValueError("No valid heatmap images found in the specified folder.")

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Add images to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()

# %%
#mutation-pathsを転置して重複を除去
def transposition_and_set(paths):
    timestep_mutation = {}

    for path in paths:
        for timestep, sample in enumerate(path):
            if timestep not in timestep_mutation:
                timestep_mutation[timestep] = set()
            timestep_mutation[timestep].add(sample)

    return timestep_mutation

#タイムステップごとのリストを1次リストに変換
def to1Dlist(timestep_data):

    mutation_list = []
    
    for timestep, mutations in timestep_data.items():
        #print(mutations)
        for mutation in mutations:
            #rint(mutation)
            mutation_list.append(mutation)
    
    return mutation_list

# 重複を除去した1次リストから変異パターンの表を作成（position step指定可能）
def mutation_table_form_1Dlist(path,step=1):
    columns = ["position","A->T","A->G","A->C","T->A","T->G","T->C","G->A","G->T","G->C","C->A","C->T","C->G"]
    df = pd.DataFrame(np.zeros((int(30000/step),len(columns)),dtype=int), columns=columns)
    df["position"] = range(1, 30001, step)

    for mutations in path:
        for mutation in mutations.split(','):
            count_mutation(df, mutation,step)
    df_all = pd.DataFrame(np.zeros((int(30000/step), 2), dtype=int), columns=["position","all"])
    df_all["position"] = range(1, 30001, step)
    df_all["all"] = df.iloc[:, 1:].sum(axis=1)
    return df,df_all

# %%
def tables_summary(strains,timestep_max_list,output_dir, tables_dir):
    output_dir_ts_sum = os.path.join(output_dir, "timestep_tables")
    #timestep_tables_summaryの作成
    ts_max = max(timestep_max_list)
    os.makedirs(output_dir_ts_sum+"/default", exist_ok=True)
    columns = ["position","A->T","A->G","A->C","T->A","T->G","T->C","G->A","G->T","G->C","C->A","C->T","C->G"]
    df_summary = pd.DataFrame(np.zeros((int(30000),len(columns)),dtype=int), columns=columns)
    df_summary["position"] = range(1, 30001)
    for ts in range(1,ts_max+1):
        df_summary_ts = pd.DataFrame(np.zeros((int(30000),len(columns)),dtype=int), columns=columns)
        df_summary_ts["position"] = range(1, 30001)
        for strain in strains:
            file_path = os.path.join(tables_dir, strain, f"timestep_{ts}.csv")
            if not os.path.exists(os.path.join(file_path)):
                continue
            #print(f"[INFO] {strain}のtimestep_{ts}.csvを読み込み中")
            df_strain = pd.read_csv(file_path)
            for col in df_strain.columns[1:]:
                df_summary_ts[col] += df_strain[col]
        save_df(df_summary_ts, output_dir_ts_sum+"/default", f"timestep_{ts}.csv")
        
        for col in df_summary_ts.columns[1:]:
            df_summary[col] += df_summary_ts[col]
    save_df(df_summary, output_dir, "tables.csv")

    df_all = pd.DataFrame(np.zeros((int(30000), 2), dtype=int), columns=["position","all"])
    df_all["position"] = range(1, 30001)
    df_all["all"] = df_summary.iloc[:, 1:].sum(axis=1)
    save_df(df_all, output_dir, "tables_sum.csv")

    return df_summary_ts
    

# %%
def compress_csv_rows(file,step):
    #print(f"Processing file: {file}")
    df = pd.read_csv(file)
    compressed_rows = []
    for i in range(0, len(df), step):
        block = df.iloc[i:i+step]
        if len(block) == 0:
            continue
        row = {}
        row['position'] = block.iloc[0, 0]  # 先頭行のposition
        # 2列目以降を合計
        for col in df.columns[1:]:
            row[col] = block[col].sum()
        compressed_rows.append(row)
    df_compressed = pd.DataFrame(compressed_rows)
    return df_compressed

def compress_csv_rows_folder(input_dir, output_dir, step=100):
    df_compressed_ts = []
    os.makedirs(output_dir, exist_ok=True)

    def natural_key(filename):
        # ファイル名から数字部分を抽出してintで返す
        m = re.search(r'(\d+)', os.path.basename(filename))
        return int(m.group(1)) if m else 0
    
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")), key=natural_key)
    for file in csv_files:
        df_compressed = compress_csv_rows(file,step)
        out_path = os.path.join(output_dir, os.path.basename(file))
        df_compressed.to_csv(out_path, index=False)
        df_compressed_ts.append(df_compressed)
        print(f"Saved: {out_path}")
    return df_compressed_ts

# %%
usher_dir = '../usher_output/'
files =  sorted([filename for filename in os.listdir(usher_dir) if not filename.startswith('.')])
print(files)
print(len(files))

strains = files

dir_ver = "260610"
output_dir = os.path.join("outputs/table_heatmap", dir_ver)
tables_dir = os.path.join(output_dir, "timestep_tables/strains")

"""
timestep_max_list = []

for strain in strains:
    names = []
    lengths = []
    paths = []
    file_paths = import_mutation_paths(usher_dir,strain)
    for file_path in file_paths:
        #print(f"[INFO]import: {file_path}")
        f = open(file_path, 'r',encoding="utf-8_sig")
        datalist = f.readlines()
        f.close()

        data_num = len(datalist)

        for i in range(1,data_num):
            data = datalist[i].split('\t')
            names.append(data[0])
            lengths.append(int(data[1]))
            paths.append(data[2].rstrip().split('>'))

    print(f"[INFO] {strain}のデータ読み込み完了: {len(paths)} サンプル")
    #print(paths[0])

    timestep_max = max(lengths)
    timestep_max_list.append(timestep_max)

    tables_dir_strain = os.path.join(tables_dir, strain)
    print(f"[INFO] {strain}のtimestep_tablesを作成中:{tables_dir_strain}")
    os.makedirs(tables_dir_strain, exist_ok=True)
    
    df_timestep = mutation_table_by_timestep(paths,timestep_max)
    save_timestep_df(df_timestep, tables_dir_strain)  
df_timestep = tables_summary(strains, timestep_max_list, output_dir, tables_dir)   

# %%
pos_step = 100
input_dir_pos_step = os.path.join(output_dir, "timestep_tables/default")
output_dir_pos_step = os.path.join(output_dir, "timestep_tables/pos-step" + str(pos_step))
df_timestep_pos_step = compress_csv_rows_folder(input_dir_pos_step, output_dir_pos_step, pos_step)

heatmaps_dir = os.path.join(output_dir, "timestep_heatmaps/pos-step" + str(pos_step))
save_heatmaps(df_timestep_pos_step, pos_step, heatmaps_dir)
create_video_from_heatmaps(heatmaps_dir, heatmaps_dir+"/mutation_heatmaps_video.mp4", fps=1)
"""

# %%
# --- データ読み込み・前処理 ---
usher_dir = '../usher_output/'
files =  sorted([filename for filename in os.listdir(usher_dir) if not filename.startswith('.')])
print(files)
print(len(files))

strains = files
#max_co_occur = 5

# 全件データの読み込み
names = []
lengths = []
paths = []
for strain in strains:
    file_paths = import_mutation_paths(usher_dir,strain)
    for file_path in file_paths:
        print(f"[INFO]import: {file_path}")
        f = open(file_path, 'r',encoding="utf-8_sig")
        datalist = f.readlines()
        f.close()

        data_num = len(datalist)

        for i in range(1,data_num):
            data = datalist[i].split('\t')
            names.append(data[0])
            lengths.append(int(data[1]))
            paths.append(data[2].rstrip().split('>'))
    
print(f"[INFO] 全件読み込み完了: {len(paths)} サンプル")

# %%
output_dir_set = os.path.join(output_dir, "table_set")
os.makedirs(output_dir_set, exist_ok=True)

timestep_data = transposition_and_set(paths)
mutation_list = to1Dlist(timestep_data)
df_set,df_set_all = mutation_table([mutation_list])
save_df(df_set, output_dir_set, output_file_name = 'table_set.csv')
save_df(df_set_all, output_dir_set, output_file_name = 'table_set_sum.csv')

# %%
pos_step = 100
df_set_path = os.path.join(output_dir_set, "table_set.csv")
df_set_pos_step = compress_csv_rows(df_set_path,pos_step)

save_df(df_set_pos_step, output_dir_set, output_file_name = 'table_set_pos-step'+str(pos_step)+'.csv')
save_heatmap(df_set_pos_step,pos_step,output_dir_set)

# %%
timestep_data

# %%



