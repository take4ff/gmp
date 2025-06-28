import os

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
def search_mutation_paths(base_dir, strain):
    # ホームディレクトリを展開
    base_dir = os.path.expanduser(base_dir)
    strain_dir = os.path.join(base_dir, strain)

    # strain直下のファイルパスを確認
    file_paths = []
    file_path = os.path.join(strain_dir, f"mutation_paths_{strain}.tsv")
    if os.path.exists(file_path):
        file_paths.append(file_path)
    
    # strain/numサブディレクトリを探索
    else:
        if os.path.exists(strain_dir) and os.path.isdir(strain_dir):
            num_dirs = [d for d in os.listdir(strain_dir) if d.isdigit()]
            num_dirs.sort(key=int)  # 数字順にソート

            for num in num_dirs:
                file_path = os.path.join(strain_dir, num, f"mutation_paths_{strain}.tsv")
                if os.path.exists(file_path):
                    file_paths.append(file_path)

    if not file_paths:
        raise FileNotFoundError(f"mutation_paths_{strain}.tsvが{strain_dir}内に見つかりませんでした。")

    return file_paths

def input(strains, usher_dir, nmax=None, nmax_per_strain=None):
    names = []
    lengths = []
    paths = []
    data_i = 0
    for strain in strains:
        file_paths = search_mutation_paths(usher_dir,strain)
        data_i_strain = 0
        data_i_strain_flag = False
        for file_path in file_paths:
            print(f"[INFO] import: {file_path}")
            f = open(file_path, 'r',encoding="utf-8_sig")
            datalist = f.readlines()
            f.close()

            data_num = len(datalist)

            for i in range(1,data_num):
                if nmax is not None and data_i >= nmax:
                    print(f"[INFO] 指定されたnmax={nmax}に達しました。")
                    print(f"[INFO] {strain}のデータを読み込みました: {data_i_strain} サンプル")
                    print(f"[INFO] 読み込み完了: {len(paths)} サンプル")
                    return names, lengths, paths
                
                data = datalist[i].split('\t')
                names.append(data[0])
                lengths.append(int(data[1]))
                paths.append(data[2].rstrip().split('>'))

                data_i += 1
                data_i_strain += 1
                if nmax_per_strain is not None and data_i_strain >= nmax_per_strain:
                    print(f"[INFO] {strain}の指定されたnmax_per_strain={nmax_per_strain}に達しました。")
                    data_i_strain_flag = True
                    break
                
            if data_i_strain_flag:
                break
        print(f"[INFO] {strain}のデータを読み込みました: {data_i_strain} サンプル")
    
    print(f"[INFO] 全件読み込み完了: {len(paths)} サンプル")
    return names, lengths, paths
