##バッチサイズ決定関数くそ、使わん
# %%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
# from bertviz import head_view, model_view # bertvizは直接バッチサイズ決定には不要なのでコメントアウトしてもOK
import matplotlib.pyplot as plt
import numpy as np
import copy # 元のリストを変更しないようにするため
import torch.nn as nn ### 追加 ### (estimate_memory_per_sampleでダミーの損失関数を使う場合や、モデル構造の理解のため)

# デバイス設定 (GPUが利用可能ならGPUを使用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
def count_numeric_subfolders(folder_path):
    """
    指定されたフォルダパス直下にあるサブフォルダのうち、
    フォルダ名が数字のみで構成されているものの数を数えます。

    Args:
        folder_path (str): サブフォルダの数を数えたいフォルダのパス。

    Returns:
        int: フォルダ名が数字のみのサブフォルダの数。

    Raises:
        FileNotFoundError: 指定されたパスが存在しない場合。
        NotADirectoryError: 指定されたパスがフォルダではない場合。
        PermissionError: 指定されたパスへのアクセス権がない場合。
        OSError: その他のOS関連のエラーが発生した場合。
    """
    # パスが存在するか、フォルダであるかを確認
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"エラー: 指定されたパスが見つかりません: {folder_path}")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"エラー: 指定されたパスはフォルダではありません: {folder_path}")

    numeric_folder_count = 0
    try:
        # フォルダ内の全てのアイテム（ファイルとフォルダ）を取得
        for item in os.listdir(folder_path):
            # アイテムのフルパスを作成
            item_path = os.path.join(folder_path, item)
            # アイテムがフォルダであり、かつ、その名前が数字のみで構成されているかを確認
            if os.path.isdir(item_path) and item.isdigit():
                numeric_folder_count += 1
    except PermissionError:
        raise PermissionError(f"エラー: '{folder_path}' へのアクセス権がありません。")
    except OSError as e:
        raise OSError(f"エラー: フォルダ '{folder_path}' の読み取り中にエラーが発生しました: {e}")

    return numeric_folder_count


# %%
max_pos = 30000

name = []
length = []
mutation_paths = []

strain_group1 = ['B.1.1.7','BA.1','B.1.617.2'] # strain を strain_group1 に変更 (重複を避けるため)

for s in strain_group1: # strain を strain_group1 に変更
    dir_path = 'sequences_20241017_'+s+'_random/' # dir を dir_path に変更
    if not os.path.exists(dir_path):
        print(f"Warning: Directory not found, skipping: {dir_path}")
        continue
    folder_num = count_numeric_subfolders(dir_path)

    for num in range(folder_num):
        file_path = os.path.join(dir_path, str(num), 'mutation_paths_'+s+'.tsv') # ファイルパス結合を修正
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
        print(f"Processing file: {file_path}")
        with open(file_path, 'r',encoding="utf-8-sig") as f: # with open を使用
            datalist = f.readlines()
        
        for i in range(1,len(datalist)):
            data = datalist[i].split('\t')
            if len(data) > 2: # データ形式のチェック
                name.append(data[0])
                length.append(int(data[1]))
                mutation_paths.append(data[2].rstrip().split('>'))
            else:
                print(f"Warning: Skipping malformed line in {file_path}: {datalist[i]}")


strain_group2 = ['P.1','B.1.351','B.1.1.529'] # strain を strain_group2 に変更

for s in strain_group2: # strain を strain_group2 に変更
    dir_path = 'sequences_20241017_'+s+'/' # dir を dir_path に変更
    if not os.path.exists(dir_path):
        print(f"Warning: Directory not found, skipping: {dir_path}")
        continue
    
    file_path = os.path.join(dir_path, 'mutation_paths_'+s+'.tsv') # ファイルパス結合
    if not os.path.exists(file_path):
        print(f"Warning: File not found, skipping: {file_path}")
        continue
    print(f"Processing file: {file_path}")
    with open(file_path, 'r',encoding="utf-8-sig") as f: # with open を使用
        datalist = f.readlines()
    
    for i in range(1,len(datalist)):
        data = datalist[i].split('\t')
        if len(data) > 2: # データ形式のチェック
            name.append(data[0])
            length.append(int(data[1]))
            mutation_paths.append(data[2].rstrip().split('>'))
        else:
            print(f"Warning: Skipping malformed line in {file_path}: {datalist[i]}")


if name: # リストが空でないことを確認
    for i in range(min(5, len(name))): # インデックスエラーを防ぐ
        print(name[i],length[i],mutation_paths[i])
print("sample_num",len(name))


# %%
def filter_co_occur(data,sample_name,data_len,max_co_occur,out_num):
    filted_data = []
    filted_sample_name =[]
    filted_data_len = []
    for i in range(len(data)):
        compare = 0
        for j in range(len(data[i])):
            mutation = data[i][j].split(',')
            if(compare < len(mutation)):
                compare = len(mutation)
        if(compare <= max_co_occur):
            filted_data.append(data[i])
            filted_sample_name.append(sample_name[i])
            filted_data_len.append(data_len[i])
        if(len(filted_data)>=out_num):
            break
    return filted_data,filted_sample_name,filted_data_len

def get_max_elements_per_dimension(nested_list, dimension=0, max_elements=None):
    if max_elements is None:
        max_elements = []
    
    if dimension >= len(max_elements):
        max_elements.append(0)
    
    max_elements[dimension] = max(max_elements[dimension], len(nested_list))
    
    for item in nested_list:
        if isinstance(item, list):
            get_max_elements_per_dimension(item, dimension + 1, max_elements)
    
    return max_elements



def get_min_elements_per_dimension(nested_list, dimension=0, min_elements=None):
    if min_elements is None:
        min_elements = []
    
    if dimension >= len(min_elements):
        min_elements.append(float('inf'))
    
    min_elements[dimension] = min(min_elements[dimension], len(nested_list))
    
    for item in nested_list:
        if isinstance(item, list):
            get_min_elements_per_dimension(item, dimension + 1, min_elements)
    
    min_elements = [0 if val == float('inf') else val for val in min_elements]
    
    return min_elements

def separate_XY(data, n):
    x_list = []
    y_list = []
    for item in data: 
        if len(item) > n:
            x = item[:-n]
            y_part = item[-n:] 

            if isinstance(y_part, list) and len(y_part) == n:
                if n == 1:
                    y = y_part[0] 
                else:
                    y = y_part 
            else:
                raise ValueError(f"Unexpected format for y_part: {y_part}")


            x_list.append(x)
            y_list.append(y) 
        else:
            pass 
    return x_list, y_list

# %%
if mutation_paths: # リストが空でないことを確認
    mutation_paths,name,length = filter_co_occur(mutation_paths,name,length,5,len(mutation_paths))

    # ラベルエンコード
    flat_data = [item for series in mutation_paths for item in series]
    print(len(flat_data))
    print(len(list(set(flat_data))))
else:
    flat_data = []
    print("mutation_paths is empty. Skipping label encoding.")


# %%
if length: # lengthが空でないか確認
    length_count = []
    length_i = []
    for i in range(0,max(length)+1): # max(length)+1 に修正
        length_count.append(length.count(i))
        length_i.append(i)
    plt.figure(figsize=[12,5])
    plt.bar(length_i,length_count)
    # plt.xticks() # xticksは自動で設定されるので、この行はなくても良い場合が多い
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.title("Distribution of Lengths")
    plt.show()

    for i in length_i:
        if(length_count[i] != 0):
            print(i,length_count[i])
else:
    print("Length list is empty. Skipping plotting length distribution.")

# %%
# strain_group_for_BA1 = ['BA.1'] # strain を strain_group_for_BA1 に変更
# BA_1_len = []

# for s in strain_group_for_BA1: # strain を strain_group_for_BA1 に変更
#     dir_path = 'sequences_20241017_'+s+'_random/' # dir を dir_path に変更
#     if not os.path.exists(dir_path):
#         print(f"Warning: Directory not found, skipping: {dir_path}")
#         continue
#     folder_num = count_numeric_subfolders(dir_path)

#     for num in range(folder_num):
#         file_path = os.path.join(dir_path, str(num), 'mutation_paths_'+s+'.tsv')
#         if not os.path.exists(file_path):
#             print(f"Warning: File not found, skipping: {file_path}")
#             continue
#         print(f"Processing file for BA.1 length: {file_path}")
#         with open(file_path, 'r',encoding="utf-8-sig") as f:
#             datalist = f.readlines()
        
#         for i in range(1,len(datalist)):
#             data = datalist[i].split('\t')
#             if len(data) > 1: # データ形式のチェック
#                 BA_1_len.append(int(data[1]))
#             else:
#                 print(f"Warning: Skipping malformed line in {file_path} for BA.1 length: {datalist[i]}")
# このセクションは最初のデータ読み込みと重複しているように見えますので、一旦コメントアウトします。
# 必要であればコメントアウトを解除してください。
# もしBA.1のデータのみを別途処理したい場合は、このままで問題ありません。


# %%
# if BA_1_len: # BA_1_lenが空でないか確認
#     length_count = []
#     length_i = []
#     for i in range(0,max(BA_1_len)+1): # max(BA_1_len)+1 に修正
#         length_count.append(BA_1_len.count(i))
#         length_i.append(i)
#     plt.figure(figsize=[12,5])
#     plt.bar(length_i,length_count)
#     plt.xlabel("Length (BA.1)")
#     plt.ylabel("Count (BA.1)")
#     plt.title("Distribution of Lengths (BA.1)")
#     plt.show()

#     for i in length_i:
#         if(length_count[i] != 0):
#             print(i,length_count[i])
# else:
#     print("BA_1_len list is empty. Skipping plotting BA.1 length distribution.")
# このセクションは最初のデータ読み込みと重複しているように見えますので、一旦コメントアウトします。


# %%
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
if flat_data: # flat_dataが空でないことを確認
    tokenizer.add_tokens(list(set(flat_data)))
    print("tokenizer customized with new tokens.")
else:
    print("flat_data is empty. Tokenizer not customized with new tokens.")
print("tokenizer complete")
if mutation_paths:
    print(get_max_elements_per_dimension(mutation_paths))


# %%
def filter_length(data,length_list,low,up): # length -> length_list に変更
    filted_data = []
    if not data or not length_list: # データが空の場合は空リストを返す
        return filted_data
    for i in range(len(length_list)): # length -> length_list に変更
        if i < len(data): # dataのインデックス範囲チェック
            if(length_list[i]>=low): # length -> length_list に変更
                if(length_list[i]>up): # length -> length_list に変更
                    filted_data.append(data[i][:up])
                else:
                    filted_data.append(data[i])
    return filted_data

# %%
if mutation_paths and length: # データが存在することを確認
    mutation_paths0 = filter_length(mutation_paths,length,0,40)
    mutation_paths1 = filter_length(mutation_paths,length,41,45)
    mutation_paths2 = filter_length(mutation_paths,length,46,50)
    mutation_paths3 = filter_length(mutation_paths,length,51,100) # length が引数として渡されていなかったので修正

    if mutation_paths0: print(get_max_elements_per_dimension(mutation_paths0))
    if mutation_paths1: print(get_max_elements_per_dimension(mutation_paths1))
    if mutation_paths2: print(get_max_elements_per_dimension(mutation_paths2))
    if mutation_paths3: print(get_max_elements_per_dimension(mutation_paths3))

    if mutation_paths3:
        test3 = [tokenizer.convert_tokens_to_ids(frag) for frag in mutation_paths3]
        print("test3_complete")
        if test3:
            print(get_max_elements_per_dimension(test3))
            print(test3[0])
    else: test3 = []

    if mutation_paths2:
        test2 = [tokenizer.convert_tokens_to_ids(frag) for frag in mutation_paths2]
        print("test2_complete")
        if test2:
            print(get_max_elements_per_dimension(test2))
            print(test2[0])
    else: test2 = []

    if mutation_paths1:
        test1 = [tokenizer.convert_tokens_to_ids(frag) for frag in mutation_paths1]
        print("test1_complete")
        if test1:
            print(get_max_elements_per_dimension(test1))
            print(test1[0])
    else: test1 = []

    if mutation_paths0:
        train = [tokenizer.convert_tokens_to_ids(frag) for frag in mutation_paths0]
        print("train_complete")
        if train:
            print(get_max_elements_per_dimension(train))
            print(train[0])
    else: train = []
else:
    print("mutation_paths or length is empty. Skipping data filtering and tokenization for train/test splits.")
    mutation_paths0, mutation_paths1, mutation_paths2, mutation_paths3 = [], [], [], []
    test3, test2, test1, train = [], [], [], []


# %%

def trim_lists_from_start(list_of_lists, target_length):
    if not isinstance(list_of_lists, list):
        raise TypeError("第一引数はリストである必要があります。")
    if not isinstance(target_length, int):
        raise TypeError("第二引数は整数である必要があります。")
    if target_length < 0:
        print("警告: target_lengthが0未満です。すべての内部リストは空になります。")
        target_length = 0 

    processed_list = []
    for inner_list in list_of_lists:
        if not isinstance(inner_list, list):
            print(f"警告: 内部要素がリストではありません。スキップします: {inner_list}")
            continue 

        current_length = len(inner_list)

        if current_length > target_length:
            start_index = current_length - target_length
            processed_list.append(inner_list[start_index:])
        else:
            processed_list.append(copy.copy(inner_list))

    return processed_list

# %%
# データが準備できているか確認
if train:
    x_train,y_train = separate_XY(train,1)
else:
    x_train, y_train = [], []
    print("Train data is empty. Skipping separate_XY for train.")

if test1:
    x_test1,y_test1 = separate_XY(test1,1)
else:
    x_test1, y_test1 = [], []
    print("Test1 data is empty. Skipping separate_XY for test1.")

if test2:
    x_test2,y_test2 = separate_XY(test2,1)
else:
    x_test2, y_test2 = [], []
    print("Test2 data is empty. Skipping separate_XY for test2.")

if test3:
    x_test3,y_test3 = separate_XY(test3,1)
else:
    x_test3, y_test3 = [], []
    print("Test3 data is empty. Skipping separate_XY for test3.")

# === 入力系列とラベルの重複チェック（デバッグ） ===
def check_input_label_overlap(x, y, label):
    overlap_count = 0
    for xi, yi in zip(x, y):
        if len(xi) > 0 and xi[-1] == yi:
            print(f"[DEBUG][{label}] 入力系列の末尾とラベルが一致: {xi[-1]} == {yi}")
            overlap_count += 1
    print(f"[DEBUG][{label}] 入力系列末尾とラベルが一致した件数: {overlap_count} / {len(x)}")

if x_train and y_train:
    check_input_label_overlap(x_train, y_train, "train")
if x_test1 and y_test1:
    check_input_label_overlap(x_test1, y_test1, "test1")
if x_test2 and y_test2:
    check_input_label_overlap(x_test2, y_test2, "test2")
if x_test3 and y_test3:
    check_input_label_overlap(x_test3, y_test3, "test3")


if x_train: print(get_max_elements_per_dimension(x_train),get_max_elements_per_dimension(y_train))
if x_train: print(get_min_elements_per_dimension(x_train),get_min_elements_per_dimension(y_train))
if x_test1: print(get_max_elements_per_dimension(x_test1),get_max_elements_per_dimension(y_test1))
if x_test1: print(get_min_elements_per_dimension(x_test1),get_min_elements_per_dimension(y_test1))
if x_test2: print(get_max_elements_per_dimension(x_test2),get_max_elements_per_dimension(y_test2))
if x_test2: print(get_min_elements_per_dimension(x_test2),get_min_elements_per_dimension(y_test2))
if x_test3: print(get_max_elements_per_dimension(x_test3),get_max_elements_per_dimension(y_test3))
if x_test3: print(get_min_elements_per_dimension(x_test3),get_min_elements_per_dimension(y_test3))


if x_train: # x_train が空でないことを確認
    # get_max_elements_per_dimension は [全体リストの長さ, 内部リストの最大長] のようなリストを返す
    # ここでは内部リストの最大長 (シーケンス長) を target_length としたい
    max_dims = get_max_elements_per_dimension(x_train)
    if len(max_dims) > 1:
        target_length = max_dims[1] # 内部リストの最大長
    else:
        # x_train が空、または1次元リストのみの場合のフォールバック
        target_length = 40 # デフォルト値 (例: mutation_paths0 の上限)
        print(f"Warning: Could not determine target_length from x_train dimensions. Using default: {target_length}")
else:
    target_length = 40 # x_train が空の場合のデフォルト値
    print(f"Warning: x_train is empty. Using default target_length: {target_length}")


print(f"Target length for trimming: {target_length}")

if x_test3: print(x_test3[0] if x_test3 else "x_test3 is empty") # x_test3が空でないか確認

X_train = trim_lists_from_start(x_train, target_length) if x_train else []
X_test3 = trim_lists_from_start(x_test3, target_length) if x_test3 else []
X_test2 = trim_lists_from_start(x_test2, target_length) if x_test2 else []
X_test1 = trim_lists_from_start(x_test1, target_length) if x_test1 else []

if X_test3: print(X_test3[0] if X_test3 else "X_test3 is empty after trimming")

if X_train: print(get_max_elements_per_dimension(X_train),get_max_elements_per_dimension(y_train))
# ... (他のprint文も同様に存在チェックを追加)

# %%
class MutationDatasetForLM(Dataset):
    def __init__(self, fragments, target_labels, tokenizer, max_len):
        self.fragments = fragments
        self.target_labels = target_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = tokenizer.pad_token_id

        for i, frag in enumerate(self.fragments):
            if len(frag) > self.max_len:
                raise ValueError(
                    f"Fragment at index {i} has length {len(frag)}, "
                    f"which exceeds max_len {self.max_len}. "
                    "Inputs must be pre-truncated."
                )

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        fragment = self.fragments[idx]
        target_label = self.target_labels[idx]

        current_len = len(fragment)
        padding_len = self.max_len - current_len
        if padding_len < 0:
            raise ValueError(
                f"Padding length is negative ({padding_len}) for fragment at index {idx}. "
                f"Fragment length ({current_len}) exceeds max_len ({self.max_len})."
            )

        input_ids = fragment + [self.pad_token_id] * padding_len
        attention_mask = [1] * current_len + [0] * padding_len

        labels = [-100] * self.max_len
        if current_len > 0:
            last_token_idx = current_len - 1
            labels[last_token_idx] = target_label

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# %%
# ### 追加ここから ###
def get_gpu_memory_usage():
    """現在のGPUメモリ使用量と総メモリ量を取得する (MB単位)"""
    if not torch.cuda.is_available():
        return 0, 0, 0 # free, total, allocated
    free_memory, total_memory_info = torch.cuda.mem_get_info(0)
    free_memory_mb = free_memory / (1024**2)
    total_memory_info_mb = total_memory_info / (1024**2)
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**2)
    cached_memory = torch.cuda.memory_reserved(0) / (1024**2)

    print(f"  Total GPU Memory: {total_memory_info_mb:.2f} MB")
    print(f"  Allocated GPU Memory: {allocated_memory:.2f} MB")
    print(f"  Cached GPU Memory (Reserved): {cached_memory:.2f} MB")
    print(f"  Free GPU Memory (from mem_get_info): {free_memory_mb:.2f} MB")
    return free_memory_mb, total_memory_info_mb, allocated_memory

def estimate_memory_per_sample_bart(model, tokenizer, max_len, device, optimizer_for_estimation=None):
    """
    BartForConditionalGenerationモデルの1サンプルあたりのGPUメモリ消費量（学習時）を見積もる。
    """
    if not torch.cuda.is_available():
        print("GPU not available for memory estimation.")
        return float('inf')

    model.train()
    # オプティマイザのstateもメモリに影響するため、推定用のオプティマイザを渡すか、
    # グローバルなオプティマイザをここで使うか選択。
    # ここでは、もしメインのオプティマイザがあればそれを使う。なければダミーを作成。
    if optimizer_for_estimation is None:
        # ダミーオプティマイザ（実際の学習ではメインのオプティマイザを使うべき）
        temp_optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        print("Warning: Using a temporary optimizer for memory estimation. For more accuracy, pass the main optimizer.")
    else:
        temp_optimizer = optimizer_for_estimation


    # ダミー入力とラベルの作成 (バッチサイズ1で)
    # max_len-1 のトークン列 + 1つの予測対象トークンを想定
    # 実際の入力に近い形にする
    dummy_input_ids_list = [tokenizer.bos_token_id] + [tokenizer.eos_token_id] * (max_len - 2) # 適当なトークン列
    if len(dummy_input_ids_list) >= max_len:
        dummy_input_ids_list = dummy_input_ids_list[:max_len-1] # 最後のトークンのためのスペースを確保

    # labels の最後の有効なトークンの位置は、入力シーケンスの最後のトークンの位置になる
    # 例: input_ids = [<s>, A, B, C, </s>, <pad>, <pad>]
    #     labels  = [-100, -100, -100, -100, Target_for_C, -100, -100]
    # ここでの推定では、input_ids の最後の非パディングトークンに対する予測をシミュレート

    effective_input_len = len(dummy_input_ids_list)
    padding_len = max_len - effective_input_len
    
    dummy_input_ids = torch.tensor([dummy_input_ids_list + [tokenizer.pad_token_id] * padding_len], dtype=torch.long).to(device)
    dummy_attention_mask = torch.tensor([[1] * effective_input_len + [0] * padding_len], dtype=torch.long).to(device)
    
    # ラベルの作成: 最後の有効トークン位置 (effective_input_len - 1) にダミーターゲットを設定
    dummy_labels_list = [-100] * max_len
    if effective_input_len > 0:
        dummy_labels_list[effective_input_len -1] = tokenizer.eos_token_id # 何か適当なID
    dummy_labels = torch.tensor([dummy_labels_list], dtype=torch.long).to(device)


    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory_peak_estimation = torch.cuda.memory_allocated(device)

    # 学習の1ステップを実行
    temp_optimizer.zero_grad(set_to_none=True) # set_to_none=True はメモリ効率が良い場合がある
    outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, labels=dummy_labels)
    loss = outputs.loss
    if loss is not None : # 損失が計算できた場合のみ backward
        loss.backward()
        temp_optimizer.step()
    else:
        print("Warning: Loss was None during memory estimation. Gradients might not be calculated.")


    peak_memory_one_sample = torch.cuda.max_memory_allocated(device) - initial_memory_peak_estimation
    
    del dummy_input_ids, dummy_attention_mask, dummy_labels, outputs, loss
    if optimizer_for_estimation is None:
        del temp_optimizer # 一時的なオプティマイザを削除
    torch.cuda.empty_cache()

    memory_per_sample_mb = peak_memory_one_sample / (1024**2)
    if memory_per_sample_mb < 0: # まれに発生することがあるので対処
        print(f"Warning: Estimated memory per sample is negative ({memory_per_sample_mb:.4f} MB). This might indicate an issue. Defaulting to a small positive value.")
        memory_per_sample_mb = 0.1 # 小さな正の値
    elif memory_per_sample_mb == 0:
         print(f"Warning: Estimated memory per sample is zero. Defaulting to a small positive value.")
         memory_per_sample_mb = 0.1 # ゼロの場合も小さな正の値

    print(f"Estimated peak memory increase for one sample (Bart model): {memory_per_sample_mb:.4f} MB")
    return memory_per_sample_mb

def determine_batch_size_for_bart(model, tokenizer, max_len, device, optimizer_for_estimation, safety_margin_gb=1.0, min_batch_size=1, default_batch_size=16):
    if not torch.cuda.is_available():
        print("GPU not available. Cannot determine batch size based on GPU memory.")
        return default_batch_size

    print("\n--- Determining Optimal Batch Size for BART ---")
    _ = model.to(device) # モデルをGPUに

    print("Initial GPU memory usage (after model to device):")
    _, _, initial_allocated_mb = get_gpu_memory_usage()


    memory_per_sample_mb = estimate_memory_per_sample_bart(model, tokenizer, max_len, device, optimizer_for_estimation)

    if memory_per_sample_mb == 0 or memory_per_sample_mb == float('inf') or memory_per_sample_mb <= 0.001: # 非常に小さい値もエラーと見なす
        print(f"Could not reliably estimate memory per sample ({memory_per_sample_mb} MB). Using default batch size: {default_batch_size}")
        return default_batch_size

    print("\nGPU memory usage after estimation attempt:")
    available_gpu_memory_mb, total_gpu_memory_mb, allocated_after_estimation_mb = get_gpu_memory_usage()

    # モデル自体のメモリ + PyTorchのオーバーヘッドを考慮した利用可能メモリ
    # 空きメモリから、現在の割り当て済みメモリ（モデルロード分など）を引くのではなく、
    # 空きメモリをそのまま使うアプローチで一度試す。
    # その上で、memory_per_sample_mb が「追加1サンプル」の純粋な増加分として機能することを期待する。
    # より保守的には: usable_for_batches_mb = available_gpu_memory_mb - (allocated_after_estimation_mb - initial_allocated_mb)
    # ここでは、空きメモリをベースにする。
    usable_for_batches_mb = available_gpu_memory_mb

    safety_margin_mb = safety_margin_gb * 1024
    
    # バッチデータに使える実際のメモリ
    effective_usable_memory_mb = usable_for_batches_mb - safety_margin_mb
    print(f"Total GPU memory: {total_gpu_memory_mb:.2f} MB")
    print(f"Available GPU memory (before safety margin): {available_gpu_memory_mb:.2f} MB")
    print(f"Allocated GPU memory (model, etc.): {allocated_after_estimation_mb:.2f} MB")
    print(f"Safety margin: {safety_margin_mb:.2f} MB")
    print(f"Effectively usable memory for batches: {effective_usable_memory_mb:.2f} MB")

    if effective_usable_memory_mb <= 0:
        print(f"Not enough usable GPU memory ({effective_usable_memory_mb:.2f} MB) after applying safety margin. Using minimum batch size: {min_batch_size}")
        return min_batch_size

    estimated_batch_size = int(effective_usable_memory_mb / memory_per_sample_mb)

    print(f"Estimated memory per sample: {memory_per_sample_mb:.4f} MB")
    print(f"Calculated batch size: {estimated_batch_size}")

    if estimated_batch_size < min_batch_size:
        print(f"Calculated batch size ({estimated_batch_size}) is less than minimum ({min_batch_size}). Setting to {min_batch_size}.")
        estimated_batch_size = min_batch_size
    
    # 念のため、極端に大きなバッチサイズにならないように上限を設けることも検討 (例: 256)
    # if estimated_batch_size > 256:
    #     print(f"Calculated batch size ({estimated_batch_size}) is very large. Capping at 256.")
    #     estimated_batch_size = 256


    print(f"--- Optimal Batch Size Determination Complete: {estimated_batch_size} ---")
    return estimated_batch_size
# ### 追加ここまで ###

# 学習データとテストデータに分割 (必要に応じて)
if X_train and y_train: # データが存在するか確認
    # train_test_splitはリストが空だとエラーになることがある
    if len(X_train) > 1 and len(y_train) > 1 : # 少なくとも2サンプル必要
        train_fragments, valid_fragments, train_labels, valid_labels = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42 # 再現性のためにrandom_stateを追加
        )
    elif len(X_train) == 1 and len(y_train) == 1: # 1サンプルの場合は検証データなし
        print("Warning: Only one sample in X_train. Using it for training, validation will be empty.")
        train_fragments, valid_fragments, train_labels, valid_labels = X_train, [], y_train, []
    else: # データが不足している場合
        print("Warning: Not enough data for train/validation split. Training and validation will be empty.")
        train_fragments, valid_fragments, train_labels, valid_labels = [], [], [], []
else:
    print("X_train or y_train is empty. Skipping train_test_split.")
    train_fragments, valid_fragments, train_labels, valid_labels = [], [], [], []


MAX_LEN = target_length # これは以前の target_length を使う

# データセットとデータローダーの準備 (データがある場合のみ)
if train_fragments and train_labels:
    train_datasetForLM = MutationDatasetForLM(train_fragments, train_labels, tokenizer, MAX_LEN)
    # BATCH_SIZE = 64 ### 変更前 ###
else:
    train_datasetForLM = None # 空のデータセット
    print("Training dataset is empty.")

if valid_fragments and valid_labels:
    valid_datasetForLM = MutationDatasetForLM(valid_fragments, valid_labels, tokenizer, MAX_LEN)
else:
    valid_datasetForLM = None
    print("Validation dataset is empty.")

# %%
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.resize_token_embeddings(len(tokenizer)) # トークナイザーの語彙サイズに合わせてモデルの埋め込み層をリサイズ
model.config.pad_token_id = tokenizer.pad_token_id # モデル設定にパディングIDを設定

# モデルを適切なデバイスに移動
model.to(device)

# オプティマイザはバッチサイズ決定後、またはここで定義しても良い
# estimate_memory_per_sample_bart に渡すためにここで定義
optimizer = optim.AdamW(model.parameters(), lr=5e-5)


# ### バッチサイズ決定処理の呼び出し ###
if device.type == 'cuda' and train_datasetForLM is not None and len(train_datasetForLM) > 0 :
    # MAX_LEN はシーケンス長
    # optimizer は推定のためにも使う
    safety_margin_gb = 0.5 # GPUメモリの2GBを安全マージンとして確保 (環境に応じて調整)
    min_batch_size = 1
    default_batch_size_if_error = 16 # エラー時のデフォルト

    # この時点でoptimizerは初期化されている必要がある
    dynamic_batch_size = determine_batch_size_for_bart(
        model,
        tokenizer,
        MAX_LEN,
        device,
        optimizer_for_estimation=optimizer, # メインのオプティマイザを渡す
        safety_margin_gb=safety_margin_gb,
        min_batch_size=min_batch_size,
        default_batch_size=default_batch_size_if_error
    )
    BATCH_SIZE = dynamic_batch_size
    print(f"Using dynamically determined BATCH_SIZE: {BATCH_SIZE}")
elif train_datasetForLM is not None and len(train_datasetForLM) > 0:
    BATCH_SIZE = 32 # CPUの場合のデフォルトバッチサイズ
    print(f"Using CPU or no training data. Default BATCH_SIZE: {BATCH_SIZE}")
else:
    BATCH_SIZE = 1 # データがない場合は形式的に1など
    print("No training data to create DataLoader. Setting BATCH_SIZE to 1 nominally.")


if train_datasetForLM:
    train_dataloader = DataLoader(train_datasetForLM, batch_size=BATCH_SIZE, shuffle=True)
else:
    train_dataloader = None

if valid_datasetForLM:
    valid_dataloader = DataLoader(valid_datasetForLM, batch_size=BATCH_SIZE) # 検証用はシャッフル不要
else:
    valid_dataloader = None


num_epochs = 50

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model_to_save):
        score = -val_loss 

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_to_save)
        elif score < self.best_score + self.delta: 
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: 
            self.best_score = score
            self.save_checkpoint(val_loss, model_to_save)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_to_save):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.path} ...')
        try:
            # 保存先ディレクトリが存在するか確認し、なければ作成
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            torch.save(model_to_save.state_dict(), self.path)
        except Exception as e:
            self.trace_func(f"Error saving model checkpoint: {e}")
        self.val_loss_min = val_loss

# %%
train_losses = []
val_losses = []
val_accuracies = []

# 保存先のディレクトリを指定
output_dir_base = "../model/20250502_bart2/" # ベースのディレクトリ
model_version = "model1" # モデルバージョンなど
output_dir = os.path.join(output_dir_base, model_version)

# 保存先のディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")


early_stopping_checkpoint_path = os.path.join(output_dir, "best_early_stop_model.pt")
early_stopper = EarlyStopping(patience=5, verbose=True, path=early_stopping_checkpoint_path, delta=0.001)


# 学習ループはデータがある場合のみ実行
if train_dataloader and valid_dataloader:
    for epoch in range(num_epochs):
        # --- トレーニングフェーズ ---
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch in train_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            if loss is not None: # 損失がNoneでないことを確認
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                train_progress_bar.set_postfix({'loss': loss.item()})
            else:
                print("Warning: Training loss is None for a batch.")


        avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        train_losses.append(avg_train_loss) 

        # --- 評価フェーズ ---
        model.eval()
        total_val_loss = 0
        total_correct_preds = 0
        total_targets = 0
        val_progress_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Eval]", leave=False)
        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                if loss is not None:
                    total_val_loss += loss.item()

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    target_mask = labels!= -100
                    correct_preds = torch.sum((predictions == labels) & target_mask)
                    num_targets = torch.sum(target_mask)
                    total_correct_preds += correct_preds.item()
                    total_targets += num_targets.item()
                    val_progress_bar.set_postfix({'val_loss': loss.item()})
                else:
                    print("Warning: Validation loss is None for a batch.")


        avg_val_loss = total_val_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0
        accuracy = total_correct_preds / total_targets if total_targets > 0 else 0
        val_losses.append(avg_val_loss) 
        val_accuracies.append(accuracy) 

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}")

        # Early Stopping の呼び出し
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
    print("Training finished or stopped early!")

    # ### Early Stopping ###
    if os.path.exists(early_stopper.path):
        print(f"Loading best model weights from {early_stopper.path}")
        try:
            model.load_state_dict(torch.load(early_stopper.path, map_location=device))
        except Exception as e:
            print(f"Error loading best model weights: {e}")
    else:
        print(f"Warning: Best model checkpoint {early_stopper.path} not found. Using the last model state.")

else:
    print("Skipping training loop as there is no data in train_dataloader or valid_dataloader.")


# --- 損失グラフの描画 ---
actual_epochs = len(train_losses) 
if actual_epochs > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss.png")) # 保存パスを修正
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy.png")) # 保存パスを修正
    plt.show()

else:
    print("No training data to plot.")

# 保存するモデルとトークナイザー (これらは学習済みである必要があります)
model_to_save = model
tokenizer_to_save = tokenizer

# モデルを保存
model_to_save.save_pretrained(output_dir)

# トークナイザーを保存
tokenizer_to_save.save_pretrained(output_dir)

print(f"モデルとトークナイザーを '{output_dir}' に保存しました。")