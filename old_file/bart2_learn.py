# %%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from bertviz import head_view, model_view
import matplotlib.pyplot as plt
import numpy as np

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

strain = ['B.1.1.7','BA.1','B.1.617.2']

for s in strain:
    dir = 'sequences_20241017_'+s+'_random/'
    folder_num = count_numeric_subfolders(dir)

    for num in range(folder_num):
        #f = open(dir+str(num)+'/mutation_paths_'+strain+'.tsv', 'r',encoding="shift-jis")
        f = open(dir+str(num)+'/mutation_paths_'+s+'.tsv', 'r',encoding="utf-8_sig")
        print(dir+str(num)+'/mutation_paths_'+s+'.tsv')
        datalist = f.readlines()
        f.close()
        
        for i in range(1,len(datalist)):
            data = datalist[i].split('\t')
            name.append(data[0])
            length.append(int(data[1]))
            mutation_paths.append(data[2].rstrip().split('>'))

strain = ['P.1','B.1.351','B.1.1.529']

for s in strain:
    dir = 'sequences_20241017_'+s+'/'

    #f = open(dir+str(num)+'/mutation_paths_'+strain+'.tsv', 'r',encoding="shift-jis")
    f = open(dir+'mutation_paths_'+s+'.tsv', 'r',encoding="utf-8_sig")
    print(dir+'mutation_paths_'+s+'.tsv')
    datalist = f.readlines()
    f.close()
    
    for i in range(1,len(datalist)):
        data = datalist[i].split('\t')
        name.append(data[0])
        length.append(int(data[1]))
        mutation_paths.append(data[2].rstrip().split('>'))

for i in range(5):
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
    
    # 次元数に対応するリストを拡張
    if dimension >= len(max_elements):
        max_elements.append(0)
    
    # 現在の次元の最大要素数を更新
    max_elements[dimension] = max(max_elements[dimension], len(nested_list))
    
    # 内部がリストの場合は再帰
    for item in nested_list:
        if isinstance(item, list):
            get_max_elements_per_dimension(item, dimension + 1, max_elements)
    
    return max_elements



def get_min_elements_per_dimension(nested_list, dimension=0, min_elements=None):
    if min_elements is None:
        min_elements = []
    
    # 必要に応じてmin_elementsを拡張
    if dimension >= len(min_elements):
        min_elements.append(float('inf'))
    
    # 現在の次元の最小要素数を更新
    min_elements[dimension] = min(min_elements[dimension], len(nested_list))
    
    # 内部がリストの場合は再帰的に探索
    for item in nested_list:
        if isinstance(item, list):
            get_min_elements_per_dimension(item, dimension + 1, min_elements)
    
    # 最小要素数が無限大の場合（空リストのみの場合）を0に修正
    min_elements = [0 if val == float('inf') else val for val in min_elements]
    
    return min_elements

def separate_XY(data, n):
    x_list = []
    y_list = []
    for item in data: # item はトークンIDのリスト [id1, id2, ..., id_last]
        if len(item) > n:
            x = item[:-n]
            y_part = item[-n:] # 例: n=1 なら [id_last] というリストになる

            # ★★★ 修正点 ★★★
            # y_part がリストの場合、中の要素（整数）を取り出す
            if isinstance(y_part, list) and len(y_part) == n:
                 # n=1 の場合は y_part[0] でOK
                 # 一般的な場合、n個の要素を持つリストになるかもしれないが、
                 # 今回のタスク (最後の1トークン予測) では n=1 のはず
                if n == 1:
                     y = y_part[0] # リストではなく整数を取り出す
                else:
                     # n > 1 の場合の処理 (必要に応じて)
                     y = y_part # または別の処理
            else:
                # 予期しない形式の場合の処理 (エラーまたはスキップ)
                # continue # 例えばスキップ
                raise ValueError(f"Unexpected format for y_part: {y_part}")


            x_list.append(x)
            y_list.append(y) # ★ ここで整数が追加されるようにする
        else:
            # n個の要素を分離できない場合の処理
            pass # スキップなど
    return x_list, y_list

# %%
mutation_paths,name,length = filter_co_occur(mutation_paths,name,length,5,len(mutation_paths))

# ラベルエンコード
flat_data = [item for series in mutation_paths for item in series]
print(len(flat_data))
print(len(list(set(flat_data))))


# %%
length_count = []
length_i = []
for i in range(0,max(length)):
    length_count.append(length.count(i))
    length_i.append(i)
plt.figure(figsize=[12,5])
plt.bar(length_i,length_count)
plt.xticks()
plt.show()

for i in length_i:
    if(length_count[i] != 0):
        print(i,length_count[i])

# %%
strain = ['BA.1']
BA_1_len = []

for s in strain:
    dir = 'sequences_20241017_'+s+'_random/'
    folder_num = count_numeric_subfolders(dir)

    for num in range(folder_num):
        #f = open(dir+str(num)+'/mutation_paths_'+strain+'.tsv', 'r',encoding="shift-jis")
        f = open(dir+str(num)+'/mutation_paths_'+s+'.tsv', 'r',encoding="utf-8_sig")
        print(dir+str(num)+'/mutation_paths_'+s+'.tsv')
        datalist = f.readlines()
        f.close()
        
        for i in range(1,len(datalist)):
            data = datalist[i].split('\t')
            BA_1_len.append(int(data[1]))

# %%
length_count = []
length_i = []
for i in range(0,max(BA_1_len)):
    length_count.append(BA_1_len.count(i))
    length_i.append(i)
plt.figure(figsize=[12,5])
plt.bar(length_i,length_count)
plt.xticks()
plt.show()

for i in length_i:
    if(length_count[i] != 0):
        print(i,length_count[i])


# %%
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokenizer.add_tokens(list(set(flat_data)))
print("tokenizer complete")
print(get_max_elements_per_dimension(mutation_paths))


# %%
def filter_length(data,length,low,up):
    filted_data = []
    for i in range(len(length)):
        if(length[i]>=low):
            if(length[i]>up):
                filted_data.append(data[i][:up])
            else:
                filted_data.append(data[i])
    return filted_data

# %%
mutation_paths0 = filter_length(mutation_paths,length,0,40)
mutation_paths1 = filter_length(mutation_paths,length,41,45)
mutation_paths2 = filter_length(mutation_paths,length,46,50)
mutation_paths3 = filter_length(mutation_paths,length,51,100)
print(get_max_elements_per_dimension(mutation_paths0))
print(get_max_elements_per_dimension(mutation_paths1))
print(get_max_elements_per_dimension(mutation_paths2))
print(get_max_elements_per_dimension(mutation_paths3))

test3 = [tokenizer.convert_tokens_to_ids(frag) for frag in mutation_paths3]
print("test3_complete")
print(get_max_elements_per_dimension(test3))
print(test3[0])

test2 = [tokenizer.convert_tokens_to_ids(frag) for frag in mutation_paths2]
print("test2_complete")
print(get_max_elements_per_dimension(test2))
print(test2[0])

test1 = [tokenizer.convert_tokens_to_ids(frag) for frag in mutation_paths1]
print("test1_complete")
print(get_max_elements_per_dimension(test1))
print(test1[0])

train = [tokenizer.convert_tokens_to_ids(frag) for frag in mutation_paths0]
print("train_complete")
print(get_max_elements_per_dimension(train))
print(train[0])

# %%
import copy # 元のリストを変更しないようにするため

def trim_lists_from_start(list_of_lists, target_length):
  """
  2次元リスト内の各1次元リストを指定した長さになるように、
  先頭から要素を削除します。

  Args:
    list_of_lists (list[list]): 処理対象の2次元リスト。
                                 内部の1次元リストの長さはバラバラで構いません。
    target_length (int): 各1次元リストの目標の最大長。
                         0以下の値を指定した場合、空のリストになります。

  Returns:
    list[list]: 処理後の新しい2次元リスト。
                元の list_of_lists オブジェクトは変更されません。
  """
  if not isinstance(list_of_lists, list):
      raise TypeError("第一引数はリストである必要があります。")
  if not isinstance(target_length, int):
      raise TypeError("第二引数は整数である必要があります。")
  if target_length < 0:
      print("警告: target_lengthが0未満です。すべての内部リストは空になります。")
      target_length = 0 # 負の長さは扱えないため0とする

  processed_list = []
  for inner_list in list_of_lists:
    if not isinstance(inner_list, list):
        # 内部要素がリストでない場合の処理（ここではスキップするかエラーにするか選択）
        print(f"警告: 内部要素がリストではありません。スキップします: {inner_list}")
        # processed_list.append(inner_list) # そのまま追加する場合
        continue # スキップする場合
        # raise TypeError("内部要素はすべてリストである必要があります。") # エラーにする場合

    current_length = len(inner_list)

    if current_length > target_length:
      # リストの長さが目標より長い場合、先頭から要素を削除
      # 削除する要素数 = current_length - target_length
      # 残す要素の開始インデックス = current_length - target_length
      start_index = current_length - target_length
      # スライスを使って、指定したインデックスから末尾までを取得
      processed_list.append(inner_list[start_index:])
    else:
      # リストの長さが目標以下の場合、そのまま（コピーして）追加
      # 元のリストの内部リストへの変更を防ぐためコピーする
      processed_list.append(copy.copy(inner_list))

  return processed_list

# %%
x_train,y_train = separate_XY(train,1)
x_test1,y_test1 = separate_XY(test1,1)
x_test2,y_test2 = separate_XY(test2,1)
x_test3,y_test3 = separate_XY(test3,1)

print(get_max_elements_per_dimension(x_train),get_max_elements_per_dimension(y_train))
print(get_min_elements_per_dimension(x_train),get_min_elements_per_dimension(y_train))
print(get_max_elements_per_dimension(x_test1),get_max_elements_per_dimension(y_test1))
print(get_min_elements_per_dimension(x_test1),get_min_elements_per_dimension(y_test1))
print(get_max_elements_per_dimension(x_test2),get_max_elements_per_dimension(y_test2))
print(get_min_elements_per_dimension(x_test2),get_min_elements_per_dimension(y_test2))
print(get_max_elements_per_dimension(x_test3),get_max_elements_per_dimension(y_test3))
print(get_min_elements_per_dimension(x_test3),get_min_elements_per_dimension(y_test3))

target_length = get_max_elements_per_dimension(x_train)[1]

print(x_test3[0])

X_train = trim_lists_from_start(x_train, target_length)
X_test3 = trim_lists_from_start(x_test3, target_length)
X_test2 = trim_lists_from_start(x_test2, target_length)
X_test1 = trim_lists_from_start(x_test1, target_length)

print(X_test3[0])

print(get_max_elements_per_dimension(X_train),get_max_elements_per_dimension(y_train))
print(get_min_elements_per_dimension(X_train),get_min_elements_per_dimension(y_train))
print(get_max_elements_per_dimension(X_test1),get_max_elements_per_dimension(y_test1))
print(get_min_elements_per_dimension(X_train),get_min_elements_per_dimension(y_train))
print(get_max_elements_per_dimension(X_test1),get_max_elements_per_dimension(y_test1))
print(get_min_elements_per_dimension(X_test1),get_min_elements_per_dimension(y_test1))
print(get_max_elements_per_dimension(X_test2),get_max_elements_per_dimension(y_test2))
print(get_min_elements_per_dimension(X_test2),get_min_elements_per_dimension(y_test2))
print(get_max_elements_per_dimension(X_test3),get_max_elements_per_dimension(y_test3))
print(get_min_elements_per_dimension(X_test3),get_min_elements_per_dimension(y_test3))

# %%
class MutationDatasetForLM(Dataset):
    """
    事前に入力フラグメントとそのターゲットラベルを受け取り、
    モデル入力用にパディングとラベル形式の調整を行うDataset。
    入力フラグメントは max_len 以下である前提。
    """
    def __init__(self, fragments, target_labels, tokenizer, max_len):
        """
        Args:
            fragments (list[list[int]]): トークンIDのリストのリスト (各要素は max_len 以下であること)
            target_labels (list[int]): 各フラグメントに対応するターゲットトークンIDのリスト
            tokenizer: 使用するトークナイザー (pad_token_id を利用)
            max_len (int): パディング後の最大シーケンス長
        """
        self.fragments = fragments
        self.target_labels = target_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = tokenizer.pad_token_id # パディングIDをキャッシュ

        # 事前チェック: 入力フラグメントが max_len を超えていないか確認
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
        # 入力 fragment は max_len 以下である前提
        fragment = self.fragments[idx]
        target_label = self.target_labels[idx]

        # パディング処理 (シーケンス長を max_len に揃える)
        current_len = len(fragment)
        padding_len = self.max_len - current_len
        if padding_len < 0:
             # このエラーは __init__ でのチェックがあれば通常発生しないはず
             raise ValueError(
                 f"Padding length is negative ({padding_len}) for fragment at index {idx}. "
                 f"Fragment length ({current_len}) exceeds max_len ({self.max_len})."
             )

        input_ids = fragment + [self.pad_token_id] * padding_len
        attention_mask = [1] * current_len + [0] * padding_len

        # ラベルテンソルの作成 (-100 でパディングし、予測対象位置にラベルを設定)
        labels = [-100] * self.max_len
        if current_len > 0:
             # 最後の有効なトークンの位置にラベルを設定
             # (注意: この実装は「最後のトークン位置で target_label を予測」するタスク用)
             last_token_idx = current_len - 1
             # last_token_idx は常に max_len より小さいはず (padding_len >= 0 のため)
             labels[last_token_idx] = target_label

        # テンソルに変換して返す
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# %%
# 学習データとテストデータに分割 (必要に応じて)
train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(
    X_train, y_train, test_size=0.2
)

MAX_LEN = target_length

train_datasetForLM = MutationDatasetForLM(train_dataset, train_labels, tokenizer, MAX_LEN)
valid_datasetForLM = MutationDatasetForLM(valid_dataset, valid_labels, tokenizer, MAX_LEN)

BATCH_SIZE = 128
train_dataloader = DataLoader(train_datasetForLM, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_datasetForLM, batch_size=BATCH_SIZE)


# %%
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# モデルを適切なデバイスに移動
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 50

# 保存先のディレクトリを指定
output_dir = "../model/20250502_bart2/model1/"

# 保存先のディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
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
        score = -val_loss # 損失は低いほど良いので、スコアは負の値にする

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_to_save)
        elif score < self.best_score + self.delta: # 改善していない (スコアが上がっていない)
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # スコアが改善した
            self.best_score = score
            self.save_checkpoint(val_loss, model_to_save)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_to_save):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.path} ...')
        try:
            torch.save(model_to_save.state_dict(), self.path)
        except Exception as e:
            self.trace_func(f"Error saving model checkpoint: {e}")
        self.val_loss_min = val_loss

# %%
train_losses = []
val_losses = []
val_accuracies = []

early_stopping_checkpoint_path = output_dir+"best_early_stop_model.pt"
early_stopper = EarlyStopping(patience=5, verbose=True, path=early_stopping_checkpoint_path, delta=0.001)

for epoch in range(num_epochs):
    # --- トレーニングフェーズ ---
    model.train()
    total_train_loss = 0
    train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
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
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        # tqdmに進捗バーと現在の損失を表示させる (オプション)
        train_progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss) # トレーニング損失を記録

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
            total_val_loss += loss.item()

            # 精度計算
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            target_mask = labels!= -100
            correct_preds = torch.sum((predictions == labels) & target_mask)
            num_targets = torch.sum(target_mask)
            total_correct_preds += correct_preds.item()
            total_targets += num_targets.item()

            # tqdmに進捗バーと現在の検証損失を表示させる (オプション)
            val_progress_bar.set_postfix({'val_loss': loss.item()})

    avg_val_loss = total_val_loss / len(valid_dataloader)
    accuracy = total_correct_preds / total_targets if total_targets > 0 else 0
    val_losses.append(avg_val_loss) # 検証損失を記録
    val_accuracies.append(accuracy) # 検証精度を記録

    # エポックごとの結果を表示
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}")

print("Training finished!")

# ### Early Stopping ###
# 最良モデルの重みをロード (EarlyStoppingが有効で、チェックポイントが存在する場合)
if os.path.exists(early_stopper.path):
    print(f"Loading best model weights from {early_stopper.path}")
    try:
        model.load_state_dict(torch.load(early_stopper.path, map_location=device))
    except Exception as e:
        print(f"Error loading best model weights: {e}")
else:
    print(f"Warning: Best model checkpoint {early_stopper.path} not found. Using the last model state.")

# --- 損失グラフの描画 ---
actual_epochs = len(train_losses) # 早期終了した場合、num_epochsより少なくなる
if actual_epochs > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(output_dir+"loss.png")

    # --- (オプション) 精度のグラフ描画 ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(output_dir+"accuracy.png")
else:
    print("No training data to plot.")

# 保存するモデルとトークナイザー (これらは学習済みである必要があります)
model_to_save = model
tokenizer_to_save = tokenizer

# モデルを保存
model_to_save.save_pretrained(output_dir)

# トークナイザーを保存 (モデルと一緒に保存されることもありますが、明示的に保存しておくと確実です)
tokenizer_to_save.save_pretrained(output_dir)

print(f"モデルとトークナイザーを '{output_dir}' に保存しました。")