# 機械学習ライブラリのインポート
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 特徴量名の定義
ts_name = [f"ts_{x}" for x in range(1, 101)]

base = ["A", "C", "G", "T"]
base_mut_name = [f"{b1}>{b2}" for b1 in base for b2 in base if b1 != b2]
base_pos_name = [f"b_{i}" for i in range(1, 30001)]

amino = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
          "M", "N", "P", "Q", "R", "S",
          "T", "V", "W", "Y", "*", "n"]
amino_mut_name = [f"{a1}>{a2}" for a1 in amino for a2 in amino]
amino_pos_name = [f"a_{i}" for i in range(0, 30001)]

mutation_type = ["syno", "non-syno"]

protein_name = [
    "non_coding1", "nsp1", "nsp2", "nsp3", "nsp4", "nsp5", "nsp6", "nsp7", "nsp8", "nsp9", "nsp10",
    "nsp12", "nsp13", "nsp14", "nsp15", "nsp16", "non_coding2", "S", "non_coding3", "ORF3a", 
    "non_coding4", "E", "non_coding5", "M", "non_coding6", "ORF6", "non_coding7", "ORF7a", 
    "ORF7b", "non_coding8", "ORF8", "non_coding9", "N", "non_coding10", "ORF10", "non_coding11"
]

codon_pos_name = ["c_0", "c_1", "c_2", "c_3"]

print(f"Number of ts_name classes: {len(ts_name)}")
print(f"Number of base_mut_name classes: {len(base_mut_name)}")
print(f"Number of base_pos_name classes: {len(base_pos_name)}")
print(f"Number of amino_mut_name classes: {len(amino_mut_name)}")
print(f"Number of amino_pos_name classes: {len(amino_pos_name)}")
print(f"Number of mutation_type classes: {len(mutation_type)}")
print(f"Number of protein classes: {len(protein_name)}")
print(f"Number of codon_pos_name classes: {len(codon_pos_name)}")


def extract_keys_in_range(data, start_key, end_key):
    filtered_values = []
    for key, value in data.items():
        if start_key <= key <= end_key:
            for v in value:
                filtered_values.append(v)
    return filtered_values

def create_time_aware_split_modified(data, test_start=31, ylen=1, val_ratio=0.2):
    train_end = test_start - 1

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = {}, {}
    
    # タイムステップ 1-30 のデータを収集
    timestep_1_30_data = []
    
    for d in data:
        seq_len = len(d)
        
        if seq_len >= test_start:  # テストデータまで含む長いシーケンス
            # 1-30のデータを収集
            seq_1_30 = extract_keys_in_range(d, 1, train_end)
            if len(seq_1_30) > ylen:  # 最低限の長さがある場合のみ
                timestep_1_30_data.append(seq_1_30)
            
            # テストデータ（31以降）
            for i in range(test_start, seq_len + 1 - ylen + 1):
                if i not in test_x:
                    test_x[i] = []
                    test_y[i] = []
                test_x[i].append(extract_keys_in_range(d, i - (test_start - train_end), i - 1))
                test_y[i].append(extract_keys_in_range(d, i, i + ylen - 1))
        
        elif seq_len > ylen:  # 短いシーケンス（テストデータなし）
            seq_data = extract_keys_in_range(d, 1, seq_len - ylen)
            seq_label = extract_keys_in_range(d, seq_len - ylen + 1, seq_len)
            if len(seq_data) > 0:
                timestep_1_30_data.append((seq_data, seq_label))
    
    # タイムステップ 1-30 のデータを訓練・検証に分割
    import random
    #random.shuffle(timestep_1_30_data)
    
    val_size = int(len(timestep_1_30_data) * val_ratio)
    train_size = len(timestep_1_30_data) - val_size
    
    print(f"  - 訓練・検証データソース: タイムステップ 1-{train_end}")
    print(f"  - テストデータソース: タイムステップ {test_start}以降")
    print(f"  - 訓練データ: {train_size}サンプル ({(1-val_ratio)*100:.0f}%)")
    print(f"  - 検証データ: {val_size}サンプル ({val_ratio*100:.0f}%)")
    
    # 同じデータソースから訓練・検証を分割
    for i, item in enumerate(timestep_1_30_data):
        if isinstance(item, tuple):  # 短いシーケンス
            seq_data, seq_label = item
        else:  # 長いシーケンス
            seq_data = item[:-ylen] if len(item) > ylen else item
            seq_label = item[-ylen:] if len(item) > ylen else [item[-1]] if item else []
        
        if i < train_size:
            train_x.append(seq_data)
            train_y.append(seq_label)
        else:
            val_x.append(seq_data)
            val_y.append(seq_label)
    
    print(f"  - 実際の訓練データ: {len(train_x)}サンプル")
    print(f"  - 実際の検証データ: {len(val_x)}サンプル")
    print(f"  - テストタイムステップ: {sorted(test_x.keys()) if test_x else 'なし'}")
    total = 0
    for key in sorted(test_x.keys()):
        total += len(test_x[key])
        print(f"    - タイムステップ {key}: {len(test_x[key])}サンプル")
    print(f"  - テストデータ総数: {total}サンプル")
    
    return train_x, train_y, val_x, val_y, test_x, test_y

def extract_feature_sequences(paths,feature_idx,feature_name=None):
    features = []
    for path in paths:
        if feature_name is not None:
            feature = [0]*len(feature_name)
        else:
            feature = []
        for mutation in path:
            if len(mutation) > feature_idx: 
                if feature_name is not None:
                    for fn in feature_name:
                        if fn == mutation[feature_idx]:
                            feature[feature_name.index(fn)] = 1
                else:
                    feature.append(mutation[feature_idx])
            else:
                return print(f"Mutation {mutation} does not have enough features.")
        features.append(feature)
    return features

def add_x_by_y(x, y):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    new_x, new_y = [], []
    for i in range(len(y)):
        for fea in y[i]:
            new_x.append(x[i])
            new_y.append([fea])
    return new_x, new_y

class MutationSequenceDataset(Dataset):
    def __init__(self, sequences, labels, feature_vocabs, max_length=None):
        self.sequences = sequences
        self.labels = labels
        self.feature_vocabs = feature_vocabs
        self.max_length = max_length if max_length else max(len(seq) for seq in sequences)
        
        # count特徴量の統計を計算（正規化用）
        all_count_values = []
        for seq in sequences:
            for mutation in seq:
                if isinstance(mutation, list) and len(mutation) >= 9:
                    count_val = float(mutation[8]) if mutation[8] is not None else 0.0
                    all_count_values.append(count_val)
        
        self.count_mean = np.mean(all_count_values) if all_count_values else 0.0
        self.count_std = np.std(all_count_values) if all_count_values else 1.0
        print(f"Count特徴量: mean={self.count_mean:.4f}, std={self.count_std:.4f}")
        
        # ラベルをエンコード
        self.label_encoder = LabelEncoder()
        flat_labels = [label[0] if isinstance(label, list) and len(label) > 0 else label for label in labels]
        self.encoded_labels = self.label_encoder.fit_transform(flat_labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        # データの前処理で語彙範囲をチェック
        self._validate_data()
        
    def _validate_data(self):
        """データが語彙の範囲内にあるかチェック"""
        print("データの語彙範囲チェックを実行中...")
        for seq_idx, sequence in enumerate(self.sequences[:100]):  # 最初の100シーケンスをチェック
            categorical_features, _ = self.encode_sequence(sequence)
            
            for feature_idx, feature_seq in enumerate(categorical_features):
                max_val = max(feature_seq) if feature_seq else 0
                vocab_size = len(self.feature_vocabs[feature_idx])
                
                if max_val >= vocab_size:
                    print(f"❌ シーケンス{seq_idx}, 特徴量{feature_idx}: max_val={max_val} >= vocab_size={vocab_size}")
                    # 原因となる実際のデータを表示
                    for mut_idx, mutation in enumerate(sequence):
                        if isinstance(mutation, list) and len(mutation) > feature_idx:
                            raw_value = str(mutation[feature_idx])
                            if raw_value not in self.feature_vocabs[feature_idx]:
                                print(f"    変異{mut_idx}: 未知の値 '{raw_value}'")
                    break
            else:
                continue
            break
        print("データ検証完了")
    
    def encode_sequence(self, sequence):
        num_categorical_features = len(self.feature_vocabs)
        categorical_features = [[] for _ in range(num_categorical_features)]
        count_features = []
        
        for mutation in sequence:
            if isinstance(mutation, list) and len(mutation) >= num_categorical_features + 1:
                # カテゴリカル特徴量をエンコード
                for i in range(num_categorical_features):
                    feature_value = str(mutation[i])
                    # 安全なエンコーディング
                    if feature_value in self.feature_vocabs[i]:
                        encoded_value = self.feature_vocabs[i][feature_value]
                    else:
                        # 未知の値は<UNK>にマップ
                        encoded_value = self.feature_vocabs[i]['<UNK>']
                        # デバッグ用（初回のみ）
                        if hasattr(self, '_debug_count'):
                            self._debug_count += 1
                        else:
                            self._debug_count = 1
                            print(f"未知の値が検出されました: 特徴量{i}, 値='{feature_value}'")
                    
                    categorical_features[i].append(encoded_value)
                
                # count特徴量を正規化
                count_value = float(mutation[num_categorical_features]) if len(mutation) > num_categorical_features else 0.0
                normalized_count = (count_value - self.count_mean) / (self.count_std + 1e-8)
                count_features.append(normalized_count)
            else:
                # 不完全なデータの場合
                for i in range(num_categorical_features):
                    categorical_features[i].append(self.feature_vocabs[i]['<UNK>'])
                count_features.append(0.0)
        
        # パディング処理
        padded_categorical = []
        for i in range(num_categorical_features):
            feature_seq = categorical_features[i]
            if len(feature_seq) > self.max_length:
                feature_seq = feature_seq[:self.max_length]
            else:
                pad_value = self.feature_vocabs[i]['<PAD>']
                feature_seq = feature_seq + [pad_value] * (self.max_length - len(feature_seq))
            padded_categorical.append(feature_seq)
        
        # count特徴量のパディング
        if len(count_features) > self.max_length:
            count_features = count_features[:self.max_length]
        else:
            count_features = count_features + [0.0] * (self.max_length - len(count_features))
        
        return padded_categorical, count_features
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.encoded_labels[idx]
        
        categorical_features, count_features = self.encode_sequence(sequence)
        
        return {
            'categorical': torch.tensor(categorical_features, dtype=torch.long),
            'count': torch.tensor(count_features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }
    

def build_feature_vocabularies_from_definitions():
    """
    カテゴリカル特徴量のみの語彙を構築（countは除く）
    """
    # カテゴリカル特徴量のみ（countは除く）
    feature_lists = [
        ts_name,           # タイムステップ
        base_mut_name,     # 塩基変異
        base_pos_name,     # 塩基位置
        amino_mut_name,    # アミノ酸変異
        amino_pos_name,    # アミノ酸位置
        mutation_type,     # 変異タイプ
        protein_name,      # プロテイン名
        codon_pos_name,    # コドン位置
    ]
    
    feature_names = ['ts', 'base_mut', 'base_pos', 'amino_mut', 'amino_pos', 'mut_type', 'protein', 'codon_pos']
    
    feature_vocabs = []
    for i, (name, feature_list) in enumerate(zip(feature_names, feature_lists)):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        # 文字列に変換してソート
        feature_strings = [str(x) for x in feature_list]
        for j, token in enumerate(sorted(set(feature_strings))):
            vocab[token] = j + 2
            
        feature_vocabs.append(vocab)
        print(f"{name}: {len(vocab)} tokens")
    
    print("注意: count特徴量は数値として直接使用されます（語彙辞書なし）")
    return feature_vocabs