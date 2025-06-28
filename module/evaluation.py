# 機械学習ライブラリのインポート
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

from module.make_dataset import extract_feature_sequences, add_x_by_y
from module.mutation_transformer import evaluate

def evaluate_test_data_timestep(model, test_x, test_y, feature_vocabs, val_dataset, device, criterion):
    """
    タイムステップごとにテストデータでの予測精度を評価
    複数ラベルがある場合、いずれかに合致すれば正解とする
    """
    print("=== タイムステップごとのテスト評価 ===")
    
    timestep_results = {}
    
    # 訓練データセットのラベルエンコーダーを取得
    train_label_encoder = val_dataset.label_encoder
    train_classes = set(train_label_encoder.classes_)
    
    for timestep in sorted(test_x.keys()):
        print(f"\nタイムステップ {timestep} の評価中...")
        
        # テストデータの準備
        test_sequences = test_x[timestep]
        test_labels = test_y[timestep]
        
        if len(test_sequences) == 0:
            print(f"  タイムステップ {timestep}: データなし")
            continue
        
        # プロテイン名を抽出（特徴量インデックス6）
        test_y_protein = extract_feature_sequences(test_labels, 6)
        test_x_expanded, test_y_expanded = add_x_by_y(test_sequences, test_y_protein)
        
        # 未知のクラスをフィルタリング
        filtered_x = []
        filtered_y = []
        filtered_protein_labels = []
        
        for i, (x, y, orig_label) in enumerate(zip(test_x_expanded, test_y_expanded, test_y_protein)):
            label = y[0] if isinstance(y, list) and len(y) > 0 else y
            if str(label) in train_classes:
                filtered_x.append(x)
                filtered_y.append(y)
                # original_labelsの対応を維持
                if i < len(test_y_protein):
                    filtered_protein_labels.append(orig_label)
        
        if len(filtered_x) == 0:
            print(f"  タイムステップ {timestep}: 既知のクラスがありません")
            continue
        
        print(f"  フィルタリング: {len(test_x_expanded)} -> {len(filtered_x)} サンプル")
        
        # カスタムデータセットクラスを作成（訓練時のエンコーダーを使用）
        test_dataset = TestMutationDataset(
            filtered_x, 
            filtered_y, 
            feature_vocabs, 
            val_dataset.max_length,
            train_label_encoder  # 訓練時のエンコーダーを使用
        )
        
        # テストデータローダー
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 評価実行
        test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
        
        # 複数ラベル対応の精度計算
        flexible_accuracy = calculate_flexible_accuracy(
            test_preds, test_targets, filtered_protein_labels, train_label_encoder
        )
        
        # 結果を保存
        timestep_results[timestep] = {
            'samples': len(test_sequences),
            'expanded_samples': len(test_x_expanded),
            'filtered_samples': len(filtered_x),
            'loss': test_loss,
            'strict_accuracy': test_acc,
            'flexible_accuracy': flexible_accuracy,
            'predictions': test_preds,
            'targets': test_targets,
            'original_labels': filtered_protein_labels
        }
        
        print(f"  サンプル数: {len(test_sequences)} (展開後: {len(test_x_expanded)}, フィルタ後: {len(filtered_x)})")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  厳密精度: {test_acc:.4f}")
        print(f"  柔軟精度: {flexible_accuracy:.4f}")
    
    return timestep_results

class TestMutationDataset(Dataset):
    """テストデータ用のデータセット（既存のエンコーダーを使用）"""
    def __init__(self, sequences, labels, feature_vocabs, max_length, label_encoder):
        self.sequences = sequences
        self.labels = labels
        self.feature_vocabs = feature_vocabs
        self.max_length = max_length
        self.label_encoder = label_encoder
        
        # 統計情報は訓練データセットから取得（簡略化のため0と1を使用）
        self.count_mean = 0.0
        self.count_std = 1.0
        
        # ラベルをエンコード（既存のエンコーダーを使用）
        flat_labels = [label[0] if isinstance(label, list) and len(label) > 0 else label for label in labels]
        self.encoded_labels = []
        
        for label in flat_labels:
            try:
                encoded = label_encoder.transform([str(label)])[0]
                self.encoded_labels.append(encoded)
            except ValueError:
                # 未知のラベルの場合はスキップ（事前にフィルタリング済み）
                print(f"Warning: Unknown label {label}")
                self.encoded_labels.append(0)  # デフォルト値
        
        self.num_classes = len(label_encoder.classes_)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.encoded_labels[idx]
        
        # シーケンスをエンコード（既存のメソッドを使用）
        categorical_features, count_features = self.encode_sequence(sequence)
        
        return {
            'categorical': torch.tensor(categorical_features, dtype=torch.long),
            'count': torch.tensor(count_features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def encode_sequence(self, sequence):
        """既存のエンコード方法を使用"""
        num_categorical_features = len(self.feature_vocabs)
        categorical_features = [[] for _ in range(num_categorical_features)]
        count_features = []
        
        for mutation in sequence:
            if isinstance(mutation, list) and len(mutation) >= num_categorical_features + 1:
                for i in range(num_categorical_features):
                    feature_value = str(mutation[i])
                    if feature_value in self.feature_vocabs[i]:
                        encoded_value = self.feature_vocabs[i][feature_value]
                    else:
                        encoded_value = self.feature_vocabs[i]['<UNK>']
                    categorical_features[i].append(encoded_value)
                
                count_value = float(mutation[num_categorical_features]) if len(mutation) > num_categorical_features else 0.0
                normalized_count = (count_value - self.count_mean) / (self.count_std + 1e-8)
                count_features.append(normalized_count)
            else:
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
        
        if len(count_features) > self.max_length:
            count_features = count_features[:self.max_length]
        else:
            count_features = count_features + [0.0] * (self.max_length - len(count_features))
        
        return padded_categorical, count_features

def calculate_flexible_accuracy(predictions, targets, original_labels, label_encoder):
    """
    複数ラベルに対応した精度計算
    予測がいずれかのラベルに合致すれば正解とする
    """
    if len(original_labels) == 0:
        return 0.0
    
    correct = 0
    total = 0
    
    # original_labelsをグループ化
    sequence_groups = {}
    current_idx = 0
    
    for seq_idx, labels_for_seq in enumerate(original_labels):
        if isinstance(labels_for_seq, list):
            num_labels = len(labels_for_seq)
        else:
            num_labels = 1
            labels_for_seq = [labels_for_seq]
        
        sequence_groups[seq_idx] = {
            'indices': list(range(current_idx, current_idx + num_labels)),
            'labels': labels_for_seq
        }
        current_idx += num_labels
    
    # グループごとに精度を計算
    for seq_idx, group_info in sequence_groups.items():
        indices = group_info['indices']
        true_labels = group_info['labels']
        
        # インデックスの範囲チェック
        valid_indices = [i for i in indices if i < len(predictions)]
        if not valid_indices:
            continue
        
        seq_predictions = [predictions[i] for i in valid_indices]
        
        for pred in seq_predictions:
            # 予測インデックスの範囲チェック
            if pred >= len(label_encoder.classes_):
                print(f"Warning: Prediction index {pred} out of range for {len(label_encoder.classes_)} classes")
                continue
                
            pred_class_name = label_encoder.classes_[pred]
            
            # 予測が正解ラベルのいずれかに合致するかチェック
            if pred_class_name in true_labels:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0

# 日本語フォントの設定
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 利用可能な日本語フォントを検索
def find_japanese_font():
    """利用可能な日本語フォントを検索"""
    font_candidates = [
        'DejaVu Sans',  # デフォルト（英語のみ）
        'Liberation Sans',
        'Arial Unicode MS',
        'Noto Sans CJK JP',
        'Takao Gothic',
        'IPAexGothic',
        'VL Gothic'
    ]
    
    for font_name in font_candidates:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path:
                return font_name
        except:
            continue
    
    return 'DejaVu Sans'  # フォールバック

# フォント設定
japanese_font = find_japanese_font()
plt.rcParams['font.family'] = japanese_font
plt.rcParams['font.size'] = 10

print(f"使用フォント: {japanese_font}")

def plot_timestep_results(timestep_results):
    """タイムステップごとの結果を可視化（英語ラベル使用）"""
    if not timestep_results:
        print("No results to display")
        return
    
    timesteps = sorted(timestep_results.keys())
    strict_accs = [timestep_results[ts]['strict_accuracy'] for ts in timesteps]
    flexible_accs = [timestep_results[ts]['flexible_accuracy'] for ts in timesteps]
    sample_counts = [timestep_results[ts]['filtered_samples'] for ts in timesteps]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(timesteps, strict_accs, 'o-', label='Strict Accuracy', linewidth=2, markersize=6)
    plt.plot(timesteps, flexible_accs, 's-', label='Flexible Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Timestep')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy by Timestep')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.bar(timesteps, sample_counts, alpha=0.7, color='skyblue')
    plt.xlabel('Timestep')
    plt.ylabel('Sample Count')
    plt.title('Sample Count by Timestep (After Filter)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    accuracy_improvement = [flexible_accs[i] - strict_accs[i] for i in range(len(timesteps))]
    colors = ['green' if improvement > 0 else 'red' for improvement in accuracy_improvement]
    plt.bar(timesteps, accuracy_improvement, color=colors, alpha=0.7)
    plt.xlabel('Timestep')
    plt.ylabel('Accuracy Improvement (Flexible - Strict)')
    plt.title('Accuracy Improvement by Multi-label')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_detailed_timestep_analysis(timestep_results, train_dataset):
    """タイムステップごとの詳細分析を表示"""
    print("\n=== Detailed Analysis by Timestep ===")
    
    class_names = train_dataset.label_encoder.classes_
    
    for timestep in sorted(timestep_results.keys()):
        result = timestep_results[timestep]
        print(f"\n--- Timestep {timestep} ---")
        print(f"Samples: {result['samples']} (Expanded: {result['expanded_samples']}, Filtered: {result['filtered_samples']})")
        print(f"Loss: {result['loss']:.4f}")
        print(f"Strict Accuracy: {result['strict_accuracy']:.4f}")
        print(f"Flexible Accuracy: {result['flexible_accuracy']:.4f}")
        print(f"Accuracy Improvement: {result['flexible_accuracy'] - result['strict_accuracy']:.4f}")
        
        unique_targets, target_counts = np.unique(result['targets'], return_counts=True)
        print(f"Target Class Distribution:")
        for label, count in zip(unique_targets, target_counts):
            if label < len(class_names):
                class_name = class_names[label]
                print(f"  {class_name}: {count} samples")

def print_test_summary(timestep_results):
    """テスト結果のサマリーを表示"""
    if not timestep_results:
        print("No test results available")
        return
    
    print("\n=== Test Results Summary ===")
    
    # 全体統計
    total_samples = sum(result['samples'] for result in timestep_results.values())
    total_expanded = sum(result['expanded_samples'] for result in timestep_results.values())
    total_filtered = sum(result['filtered_samples'] for result in timestep_results.values())
    
    # 重み付き平均精度（サンプル数で重み付け）
    weighted_strict_acc = sum(
        result['strict_accuracy'] * result['filtered_samples'] 
        for result in timestep_results.values()
    ) / total_filtered if total_filtered > 0 else 0
    
    weighted_flexible_acc = sum(
        result['flexible_accuracy'] * result['filtered_samples'] 
        for result in timestep_results.values()
    ) / total_filtered if total_filtered > 0 else 0
    
    print(f"Total Samples: {total_samples} (Expanded: {total_expanded}, Filtered: {total_filtered})")
    print(f"Test Timesteps: {len(timestep_results)}")
    print(f"Timestep Range: {min(timestep_results.keys())} - {max(timestep_results.keys())}")
    print(f"Weighted Average Strict Accuracy: {weighted_strict_acc:.4f}")
    print(f"Weighted Average Flexible Accuracy: {weighted_flexible_acc:.4f}")
    print(f"Average Accuracy Improvement: {weighted_flexible_acc - weighted_strict_acc:.4f}")
    
    # 最高・最低精度のタイムステップ
    best_strict_ts = max(timestep_results.keys(), 
                        key=lambda ts: timestep_results[ts]['strict_accuracy'])
    worst_strict_ts = min(timestep_results.keys(), 
                         key=lambda ts: timestep_results[ts]['strict_accuracy'])
    
    best_flexible_ts = max(timestep_results.keys(), 
                          key=lambda ts: timestep_results[ts]['flexible_accuracy'])
    worst_flexible_ts = min(timestep_results.keys(), 
                           key=lambda ts: timestep_results[ts]['flexible_accuracy'])
    
    print(f"\nBest Strict Accuracy: Timestep {best_strict_ts} ({timestep_results[best_strict_ts]['strict_accuracy']:.4f})")
    print(f"Worst Strict Accuracy: Timestep {worst_strict_ts} ({timestep_results[worst_strict_ts]['strict_accuracy']:.4f})")
    print(f"Best Flexible Accuracy: Timestep {best_flexible_ts} ({timestep_results[best_flexible_ts]['flexible_accuracy']:.4f})")
    print(f"Worst Flexible Accuracy: Timestep {worst_flexible_ts} ({timestep_results[worst_flexible_ts]['flexible_accuracy']:.4f})")
    
    # 精度向上が最も大きかったタイムステップ
    max_improvement_ts = max(timestep_results.keys(), 
                            key=lambda ts: timestep_results[ts]['flexible_accuracy'] - timestep_results[ts]['strict_accuracy'])
    max_improvement = timestep_results[max_improvement_ts]['flexible_accuracy'] - timestep_results[max_improvement_ts]['strict_accuracy']
    
    print(f"Maximum Accuracy Improvement: Timestep {max_improvement_ts} (+{max_improvement:.4f})")