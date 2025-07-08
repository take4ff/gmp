# 機械学習ライブラリのインポート
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

from module.make_dataset import extract_feature_sequences, add_x_by_y
from module.mutation_transformer2 import evaluate

def evaluate_test_data_timestep(model, test_x, test_y, feature_idx, feature_vocabs, val_dataset, device, criterion):
    """
    タイムステップごとにテストデータでの予測精度を評価
    
    Args:
        feature_idx: プロテイン特徴量のインデックス（通常は6）
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
        
        # プロテイン名を抽出（ここで生成）
        test_y_protein = extract_feature_sequences(test_labels, feature_idx)
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
        
        # シーケンス単位での予測と真のラベルを準備
        predictions_per_sequence = []
        true_labels_per_sequence = []
        
        # filtered_protein_labelsをシーケンス単位に整理
        for seq_idx, orig_labels in enumerate(filtered_protein_labels):
            if isinstance(orig_labels, list):
                true_labels_set = set(orig_labels)
            else:
                true_labels_set = {orig_labels}
            
            true_labels_per_sequence.append(true_labels_set)
            
            # 対応する予測を取得（シーケンスごとに1つの予測）
            if seq_idx < len(test_preds):
                pred_idx = test_preds[seq_idx]
                if pred_idx < len(train_label_encoder.classes_):
                    pred_class_name = train_label_encoder.classes_[pred_idx]
                    predictions_per_sequence.append([pred_class_name])
                else:
                    predictions_per_sequence.append(['<UNK>'])
            else:
                predictions_per_sequence.append(['<UNK>'])
        
        # シーケンス単位の精度計算（2つの指標のみ）
        strict_sequence_accuracy = calculate_strict_sequence_accuracy(
            predictions_per_sequence, true_labels_per_sequence
        )
        flexible_sequence_accuracy = calculate_flexible_sequence_accuracy(
            predictions_per_sequence, true_labels_per_sequence
        )
        
        # 結果を保存（2つの精度指標のみ）
        timestep_results[timestep] = {
            'samples': len(test_sequences),
            'expanded_samples': len(test_x_expanded),
            'filtered_samples': len(filtered_x),
            'loss': test_loss,
            'strict_sequence_accuracy': strict_sequence_accuracy,  # シーケンス単位の厳密精度
            'flexible_sequence_accuracy': flexible_sequence_accuracy,  # シーケンス単位の柔軟精度
            'predictions': test_preds,
            'targets': test_targets,
            'original_labels': filtered_protein_labels,
            'predictions_per_sequence': predictions_per_sequence,
            'true_labels_per_sequence': [list(labels) for labels in true_labels_per_sequence]
        }
        
        print(f"  サンプル数: {len(test_sequences)} (展開後: {len(test_x_expanded)}, フィルタ後: {len(filtered_x)})")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  厳密シーケンス精度: {strict_sequence_accuracy:.4f}")
        print(f"  柔軟シーケンス精度: {flexible_sequence_accuracy:.4f}")
    
    return timestep_results

def calculate_strict_sequence_accuracy(predictions_per_sequence, true_labels_per_sequence):
    """
    厳密正解率：単一ラベルの場合のみ正解、複数ラベルは必ず不正解
    """
    correct_sequences = 0
    total_sequences = len(true_labels_per_sequence)
    
    for pred_list, true_set in zip(predictions_per_sequence, true_labels_per_sequence):
        pred = pred_list[0]  # 単一予測
        
        # 真のラベルが1つの場合のみ正解の可能性あり
        if len(true_set) == 1 and pred in true_set:
            correct_sequences += 1
        # 複数ラベルの場合は必ず不正解
    
    return correct_sequences / total_sequences if total_sequences > 0 else 0.0

def calculate_flexible_sequence_accuracy(predictions_per_sequence, true_labels_per_sequence):
    """
    柔軟正解率：予測が真のラベルのいずれかに含まれていれば正解
    """
    correct_sequences = 0
    total_sequences = len(true_labels_per_sequence)
    
    for pred_list, true_set in zip(predictions_per_sequence, true_labels_per_sequence):
        pred = pred_list[0]  # 単一予測
        
        # 予測が真のラベルセットに含まれていれば正解
        if pred in true_set:
            correct_sequences += 1
    
    return correct_sequences / total_sequences if total_sequences > 0 else 0.0

def predict_for_single_sequence(model, sequence, feature_vocabs, val_dataset, device):
    """
    単一シーケンスに対する単一予測
    """
    temp_dataset = TestMutationDataset(
        [sequence], [['dummy']], feature_vocabs, 
        val_dataset.max_length, val_dataset.label_encoder
    )
    temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False)
    
    model.eval()
    
    with torch.no_grad():
        for batch in temp_loader:
            categorical = batch['categorical'].to(device)
            count = batch['count'].to(device)
            
            outputs = model(categorical, count)
            probabilities = torch.softmax(outputs, dim=1)
            
            # 最高確率のクラス1つのみを予測
            best_idx = torch.argmax(probabilities[0])
            class_name = val_dataset.label_encoder.classes_[best_idx.item()]
            
            return [class_name]

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

def plot_timestep_results(timestep_results, save_path=None):
    """タイムステップごとの結果を可視化（2つの精度指標のみ）"""
    if not timestep_results:
        print("No results to display")
        return
    
    timesteps = sorted(timestep_results.keys())
    strict_seq_accs = [timestep_results[ts]['strict_sequence_accuracy'] for ts in timesteps]
    flexible_seq_accs = [timestep_results[ts]['flexible_sequence_accuracy'] for ts in timesteps]
    sample_counts = [timestep_results[ts]['filtered_samples'] for ts in timesteps]
    
    plt.figure(figsize=(15, 8))
    
    # シーケンス精度比較
    plt.subplot(2, 3, 1)
    plt.plot(timesteps, strict_seq_accs, 's-', label='Strict Sequence Accuracy', linewidth=2, markersize=6)
    plt.plot(timesteps, flexible_seq_accs, '^-', label='Flexible Sequence Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Timestep')
    plt.ylabel('Accuracy')
    plt.title('Sequence Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サンプル数
    plt.subplot(2, 3, 2)
    plt.bar(timesteps, sample_counts, alpha=0.7, color='skyblue')
    plt.xlabel('Timestep')
    plt.ylabel('Sample Count')
    plt.title('Sample Count by Timestep (After Filter)')
    plt.grid(True, alpha=0.3)
    
    # 精度向上分析
    plt.subplot(2, 3, 3)
    seq_improvement = [flexible_seq_accs[i] - strict_seq_accs[i] for i in range(len(timesteps))]
    colors = ['green' if improvement > 0 else 'red' for improvement in seq_improvement]
    plt.bar(timesteps, seq_improvement, color=colors, alpha=0.7)
    plt.xlabel('Timestep')
    plt.ylabel('Accuracy Improvement')
    plt.title('Sequence Accuracy Improvement (Flexible - Strict)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # 両精度の時系列推移
    plt.subplot(2, 3, 4)
    plt.plot(timesteps, strict_seq_accs, 's-', label='Strict Sequence', linewidth=3, markersize=8)
    plt.plot(timesteps, flexible_seq_accs, '^-', label='Flexible Sequence', linewidth=3, markersize=8)
    plt.xlabel('Timestep')
    plt.ylabel('Accuracy')
    plt.title('Sequence Accuracy Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 精度分布
    plt.subplot(2, 3, 5)
    plt.hist(strict_seq_accs, alpha=0.7, label='Strict Sequence', bins=10)
    plt.hist(flexible_seq_accs, alpha=0.7, label='Flexible Sequence', bins=10)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Accuracy Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 統計サマリー
    plt.subplot(2, 3, 6)
    metrics = ['Mean', 'Std', 'Min', 'Max']
    strict_stats = [np.mean(strict_seq_accs), np.std(strict_seq_accs), 
                   np.min(strict_seq_accs), np.max(strict_seq_accs)]
    flexible_stats = [np.mean(flexible_seq_accs), np.std(flexible_seq_accs),
                     np.min(flexible_seq_accs), np.max(flexible_seq_accs)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, strict_stats, width, label='Strict Sequence', alpha=0.7)
    plt.bar(x + width/2, flexible_stats, width, label='Flexible Sequence', alpha=0.7)
    plt.xlabel('Statistics')
    plt.ylabel('Value')
    plt.title('Accuracy Statistics')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存処理
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存: {save_path}")
    
    plt.show()

def print_detailed_timestep_analysis(timestep_results, train_dataset):
    """タイムステップごとの詳細分析を表示（2つの精度指標のみ）"""
    print("\n=== Detailed Analysis by Timestep ===")
    
    class_names = train_dataset.label_encoder.classes_
    
    for timestep in sorted(timestep_results.keys()):
        result = timestep_results[timestep]
        print(f"\n--- Timestep {timestep} ---")
        print(f"Samples: {result['samples']} (Expanded: {result['expanded_samples']}, Filtered: {result['filtered_samples']})")
        print(f"Loss: {result['loss']:.4f}")
        print(f"Strict Sequence Accuracy: {result['strict_sequence_accuracy']:.4f}")
        print(f"Flexible Sequence Accuracy: {result['flexible_sequence_accuracy']:.4f}")
        print(f"Sequence Accuracy Improvement: {result['flexible_sequence_accuracy'] - result['strict_sequence_accuracy']:.4f}")
        
        unique_targets, target_counts = np.unique(result['targets'], return_counts=True)
        print(f"Target Class Distribution:")
        for label, count in zip(unique_targets, target_counts):
            if label < len(class_names):
                class_name = class_names[label]
                print(f"  {class_name}: {count} samples")

def print_test_summary(timestep_results):
    """テスト結果のサマリーを表示（2つの精度指標のみ）"""
    if not timestep_results:
        print("No test results available")
        return
    
    print("\n=== Test Results Summary ===")
    
    # 全体統計
    total_samples = sum(result['samples'] for result in timestep_results.values())
    total_expanded = sum(result['expanded_samples'] for result in timestep_results.values())
    total_filtered = sum(result['filtered_samples'] for result in timestep_results.values())
    
    # 重み付き平均精度（サンプル数で重み付け）
    weighted_strict_seq_acc = sum(
        result['strict_sequence_accuracy'] * result['filtered_samples'] 
        for result in timestep_results.values()
    ) / total_filtered if total_filtered > 0 else 0
    
    weighted_flexible_seq_acc = sum(
        result['flexible_sequence_accuracy'] * result['filtered_samples'] 
        for result in timestep_results.values()
    ) / total_filtered if total_filtered > 0 else 0
    
    print(f"Total Samples: {total_samples} (Expanded: {total_expanded}, Filtered: {total_filtered})")
    print(f"Test Timesteps: {len(timestep_results)}")
    print(f"Timestep Range: {min(timestep_results.keys())} - {max(timestep_results.keys())}")
    print(f"Weighted Average Strict Sequence Accuracy: {weighted_strict_seq_acc:.4f}")
    print(f"Weighted Average Flexible Sequence Accuracy: {weighted_flexible_seq_acc:.4f}")
    print(f"Sequence Accuracy Improvement: {weighted_flexible_seq_acc - weighted_strict_seq_acc:.4f}")
    
    # 最高・最低精度のタイムステップ
    best_strict_seq_ts = max(timestep_results.keys(), 
                            key=lambda ts: timestep_results[ts]['strict_sequence_accuracy'])
    worst_strict_seq_ts = min(timestep_results.keys(), 
                             key=lambda ts: timestep_results[ts]['strict_sequence_accuracy'])
    
    best_flexible_seq_ts = max(timestep_results.keys(), 
                              key=lambda ts: timestep_results[ts]['flexible_sequence_accuracy'])
    worst_flexible_seq_ts = min(timestep_results.keys(), 
                               key=lambda ts: timestep_results[ts]['flexible_sequence_accuracy'])
    
    print(f"\nBest Strict Sequence Accuracy: Timestep {best_strict_seq_ts} ({timestep_results[best_strict_seq_ts]['strict_sequence_accuracy']:.4f})")
    print(f"Worst Strict Sequence Accuracy: Timestep {worst_strict_seq_ts} ({timestep_results[worst_strict_seq_ts]['strict_sequence_accuracy']:.4f})")
    print(f"Best Flexible Sequence Accuracy: Timestep {best_flexible_seq_ts} ({timestep_results[best_flexible_seq_ts]['flexible_sequence_accuracy']:.4f})")
    print(f"Worst Flexible Sequence Accuracy: Timestep {worst_flexible_seq_ts} ({timestep_results[worst_flexible_seq_ts]['flexible_sequence_accuracy']:.4f})")
    
    # 精度向上が最も大きかったタイムステップ
    max_improvement_ts = max(timestep_results.keys(), 
                            key=lambda ts: timestep_results[ts]['flexible_sequence_accuracy'] - timestep_results[ts]['strict_sequence_accuracy'])
    max_improvement = timestep_results[max_improvement_ts]['flexible_sequence_accuracy'] - timestep_results[max_improvement_ts]['strict_sequence_accuracy']
    
    print(f"Maximum Sequence Accuracy Improvement: Timestep {max_improvement_ts} (+{max_improvement:.4f})")