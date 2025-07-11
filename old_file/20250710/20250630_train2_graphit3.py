# %%
# 必要なライブラリのインポート
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import importlib
import pickle
import hashlib
import traceback
from datetime import datetime
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import classification_report

# 高速化用インポート
import torch.cuda.amp as amp  # 混合精度
try:
    torch._dynamo.config.suppress_errors = True  # torch.compile エラー抑制
except:
    pass

# カスタムモジュールのインポート
import module.input_mutation_path as imp
import module.get_feature as gfea
import module.mutation_graphit as mt
import module.make_dataset as mds
import module.evaluation2 as ev
import module.save2 as save

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# モジュールの再読み込み（開発時のみ）
importlib.reload(imp)
importlib.reload(gfea)
importlib.reload(mt)
importlib.reload(mds)
importlib.reload(ev)
importlib.reload(save)

# %%
# 実験設定とハイパーパラメータ
# =============================================================================

# 保存ディレクトリの設定
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = "../model/20250630_train2/"
save_dir = os.path.join(folder_name, current_time)
os.makedirs(save_dir, exist_ok=True)

# モデルハイパーパラメータ
model_config = {
    'num_epochs': 50,
    'batch_size': 128,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'auto_adjust': True,  # パラメータ自動調整機能
    
    # 🚀 GraphiT高速化設定
    'optimization': {
        'enable_graph_cache': True,        # グラフキャッシュを有効化
        'max_graph_connections': 8,        # 最大接続数制限（デフォルト5→8）
        'similarity_threshold': 0.75,      # 類似性閾値を上げて接続数削減
        'use_compile': True,               # torch.compile有効化（PyTorch 2.0+）
        'mixed_precision': True,           # 混合精度演算
        'gradient_checkpointing': False,   # メモリ節約（速度低下）
        'fast_math': True                  # 高速数学演算
    },
    
    # 開発・デバッグ用高速モード
    'dev_mode': {
        'enabled': False,                  # Trueで高速モード
        'sample_ratio': 0.1,              # 全データの10%使用
        'quick_epochs': 10,               # 短縮エポック数
        'batch_size_multiplier': 2        # バッチサイズ増加
    }
}

# 特徴量マスク設定（使用する特徴量を指定）
feature_mask = [
    True,   # ts (タイムステップ)
    True,   # base_mut (塩基変異)
    True,   # base_pos (塩基位置)
    True,   # amino_mut (アミノ酸変異)
    True,   # amino_pos (アミノ酸位置)
    True,   # mut_type (変異タイプ)
    True,   # protein (プロテイン)
    True,   # codon_pos (コドン位置)
    True    # count (カウント)
]

# データ分割設定
data_config = {
    'test_start': 36,
    'ylen': 1,
    'val_ratio': 0.2,
    'feature_idx': 6,  # protein特徴量のインデックス
    'nmax': 100000000,
    'nmax_per_strain': 1000000
}

# 🚀 開発モード適用
if model_config['dev_mode']['enabled']:
    print("⚡ 開発用高速モードを適用中...")
    data_config['nmax_per_strain'] = int(data_config['nmax_per_strain'] * model_config['dev_mode']['sample_ratio'])
    model_config['num_epochs'] = model_config['dev_mode']['quick_epochs']
    model_config['batch_size'] = model_config['batch_size'] * model_config['dev_mode']['batch_size_multiplier']
    print(f"  - データ制限: {data_config['nmax_per_strain']:,} samples per strain")
    print(f"  - エポック数: {model_config['num_epochs']}")
    print(f"  - バッチサイズ: {model_config['batch_size']}")

# データセット設定
dataset_config = {
    'strains': ['B.1.1.7'],  # ['B.1.1.7','P.1','BA.2','BA.1.1','BA.1','B.1.617.2','B.1.351','B.1.1.529']
    'usher_dir': '../usher_output/',
    'bunpu_csv': "table_heatmap/250621/table_set/table_set.csv",
    'codon_csv': 'meta_data/codon_mutation4.csv',
    'cache_dir': '../cache'  # 特徴データキャッシュ用ディレクトリ
}


print(f"実験設定完了 - 保存先: {save_dir}")
print(f"対象変異株: {dataset_config['strains']}")
print(f"モデル設定: d_model={model_config['d_model']}, nhead={model_config['nhead']}, num_layers={model_config['num_layers']}")

# キャッシュディレクトリの初期化と既存キャッシュの確認
cache_dir = dataset_config['cache_dir']
os.makedirs(cache_dir, exist_ok=True)

if os.path.exists(cache_dir):
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith('feature_data_cache_') and f.endswith('.pkl')]
    if cache_files:
        print(f"\n既存のキャッシュファイル ({len(cache_files)}個):")
        total_cache_size = 0
        for cache_file in sorted(cache_files):
            cache_path = os.path.join(cache_dir, cache_file)
            cache_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB
            total_cache_size += cache_size
            mtime = os.path.getmtime(cache_path)
            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {cache_file}: {cache_size:.1f}MB (作成: {mtime_str})")
        print(f"総キャッシュサイズ: {total_cache_size:.1f}MB")
    else:
        print("\n既存のキャッシュファイルはありません")
else:
    print(f"\nキャッシュディレクトリを作成: {cache_dir}")

# %%
# データの読み込みと前処理
# =============================================================================

# 入力データの読み込み
names, lengths, base_HGVS_paths = imp.input(
    dataset_config['strains'], 
    dataset_config['usher_dir'], 
    nmax=data_config['nmax'], 
    nmax_per_strain=data_config['nmax_per_strain']
)
bunpu_df = pd.read_csv(dataset_config['bunpu_csv'])
codon_df = pd.read_csv(dataset_config['codon_csv'])

# キャッシュファイルの設定
import pickle
import hashlib

# キャッシュキーの生成（変異株、nmax、nmax_per_strainから一意のハッシュを作成）
cache_key_data = {
    'strains': sorted(dataset_config['strains']),  # ソートして順序を統一
    'nmax': data_config['nmax'],
    'nmax_per_strain': data_config['nmax_per_strain'],
    'bunpu_csv': dataset_config['bunpu_csv'],
    'codon_csv': dataset_config['codon_csv']
}
cache_key_str = str(cache_key_data)
cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:12]  # 短縮ハッシュ
cache_filename = f"feature_data_cache_{cache_hash}.pkl"
cache_dir = dataset_config['cache_dir']
os.makedirs(cache_dir, exist_ok=True)
cache_filepath = os.path.join(cache_dir, cache_filename)

print(f"キャッシュキー: {cache_key_data}")
print(f"キャッシュファイル: {cache_filepath}")

# タイムステップを含む特徴データの抽出（キャッシュ機能付き）
cache_valid = False
if os.path.exists(cache_filepath):
    # キャッシュファイルの更新日時をチェック
    cache_mtime = os.path.getmtime(cache_filepath)
    
    # 依存ファイルの更新日時をチェック
    source_files = [dataset_config['bunpu_csv'], dataset_config['codon_csv']]
    latest_source_mtime = 0
    
    for source_file in source_files:
        if os.path.exists(source_file):
            source_mtime = os.path.getmtime(source_file)
            latest_source_mtime = max(latest_source_mtime, source_mtime)
        else:
            print(f"警告: ソースファイルが見つかりません: {source_file}")
    
    # キャッシュがソースファイルより新しい場合のみ有効
    if cache_mtime > latest_source_mtime:
        cache_valid = True
        print("キャッシュされた特徴データを読み込み中...")
        start_time = time.time()
        with open(cache_filepath, 'rb') as f:
            data = pickle.load(f)
        load_time = time.time() - start_time
        cache_size = os.path.getsize(cache_filepath) / (1024 * 1024)  # MB
        print(f"キャッシュからの読み込み完了: {len(data)} sequences ({load_time:.2f}秒, {cache_size:.1f}MB)")
    else:
        print("ソースファイルがキャッシュより新しいため、キャッシュを無効化します")

if not cache_valid:
    print("特徴データを新規生成中...")
    start_time = time.time()
    data = gfea.Feature_path_incl_ts(base_HGVS_paths, codon_df, bunpu_df)
    extraction_time = time.time() - start_time
    print(f"特徴データ生成完了: {len(data)} sequences ({extraction_time:.2f}秒)")
    
    # キャッシュに保存
    print("特徴データをキャッシュに保存中...")
    save_start = time.time()
    with open(cache_filepath, 'wb') as f:
        pickle.dump(data, f)
    save_time = time.time() - save_start
    cache_size = os.path.getsize(cache_filepath) / (1024 * 1024)  # MB
    print(f"キャッシュ保存完了: {cache_filepath} ({save_time:.2f}秒, {cache_size:.1f}MB)")

print(f"Sample data structure: {data[0][1]}")

# データ分割の実行
train_x, train_y, val_x, val_y, test_x, test_y = mds.create_time_aware_split_modified(
    data, data_config['test_start'], data_config['ylen'], data_config['val_ratio']
)

# プロテイン特徴量の抽出
train_y_protein = mds.extract_feature_sequences(train_y, data_config['feature_idx'])
val_y_protein = mds.extract_feature_sequences(val_y, data_config['feature_idx'])

# データとラベルの結合
train_x2, train_y2 = mds.add_x_by_y(train_x, train_y_protein)
val_x2, val_y2 = mds.add_x_by_y(val_x, val_y_protein)

print(f"データ分割完了:")
print(f"  訓練データ: {len(train_x2)} サンプル")
print(f"  検証データ: {len(val_x2)} サンプル")
print(f"  テストデータ: {len(test_x)} タイムステップ")

# %%
# 語彙構築とデータセット作成
# =============================================================================

# 語彙を構築
print("定義済み特徴量名から語彙を構築中（制限なし）...")
feature_vocabs = mds.build_feature_vocabularies_from_definitions()

print(f"\n総特徴量数: {len(feature_vocabs)}")
print(f"総語彙サイズ: {sum(len(vocab) for vocab in feature_vocabs):,}")

# 各特徴量の語彙サイズを詳細表示（カテゴリカル特徴量のみ）
print("\n各特徴量の詳細:")
feature_names = ['ts', 'base_mut', 'base_pos', 'amino_mut', 'amino_pos', 'mut_type', 'protein', 'codon_pos']
for i, name in enumerate(feature_names):
    print(f"  {name}: {len(feature_vocabs[i]):,} tokens")

print(f"  count: 数値（語彙辞書なし）")

# データセットとデータローダーを作成
print("データセットを作成中...")

# 訓練データセット
train_dataset = mds.MutationSequenceDataset(train_x2, train_y2, feature_vocabs)
val_dataset = mds.MutationSequenceDataset(val_x2, val_y2, feature_vocabs, train_dataset.max_length)

# データローダー（高速化設定付き）
train_loader = DataLoader(
    train_dataset, 
    batch_size=model_config['batch_size'], 
    shuffle=True,
    num_workers=4,          # 並列データ読み込み
    pin_memory=True,        # GPU転送高速化
    persistent_workers=True # ワーカープロセス再利用
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=model_config['batch_size'], 
    shuffle=False,
    num_workers=2,          # 検証用は少なめ
    pin_memory=True,
    persistent_workers=True
)

print(f"訓練データセット: {len(train_dataset)} サンプル")
print(f"検証データセット: {len(val_dataset)} サンプル")
print(f"クラス数: {train_dataset.num_classes}")
print(f"最大シーケンス長: {train_dataset.max_length}")
print(f"クラス: {train_dataset.label_encoder.classes_}")

# %%
# デバッグ用の語彙・データ整合性チェック
def debug_vocab_mismatch(train_loader, feature_vocabs):
    print("=== 語彙とデータの不整合チェック ===")
    
    for batch_idx, batch in enumerate(train_loader):
        categorical_data = batch['categorical']
        print(f"バッチ {batch_idx}: 形状 {categorical_data.shape}")
        
        for feature_idx in range(categorical_data.shape[1]):
            feature_data = categorical_data[:, feature_idx, :]
            max_val = feature_data.max().item()
            min_val = feature_data.min().item()
            vocab_size = len(feature_vocabs[feature_idx])
            
            print(f"  特徴量{feature_idx}: min={min_val}, max={max_val}, vocab_size={vocab_size}")
            
            if max_val >= vocab_size:
                print(f"    ❌ エラー: max_val({max_val}) >= vocab_size({vocab_size})")
                # 実際のデータを確認
                problematic_values = feature_data[feature_data >= vocab_size]
                print(f"    問題のある値: {problematic_values[:10].tolist()}")
                return feature_idx, max_val, vocab_size
            elif max_val < 0:
                print(f"    ❌ エラー: 負の値が検出されました")
                return feature_idx, max_val, vocab_size
        
        if batch_idx >= 2:  # 最初の3バッチのみチェック
            break
    
    print("✅ 全ての特徴量で語彙サイズの範囲内です")
    return None, None, None

# デバッグを実行
debug_vocab_mismatch(train_loader, feature_vocabs)

# %%
# モデル初期化と訓練設定
# =============================================================================

# モデル、オプティマイザー、損失関数の初期化
print("モデルを初期化中...")

# モデルのインスタンス化
model = mt.MutationGraphiT(
    feature_vocabs=feature_vocabs,
    d_model=model_config['d_model'],
    nhead=model_config['nhead'],
    num_layers=model_config['num_layers'],
    num_classes=train_dataset.num_classes,
    max_seq_length=train_dataset.max_length,
    feature_mask=feature_mask,
    auto_adjust=model_config['auto_adjust'],
    similarity_threshold=model_config['optimization']['similarity_threshold']  # 🚀 設定から閾値を取得
).to(device)

# 🚀 高速化最適化の適用
print("高速化最適化を適用中...")

# torch.compile最適化（PyTorch 2.0+）
if model_config['optimization']['use_compile'] and hasattr(torch, 'compile'):
    try:
        print("  - torch.compile最適化を適用中...")
        model = torch.compile(model, mode='max-autotune')
        print("  ✅ torch.compile最適化完了")
    except Exception as e:
        print(f"  ⚠️ torch.compile最適化失敗: {e}")

# 混合精度用のスケーラー初期化
scaler = None
if model_config['optimization']['mixed_precision'] and device.type == 'cuda':
    print("  - 混合精度演算を有効化...")
    scaler = torch.amp.GradScaler('cuda')
    print("  ✅ 混合精度設定完了")

# 高速数学演算の有効化
if model_config['optimization']['fast_math'] and device.type == 'cuda':
    print("  - CUDA高速数学演算を有効化...")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("  ✅ 高速数学演算設定完了")

# 損失関数とオプティマイザー
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=model_config['learning_rate'], weight_decay=model_config['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

print(f"実際のd_model: {model.actual_d_model}")
print(f"実際のnhead: {model.actual_nhead}")
print(f"実際のnum_layers: {model.actual_num_layers}")
print(f"モデルのパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
print(f"クラス数: {train_dataset.num_classes}")
print(f"最大シーケンス長: {train_dataset.max_length}")

# %%
# 🚀 高速化された訓練関数
def train_epoch_mixed_precision(model, dataloader, criterion, optimizer, device, scaler):
    """混合精度を使用した高速化訓練エポック"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        categorical_data = batch['categorical'].to(device, non_blocking=True)
        count_data = batch['count'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 混合精度での順伝播
        with torch.amp.autocast('cuda'):
            outputs = model(categorical_data, count_data)
            loss = criterion(outputs, labels)
        
        # 混合精度での逆伝播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# モデルの訓練
best_val_acc = 0
best_model_state = None
train_losses = []
val_losses = []
train_accs = []
val_accs = []
epoch_times = []  # エポック時間を記録

print("訓練を開始します...")
training_start_time = time.time()  # 全体の開始時間

try:
    for epoch in range(model_config['num_epochs']):
        epoch_start_time = time.time()  # エポック開始時間
        print(f"\nEpoch {epoch+1}/{model_config['num_epochs']}")
        
        # 🚀 高速化された訓練
        if scaler is not None:
            # 混合精度訓練
            train_loss, train_acc = train_epoch_mixed_precision(
                model, train_loader, criterion, optimizer, device, scaler
            )
        else:
            # 標準訓練
            train_loss, train_acc = mt.train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 検証
        val_loss, val_acc, val_preds, val_targets = mt.evaluate(model, val_loader, criterion, device)
        
        # スケジューラを更新
        scheduler.step(val_loss)
        
        # エポック終了時間計算
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # 結果を記録
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 時間情報を含む出力
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Epoch Time: {epoch_duration:.2f}s ({epoch_duration/60:.1f}min)")
        
        # 累積時間と推定残り時間
        total_elapsed = sum(epoch_times)
        avg_epoch_time = total_elapsed / len(epoch_times)
        remaining_epochs = model_config['num_epochs'] - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        print(f"Elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min), "
              f"ETA: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f}min)")
        
        # 最良モデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"新しい最良モデル (Val Acc: {val_acc:.4f})")

    # 訓練完了時の統計
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    print(f"\n=== 訓練完了! ===")
    print(f"最良検証精度: {best_val_acc:.4f}")
    print(f"総訓練時間: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
    print(f"平均エポック時間: {np.mean(epoch_times):.2f}s")
    print(f"最速エポック: {min(epoch_times):.2f}s")
    print(f"最遅エポック: {max(epoch_times):.2f}s")

    # 最良モデルをロード
    if best_model_state:
        model.load_state_dict(best_model_state)
        
except Exception as e:
    print(f"訓練中にエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()

# %%
# 訓練結果の分析と可視化
# =============================================================================

# 訓練結果の可視化
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
# 検証データでの最終評価
val_loss, val_acc, val_preds, val_targets = mt.evaluate(model, val_loader, criterion, device)
print(f"最終検証精度: {val_acc:.4f}")

# クラス名とラベルの対応を確認
class_names = train_dataset.label_encoder.classes_
print(f"全クラス数: {len(class_names)}")
print(f"検証データに含まれるユニークなクラス数: {len(set(val_targets))}")
print(f"予測に含まれるユニークなクラス数: {len(set(val_preds))}")

# 実際に使用されているクラスのみを取得
unique_labels = sorted(set(val_targets) | set(val_preds))
actual_class_names = [class_names[i] for i in unique_labels]

print(f"実際に使用されているクラス: {actual_class_names}")

# 分類レポート（実際に使用されているクラスのみ）
print("\n分類レポート:")
print(classification_report(
    val_targets, 
    val_preds, 
    labels=unique_labels,
    target_names=actual_class_names, 
    zero_division=0
))

# 追加の統計情報
print(f"\n詳細統計:")
print(f"検証データのクラス分布:")
unique_targets, target_counts = np.unique(val_targets, return_counts=True)
for label, count in zip(unique_targets, target_counts):
    class_name = class_names[label]
    print(f"  {class_name}: {count}サンプル")

plt.tight_layout()
plt.show()

# %%
# クラス分布の詳細分析
# =============================================================================

print("=== クラス分布の詳細分析 ===")

# 訓練データのクラス分布
train_labels = [train_dataset.encoded_labels[i] for i in range(len(train_dataset))]
train_unique, train_counts = np.unique(train_labels, return_counts=True)

print(f"\n訓練データのクラス分布:")
for label, count in zip(train_unique, train_counts):
    class_name = class_names[label]
    print(f"  {class_name}: {count}サンプル")

# 検証データのクラス分布
val_labels = [val_dataset.encoded_labels[i] for i in range(len(val_dataset))]
val_unique, val_counts = np.unique(val_labels, return_counts=True)

print(f"\n検証データのクラス分布:")
for label, count in zip(val_unique, val_counts):
    class_name = class_names[label]
    print(f"  {class_name}: {count}サンプル")

# 訓練データにあって検証データにないクラス
train_only = set(train_unique) - set(val_unique)
if train_only:
    print(f"\n訓練データにのみ存在するクラス:")
    for label in sorted(train_only):
        print(f"  {class_names[label]}")

# 検証データにあって訓練データにないクラス
val_only = set(val_unique) - set(train_unique)
if val_only:
    print(f"\n検証データにのみ存在するクラス:")
    for label in sorted(val_only):
        print(f"  {class_names[label]}")

# %%
# テストデータ評価
# =============================================================================

print("タイムステップごとのテスト評価を開始します...")
# テストデータ評価の呼び出し（シンプル）
timestep_results = ev.evaluate_test_data_timestep(
    model, test_x, test_y, data_config['feature_idx'],
    feature_vocabs, val_dataset, device, criterion
)

# 結果の可視化（evaluation2.pyの関数呼び出しのみ、ファイル保存なし）
print("結果を可視化中...")
ev.plot_timestep_results(timestep_results)

# 詳細分析の表示
ev.print_detailed_timestep_analysis(timestep_results, train_dataset)

# サマリーを表示
ev.print_test_summary(timestep_results)

# %%
# モデルと結果の保存
# =============================================================================

print("=== モデルと結果の保存を開始 ===")

# 1. モデルの保存
save.save_model_and_training_state(save_dir, best_model_state, model, optimizer, scheduler)

# 2. 設定の保存
save.save_hyperparameters_and_config_legacy(
    dataset_config['strains'], data_config['nmax'], data_config['nmax_per_strain'], 
    data_config['test_start'], data_config['ylen'], data_config['val_ratio'],
    data_config['feature_idx'], train_dataset, val_dataset, model,
    model_config['num_epochs'], model_config['batch_size'], train_losses, val_losses,
    train_accs, val_accs, best_val_acc, feature_vocabs,
    device, save_dir, feature_mask=feature_mask
)

# 3. 語彙辞書の保存
save.save_vocabularies(save_dir, feature_vocabs, train_dataset)

# 4. 訓練履歴の保存
save.save_training_plots(train_losses, val_losses, train_accs, val_accs, scheduler, save_dir)

# 5. テスト結果の保存
save.save_test_results(timestep_results, save_dir)

# 6. READMEの保存
save.save_readme(dataset_config['strains'], model, train_dataset, val_dataset,
                  model_config['num_epochs'], train_accs, val_accs, best_val_acc,
                  feature_vocabs, save_dir)

# 7. 実験サマリーを保存
save.save_experiment_summary(dataset_config['strains'], train_dataset, val_dataset, model,
                             model_config['num_epochs'], train_accs, val_accs, best_val_acc,
                             timestep_results, save_dir)

print(f"\n=== 全ての保存が完了しました ===")
print(f"保存先: {os.path.abspath(save_dir)}")

# 最終的なファイル確認
if os.path.exists(save_dir):
    files = os.listdir(save_dir)
    print(f"保存ファイル数: {len(files)}")
    print("保存されたファイル:")
    for file in sorted(files):
        file_path = os.path.join(save_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file}: {size:,} bytes")
    
    # 保存完了の確認
    expected_files = [
        "best_model.pth", "config.json", "feature_vocabularies.pkl", 
        "label_encoder.pkl", "training_history.png", "test_results.json",
        "README.md", "experiment_summary.json"
    ]
    
    missing_files = [f for f in expected_files if f not in files]
    if missing_files:
        print(f"\n警告: 以下のファイルが見つかりません: {missing_files}")
    else:
        print(f"\n✅ 全ての重要ファイルが正常に保存されました")
else:
    print(f"❌ 保存ディレクトリが見つかりません: {save_dir}")

print("\n=== 実験完了 ===")
print(f"実験名: {current_time}")
print(f"対象変異株: {dataset_config['strains']}")
print(f"最良検証精度: {best_val_acc:.4f}")
print(f"総エポック数: {model_config['num_epochs']}")
print(f"モデル設定: d_model={model_config['d_model']}, nhead={model_config['nhead']}, num_layers={model_config['num_layers']}")

# %%