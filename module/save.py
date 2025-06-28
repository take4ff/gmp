import os
import json
import pickle
from datetime import datetime
import torch
import matplotlib.pyplot as plt

# 1. モデルの保存
def save_model_and_training_state(save_dir, best_model_state, model, optimizer, scheduler):
    """モデルの重みと訓練状態を保存"""
    
    # 最良モデルの保存
    if best_model_state:
        model_path = os.path.join(save_dir, "best_model.pth")
        torch.save(best_model_state, model_path)
        print(f"最良モデルを保存: {model_path}")
    
    # 現在のモデル状態も保存
    current_model_path = os.path.join(save_dir, "current_model.pth")
    torch.save(model.state_dict(), current_model_path)
    print(f"現在のモデルを保存: {current_model_path}")
    
    # オプティマイザー状態の保存
    optimizer_path = os.path.join(save_dir, "optimizer.pth")
    torch.save(optimizer.state_dict(), optimizer_path)
    print(f"オプティマイザー状態を保存: {optimizer_path}")
    
    # スケジューラー状態の保存
    scheduler_path = os.path.join(save_dir, "scheduler.pth")
    torch.save(scheduler.state_dict(), scheduler_path)
    print(f"スケジューラー状態を保存: {scheduler_path}")

# 2. ハイパーパラメータとモデル設定の保存
def save_hyperparameters_and_config(strains, nmax, nmax_per_strain, test_start, ylen, val_ratio,
                                     feature_idx, train_dataset, val_dataset, model,
                                     num_epochs, batch_size, train_losses, val_losses,
                                     train_accs, val_accs, best_val_acc, feature_vocabs,
                                     device, save_dir):
    """ハイパーパラメータとモデル設定を保存"""
    
    config = {
        # データセット設定
        "dataset_config": {
            "strains": strains,
            "nmax": nmax,
            "nmax_per_strain": nmax_per_strain,
            "test_start": test_start,
            "ylen": ylen,
            "val_ratio": val_ratio,
            "feature_idx": feature_idx
        },
        
        # モデル設定
        "model_config": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "num_classes": train_dataset.num_classes,
            "max_seq_length": train_dataset.max_length,
            "model_parameters": sum(p.numel() for p in model.parameters())
        },
        
        # 訓練設定
        "training_config": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_params": {
                "mode": "min",
                "factor": 0.5,
                "patience": 2
            }
        },
        
        # データ統計
        "data_statistics": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "feature_vocab_sizes": [len(vocab) for vocab in feature_vocabs],
            "total_vocab_size": sum(len(vocab) for vocab in feature_vocabs),
            "class_names": train_dataset.label_encoder.classes_.tolist()
        },
        
        # 訓練結果
        "training_results": {
            "best_val_accuracy": best_val_acc,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "final_train_acc": train_accs[-1] if train_accs else None,
            "final_val_acc": val_accs[-1] if val_accs else None,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs
        },
        
        # メタデータ
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "python_version": "3.9",
            "pytorch_version": torch.__version__
        }
    }
    
    # JSON形式で保存
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"設定ファイルを保存: {config_path}")

# 3. 語彙辞書の保存
def save_vocabularies(save_dir, feature_vocabs, train_dataset):
    """特徴量語彙辞書を保存"""
    
    vocab_path = os.path.join(save_dir, "feature_vocabularies.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(feature_vocabs, f)
    print(f"語彙辞書を保存: {vocab_path}")
    
    # ラベルエンコーダーの保存
    label_encoder_path = os.path.join(save_dir, "label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(train_dataset.label_encoder, f)
    print(f"ラベルエンコーダーを保存: {label_encoder_path}")

# 4. 訓練履歴グラフの保存
def save_training_plots(train_losses, val_losses, train_accs, val_accs, scheduler, save_dir):
    """訓練履歴のグラフを保存"""
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, [scheduler.get_last_lr()[0]] * len(epochs), label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    # グラフを保存
    plot_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"訓練履歴グラフを保存: {plot_path}")
    plt.show()

# 5. テスト結果の保存
def save_test_results(timestep_results, save_dir):
    """テスト結果を保存"""
    
    # 修正: globals()チェックを削除し、引数を直接評価
    if timestep_results and len(timestep_results) > 0:
        # テスト結果をJSON形式で保存
        test_results_json = {}
        for timestep, result in timestep_results.items():
            test_results_json[str(timestep)] = {
                'samples': int(result['samples']),
                'expanded_samples': int(result['expanded_samples']),
                'filtered_samples': int(result['filtered_samples']),
                'loss': float(result['loss']),
                'strict_accuracy': float(result['strict_accuracy']),
                'flexible_accuracy': float(result['flexible_accuracy']),
                'accuracy_improvement': float(result['flexible_accuracy'] - result['strict_accuracy']),
                'predictions': [int(p) for p in result['predictions']],
                'targets': [int(t) for t in result['targets']]
            }
        
        # メタデータを追加
        test_results_json['metadata'] = {
            'num_timesteps': len(timestep_results),
            'timestep_range': [min(timestep_results.keys()), max(timestep_results.keys())],
            'total_samples': sum(result['filtered_samples'] for result in timestep_results.values()),
            'avg_strict_accuracy': sum(result['strict_accuracy'] for result in timestep_results.values()) / len(timestep_results),
            'avg_flexible_accuracy': sum(result['flexible_accuracy'] for result in timestep_results.values()) / len(timestep_results),
            'creation_date': datetime.now().isoformat()
        }
        
        test_results_path = os.path.join(save_dir, "test_results.json")
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results_json, f, indent=2, ensure_ascii=False)
        print(f"テスト結果を保存: {test_results_path}")
        
        # テスト結果グラフの保存
        if len(timestep_results) > 0:
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
            plt.ylabel('Accuracy Improvement')
            plt.title('Multi-label Accuracy Improvement')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            test_plot_path = os.path.join(save_dir, "test_results.png")
            plt.savefig(test_plot_path, dpi=300, bbox_inches='tight')
            print(f"テスト結果グラフを保存: {test_plot_path}")
            plt.show()
    else:
        # timestep_resultsが空またはNoneの場合
        print("⚠️ timestep_resultsが空またはNoneです。空のtest_results.jsonを作成します。")
        empty_results = {
            'message': 'テスト結果が利用できません',
            'reason': 'timestep_resultsが空または無効です',
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'status': 'no_test_data'
            }
        }
        test_results_path = os.path.join(save_dir, "test_results.json")
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(empty_results, f, indent=2, ensure_ascii=False)
        print(f"空のtest_results.jsonを作成: {test_results_path}")

# 6. README.mdファイルの生成
def save_readme(strains, model, train_dataset, val_dataset,
                     num_epochs, train_accs, val_accs, best_val_acc,
                     feature_vocabs, save_dir):
    """実験の概要をREADME.mdとして保存"""
    
    # 安全な値の事前計算
    try:
        model_params = sum(p.numel() for p in model.parameters())
        model_params_str = f"{model_params:,}"
    except:
        model_params_str = "不明"
    
    try:
        total_vocab_size = sum(len(vocab) for vocab in feature_vocabs)
        vocab_size_str = f"{total_vocab_size:,}"
    except:
        vocab_size_str = "不明"
    
    try:
        final_train_acc = f"{train_accs[-1]:.4f}" if train_accs else "N/A"
    except:
        final_train_acc = "N/A"
    
    try:
        final_val_acc = f"{val_accs[-1]:.4f}" if val_accs else "N/A"
    except:
        final_val_acc = "N/A"
    
    # 実験日の事前計算
    experiment_date = datetime.now().strftime('%Y年%m月%d日')
    strain_list = ', '.join(strains)
    
    # f-stringを使わない方法で文字列を構築
    readme_content = """# COVID-19 Mutation Prediction Model

## 実験概要
- **実験日**: {}
- **対象変異株**: {}
- **訓練期間**: タイムステップ 1-30
- **テスト期間**: タイムステップ 31以降

## モデル設定
- **アーキテクチャ**: Mutation Transformer
- **パラメータ数**: {}
- **特徴量次元**: 9次元（8カテゴリカル + 1数値）
- **クラス数**: {}
- **最大シーケンス長**: {}

## データセット
- **訓練サンプル**: {:,}
- **検証サンプル**: {:,}
- **総語彙サイズ**: {}

## 訓練結果
- **最良検証精度**: {:.4f}
- **訓練エポック数**: {}
- **最終訓練精度**: {}
- **最終検証精度**: {}

## ハイパーパラメータ
- **学習率**: 1e-4
- **バッチサイズ**: 16
- **オプティマイザー**: AdamW
- **重み減衰**: 1e-5
- **スケジューラー**: ReduceLROnPlateau

## ファイル構成
- `best_model.pth`: 最良モデルの重み
- `current_model.pth`: 最終モデルの重み
- `optimizer.pth`: オプティマイザー状態
- `scheduler.pth`: スケジューラー状態
- `config.json`: 全設定とハイパーパラメータ
- `feature_vocabularies.pkl`: 特徴量語彙辞書
- `label_encoder.pkl`: ラベルエンコーダー
- `training_history.png`: 訓練履歴グラフ
- `test_results.json`: テスト評価結果
- `test_results.png`: テスト結果グラフ

## 使用方法
```python
# 必要なライブラリのインポート
import torch
import pickle
import json
import module.mutation_transformer as mt

# 設定の読み込み
with open('config.json', 'r') as f:
    config = json.load(f)

# 語彙辞書のロード
with open('feature_vocabularies.pkl', 'rb') as f:
    feature_vocabs = pickle.load(f)

# ラベルエンコーダーのロード
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# モデルの初期化
model = mt.MutationTransformer(
    feature_vocabs=feature_vocabs,
    d_model=256,
    nhead=8,
    num_layers=4,
    num_classes={},
    max_seq_length={}
)

# 最良モデルの重みをロード
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

## 注意事項
- このモデルは特定の変異株（{}）で訓練されています
- 他の変異株に適用する場合は、転移学習を検討してください
- 予測結果は研究目的での使用を想定しています
- モデルの性能は訓練データの品質と量に依存します

## 実験詳細
- **デバイス**: GPU利用可能
- **データ分割**: 時系列考慮分割
- **評価方法**: 厳密評価 + 柔軟評価（マルチラベル対応）
- **早期停止**: 最良検証精度でモデル保存
""".format(
        experiment_date,
        strain_list,
        model_params_str,
        train_dataset.num_classes,
        train_dataset.max_length,
        len(train_dataset),
        len(val_dataset),
        vocab_size_str,
        best_val_acc,
        num_epochs,
        final_train_acc,
        final_val_acc,
        train_dataset.num_classes,
        train_dataset.max_length,
        strain_list
    )
    
    readme_path = os.path.join(save_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"READMEファイルを保存: {readme_path}")

# 実験結果のサマリーも追加保存
def save_experiment_summary(strains, train_dataset, val_dataset, model,
                             num_epochs, train_accs, val_accs, best_val_acc,
                             timestep_results, save_dir):
    """実験結果のサマリーをJSONで保存"""
    
    summary = {
        "experiment_info": {
            "name": "20250628_train1",
            "date": datetime.now().isoformat(),
            "status": "completed"
        },
        "dataset": {
            "strains": strains,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "num_classes": train_dataset.num_classes,
            "max_sequence_length": train_dataset.max_length
        },
        "model": {
            "architecture": "MutationTransformer",
            "parameters": sum(p.numel() for p in model.parameters()),
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 4
        },
        "training": {
            "epochs": num_epochs,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "best_val_accuracy": float(best_val_acc),
            "final_train_accuracy": float(train_accs[-1]) if train_accs else None,
            "final_val_accuracy": float(val_accs[-1]) if val_accs else None
        },
        "test_results": {
            "available": 'timestep_results' in globals() and timestep_results is not None,
            "num_timesteps": len(timestep_results) if 'timestep_results' in globals() and timestep_results else 0
        }
    }
    
    summary_path = os.path.join(save_dir, "experiment_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"実験サマリーを保存: {summary_path}")