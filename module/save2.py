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

# 旧版との互換性を保つためのwrapper関数
def save_hyperparameters_and_config_legacy(strains, nmax, nmax_per_strain, test_start, ylen, val_ratio,
                                          feature_idx, train_dataset, val_dataset, model,
                                          num_epochs, batch_size, train_losses, val_losses,
                                          train_accs, val_accs, best_val_acc, feature_vocabs,
                                          device, save_dir, feature_mask=None):
    """旧版互換のためのwrapper関数"""
    return save_hyperparameters_and_config(model, train_dataset, val_dataset, feature_vocabs, save_dir,
                                          strains=strains, nmax=nmax, nmax_per_strain=nmax_per_strain,
                                          test_start=test_start, ylen=ylen, val_ratio=val_ratio,
                                          feature_idx=feature_idx, num_epochs=num_epochs,
                                          batch_size=batch_size, train_losses=train_losses,
                                          val_losses=val_losses, train_accs=train_accs,
                                          val_accs=val_accs, best_val_acc=best_val_acc, device=device,
                                          feature_mask=feature_mask)

# 2. ハイパーパラメータとモデル設定の保存
def save_hyperparameters_and_config(model, train_dataset, val_dataset, 
                                         feature_vocabs, save_dir, **kwargs):
    """修正版：実際のパラメータ数を動的に計算して保存"""
    
    # JSON非対応オブジェクトを文字列に変換
    def make_json_serializable(obj):
        """JSONシリアライズ可能な形式に変換"""
        if hasattr(obj, 'type'):  # PyTorch device
            return str(obj)
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    # kwargsの全ての値をJSONシリアライズ可能に変換
    for key, value in kwargs.items():
        kwargs[key] = make_json_serializable(value)
    
    # モデルのパラメータ数を動的に計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # モデル設定を動的に取得
    try:
        transformer_layer = model.transformer.layers[0]
        actual_nhead = transformer_layer.self_attn.num_heads
        actual_d_model = model.actual_d_model if hasattr(model, 'actual_d_model') else model.d_model
        actual_num_layers = len(model.transformer.layers)
    except:
        actual_nhead = 'unknown'
        actual_d_model = 'unknown'
        actual_num_layers = 'unknown'
    
    config = {
        "experiment_info": {
            "experiment_date": datetime.now().isoformat(),
            "experiment_name": os.path.basename(save_dir),
        },
        "model_architecture": {
            "type": "MutationTransformer",
            "d_model": actual_d_model,
            "nhead": actual_nhead,
            "num_layers": actual_num_layers,
            "num_classes": train_dataset.num_classes,
            "max_seq_length": train_dataset.max_length,
            "total_parameters": total_params,  # ✅ 動的計算
            "trainable_parameters": trainable_params,  # ✅ 動的計算
            "parameter_breakdown": get_parameter_breakdown(model)  # ✅ 詳細分析
        },
        "dataset_info": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "feature_vocabs_sizes": [len(vocab) for vocab in feature_vocabs],
            "total_vocab_size": sum(len(vocab) for vocab in feature_vocabs)
        }
    }
    
    # 追加の引数をマージ
    config.update(kwargs)
    
    # 保存
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 設定保存完了: {config_path}")
    print(f"✅ 実際のパラメータ数: {total_params:,}")
    
    return config

def get_parameter_breakdown(model):
    """パラメータの詳細分析"""
    breakdown = {}
    
    for name, param in model.named_parameters():
        breakdown[name] = {
            "shape": list(param.shape),
            "parameters": param.numel(),
            "requires_grad": param.requires_grad
        }
    
    return breakdown

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
            # 実際のキー構造に基づいて安全にアクセス
            test_results_json[str(timestep)] = {
                'samples': int(result.get('samples', 0)),
                'expanded_samples': int(result.get('expanded_samples', 0)),
                'filtered_samples': int(result.get('filtered_samples', 0)),
                'loss': float(result.get('loss', 0.0)),
                'strict_sequence_accuracy': float(result.get('strict_sequence_accuracy', 0.0)),
                'flexible_sequence_accuracy': float(result.get('flexible_sequence_accuracy', 0.0)),
                'predictions': [int(p) for p in result.get('predictions', [])],
                'targets': [int(t) for t in result.get('targets', [])],
                # 追加情報（存在する場合のみ）
                'original_labels': result.get('original_labels', []),
                'predictions_per_sequence': result.get('predictions_per_sequence', []),
                'true_labels_per_sequence': result.get('true_labels_per_sequence', [])
            }
        
        # メタデータを追加
        test_results_json['metadata'] = {
            'num_timesteps': len(timestep_results),
            'timestep_range': [min(timestep_results.keys()), max(timestep_results.keys())] if timestep_results else [0, 0],
            'total_samples': sum(result.get('filtered_samples', 0) for result in timestep_results.values()),
            'avg_strict_sequence_accuracy': sum(result.get('strict_sequence_accuracy', 0.0) for result in timestep_results.values()) / len(timestep_results) if timestep_results else 0.0,
            'avg_flexible_sequence_accuracy': sum(result.get('flexible_sequence_accuracy', 0.0) for result in timestep_results.values()) / len(timestep_results) if timestep_results else 0.0,
            'creation_date': datetime.now().isoformat()
        }
        
        test_results_path = os.path.join(save_dir, "test_results.json")
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results_json, f, indent=2, ensure_ascii=False)
        print(f"テスト結果を保存: {test_results_path}")
        
        # テスト結果グラフの保存
        if len(timestep_results) > 0:
            timesteps = sorted(timestep_results.keys())
            # 実際のキー構造に基づいて安全にアクセス
            strict_seq_accs = [timestep_results[ts].get('strict_sequence_accuracy', 0.0) for ts in timesteps]
            flexible_seq_accs = [timestep_results[ts].get('flexible_sequence_accuracy', 0.0) for ts in timesteps]
            sample_counts = [timestep_results[ts].get('filtered_samples', 0) for ts in timesteps]
            
            # 添付ファイルと同じスタイルの3つのサブプロット
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 1. Prediction Accuracy by Timestep
            axes[0].plot(timesteps, strict_seq_accs, 'o-', label='Strict Accuracy', linewidth=2, markersize=6, color='#1f77b4')
            axes[0].plot(timesteps, flexible_seq_accs, 's-', label='Flexible Accuracy', linewidth=2, markersize=6, color='#ff7f0e')
            axes[0].set_xlabel('Timestep')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Prediction Accuracy by Timestep')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, max(max(strict_seq_accs), max(flexible_seq_accs)) * 1.1)
            
            # 2. Sample Count by Timestep (After Filter)
            axes[1].bar(timesteps, sample_counts, alpha=0.7, color='skyblue')
            axes[1].set_xlabel('Timestep')
            axes[1].set_ylabel('Sample Count')
            axes[1].set_title('Sample Count by Timestep (After Filter)')
            axes[1].grid(True, alpha=0.3)
            
            # 3. Multi-label Accuracy Improvement
            accuracy_improvement = [flexible_seq_accs[i] - strict_seq_accs[i] for i in range(len(timesteps))]
            colors = ['green' if improvement >= 0 else 'red' for improvement in accuracy_improvement]
            axes[2].bar(timesteps, accuracy_improvement, color=colors, alpha=0.7)
            axes[2].set_xlabel('Timestep')
            axes[2].set_ylabel('Accuracy Improvement')
            axes[2].set_title('Multi-label Accuracy Improvement')
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[2].grid(True, alpha=0.3)
            
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
            "name": "20250630_train2",
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
            "available": timestep_results is not None and len(timestep_results) > 0,
            "num_timesteps": len(timestep_results) if timestep_results else 0
        }
    }
    
    summary_path = os.path.join(save_dir, "experiment_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"実験サマリーを保存: {summary_path}")