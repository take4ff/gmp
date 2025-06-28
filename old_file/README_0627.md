# COVID-19変異予測パッケージ

深層学習を用いたCOVID-19ウイルス変異の予測と解析のための包括的なPythonパッケージです。

## 🌟 特徴

- **🤖 高度なTransformerモデル**: 変異パターンの学習に最適化
- **📊 包括的な評価メトリクス**: 多角的なモデル性能評価
- **⚡ ハイパーパラメータ最適化**: Optunaによる自動チューニング
- **🎯 アンサンブル学習**: 複数モデルの組み合わせによる予測精度向上
- **📈 クラス不均衡対応**: 実世界データに適応した学習戦略
- **🔄 モジュール設計**: 再利用可能で拡張しやすい構造

## 📦 インストール

### PyPIからのインストール（推奨）
```bash
pip install covid-mutation-prediction
```

### 開発版のインストール
```bash
git clone https://github.com/example/covid-mutation-prediction.git
cd covid-mutation-prediction
pip install -e .[dev]
```

## 🚀 クイックスタート

### 1. 基本的な使用方法

```python
from covid_mutation_prediction import CovidMutationPredictor

# 予測器の初期化
predictor = CovidMutationPredictor()

# データの読み込みと訓練
predictor.train(data_path="your_data/", config_path="config.yaml")

# 予測の実行
predictions = predictor.predict(input_sequences)
```

### 2. コマンドライン使用

```bash
# モデルの訓練
covid-mutation-predict --mode train --config config.yaml

# ハイパーパラメータ最適化付き訓練
covid-mutation-predict --mode train --config config.yaml --optimize

# 予測の実行
covid-mutation-predict --mode predict --input data.csv --output predictions.csv
```

## 📊 パッケージ構造

```
covid_mutation_prediction/
├── __init__.py                 # パッケージ初期化
├── config/                     # 設定管理
│   ├── __init__.py
│   ├── settings.py            # 設定クラス
│   └── config.yaml            # デフォルト設定
├── utils/                      # ユーティリティ
│   ├── __init__.py
│   └── reproducibility.py    # 再現性確保
├── data/                       # データ処理
│   ├── __init__.py
│   ├── processor.py           # データ前処理
│   └── dataset.py             # データセットクラス
├── models/                     # モデル定義
│   ├── __init__.py
│   ├── transformer.py         # Transformerモデル
│   └── loss_functions.py      # 損失関数
├── training/                   # 訓練ロジック
│   ├── __init__.py
│   └── pipeline.py            # 訓練パイプライン
├── evaluation/                 # 評価システム
│   ├── __init__.py
│   └── metrics.py             # 評価メトリクス
├── ensemble/                   # アンサンブル学習
│   ├── __init__.py
│   └── ensemble.py            # アンサンブル手法
└── optimization/               # ハイパーパラメータ最適化
    ├── __init__.py
    └── hyperparameter_tuning.py
```

## 🔧 設定

設定は `config.yaml` ファイルで管理されます：

```yaml
# モデル設定
model:
  input_dim: 256
  hidden_dim: 512
  num_layers: 6
  dropout_rate: 0.1

# 訓練設定  
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  patience: 15
```

## 📈 高度な使用例

### アンサンブル学習

```python
from covid_mutation_prediction.ensemble import EnsemblePredictor

# 複数モデルのアンサンブル
ensemble = EnsemblePredictor(models=[model1, model2, model3])
ensemble_predictions = ensemble.predict(test_data)
```

### ハイパーパラメータ最適化

```python
from covid_mutation_prediction.optimization import OptunaOptimizer

# 最適化の実行
optimizer = OptunaOptimizer(n_trials=100)
best_params = optimizer.optimize(param_space, train_data, val_data)
```

## 📊 評価メトリクス

パッケージは以下のメトリクスをサポートします：

- **分類メトリクス**: Accuracy, Precision, Recall, F1-Score
- **確率メトリクス**: AUC-ROC, AUC-PR
- **クラス別メトリクス**: Per-class metrics
- **混同行列**: 詳細な分類結果分析

## 🤝 コントリビューション

コントリビューションを歓迎します！以下の手順でご参加ください：

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は `LICENSE` ファイルをご覧ください。

## 📞 サポート

- **Issues**: [GitHub Issues](https://github.com/example/covid-mutation-prediction/issues)
- **ドキュメント**: [完全ドキュメント](https://covid-mutation-prediction.readthedocs.io/)
- **Email**: research@example.com

## 🔬 論文・引用

このパッケージを研究でご利用の場合は、以下の形式で引用してください：

```bibtex
@software{covid_mutation_prediction,
  title={COVID-19変異予測パッケージ},
  author={AI Research Team},
  year={2025},
  url={https://github.com/example/covid-mutation-prediction}
}
```

## 🎯 ロードマップ

- [ ] Web API の提供
- [ ] リアルタイム予測システム
- [ ] 他のウイルス種への対応
- [ ] GPU最適化の強化
- [ ] AutoMLの完全実装

---

**注意**: このパッケージは研究目的で開発されており、医療診断には使用しないでください。
