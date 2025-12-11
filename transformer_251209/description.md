# Transformer 251209 - 変更履歴

## 概要
このバージョンでは、評価結果の表示・保存・可視化機能の大幅な強化と、コードのリファクタリングを行いました。

---

## 主な変更点

### 1. 強度スコア予測の追加
- モデルに新しい予測ヘッド `strength_head` を追加
- 回帰タスクとしてMSE損失を使用
- 評価時にStrength MAEを計算・表示

### 2. タンパク質位置 (Protein Position) の予測
- `aa_position_head` を `protein_pos` として変数名を統一
- 予測ターゲットとして追加（Ablationマスクから除外）
- Hit Rate, Precision, Recall, F1を計算

### 3. 統合評価レポート (`print_combined_report`)
ValidationとTestを統合した4部構成のレポートを実装：

#### 1. Executive Summary (強度スコア別)
- High/Medium/Lowの3カテゴリで精度を比較
- 閾値は `config.py` で設定可能

#### 2. Biological Analysis (タンパク質領域別)
- 各領域ごとのPrecision, Recall, F1を表示
- ValidationとTestを並列表示

#### 3. Temporal Dynamics (パス長別)
- 長さ1刻みでの精度推移を表示
- ValidationとTestを並列表示

### 4. CSV保存機能
| 関数 | 出力ファイル |
|------|-------------|
| `save_metrics_csv()` | `*_metrics_by_timestep.csv` |
| `save_category_metrics_csv()` | `*_metrics_by_category.csv` |
| `save_prediction_results()` | `*_predictions.csv` |

### 5. グラフ可視化
| 関数 | グラフ内容 |
|------|-----------|
| `plot_metrics_by_timestep()` | Hit Rate, Precision, Recall, F1, MAE, Sample Count (3×2) |
| `plot_category_metrics()` | カテゴリ別Hit Rate, MAE, 分布 |

### 6. コードリファクタリング
| 移動先 | 追加関数 |
|--------|----------|
| `dataset.py` | `prepare_all_data()` - データロード・前処理を一括処理 |
| `utils.py` | `init_wandb()` - WandB初期化 |
| `utils.py` | `print_combined_report()` - 統合レポート表示 |
| `utils.py` | `print_sample_structure()` - サンプルデータ構造表示 |

**main.py**: 463行 → 187行 に簡素化

### 7. サンプリングモードの柔軟化 (2025/12/10追加)
2つのサンプリングモードを `SAMPLING_MODE` で切り替え可能：

| モード | 説明 |
|--------|------|
| `proportional` | 全データから比率ベースでサンプリング (`MAX_NUM` 件) |
| `fixed_per_strain` | 株数制限 (`MAX_STRAIN_NUM`) × 株ごとサンプル上限 (`MAX_NUM_PER_STRAIN`) |

### 8. WandBオフラインモード対応 (2025/12/10追加)
- `WANDB_OFFLINE = True` でオフライン実行
- `finish_wandb()` で自動 `wandb sync` を実行

### 9. 動的強度スコア閾値 (2025/12/10追加)
- 全データの強度スコア分布からLow/Medium/Highの閾値を自動計算
- 3等分の整数閾値を使用（データ依存）
- `data_info['strength_low_max']`, `data_info['strength_med_max']` で取得可能

### 10. 統合レポートの改善 (2025/12/10追加)
- ヘッダー名を簡潔化: 「タンパク質」「塩基位置」「アミノ酸位置」
- Prediction Metrics Summary セクション追加（適合率/再現率/F1を予測対象別に表示）
- 動的閾値をレポートに表示

---

## 設定パラメータ (`config.py`)

```python
# 強度カテゴリ閾値
STRENGTH_CATEGORY_LOW_MAX = 3.0   # Low: < 3.0
STRENGTH_CATEGORY_MED_MAX = 5.0   # Medium: 3.0-5.0, High: ≥ 5.0

# 損失重み
LOSS_WEIGHT_STRENGTH = 1.0
LOSS_WEIGHT_PROTEIN_POS = 1.0
```

---

## 出力例

```
=====================================================================================
【統合評価レポート - Validation vs Test】
=====================================================================================

=== 1. Executive Summary (強度スコア別: High/Medium/Low) ===
  Category |  Val Count | Val Reg | Val Pos | Val Prot ||  Test Count | Test Reg | Test Pos | Test Prot
  high     |       3101 |  36.4%  |  15.7%  |   19.0%  ||        9887 |   33.3%  |   13.4%  |    18.4% ★

=== 2. Biological Analysis (タンパク質領域別) ===
  Region     |  Val Count |  Val Prec |  Val Rec  |  Val F1   || Test Count | Test Prec | Test Rec  | Test F1
  S          |        876 |    75.7%  |    54.7%  |    63.5%  ||       2742 |    66.4%  |    49.5%  |    56.7% ★

=== 3. Temporal Dynamics (パス長別 - 長さ1刻み) ===
  Length |  Val n | Val Reg | Val Pos | Val Prot ||  Test n | Test Reg | Test Pos | Test Prot
      38 |   1146 |  35.3%  |  16.8%  |   20.9%  ||       - |       -  |       -  |        -
      41 |      - |      -  |      -  |        - ||    3566 |   35.0%  |   15.1%  |    19.1%

---

## ファイル構成

```
transformer_251209/
├── __init__.py
├── config.py          # 設定パラメータ
├── dataset.py         # データ処理 + prepare_all_data()
├── model.py           # モデル定義 (4タスク出力)
├── train.py           # 訓練ループ
├── evaluate.py        # 評価 (カテゴリ別メトリクス含む)
├── utils.py           # ユーティリティ (CSV/グラフ/レポート)
├── main.py            # メインスクリプト (簡素化済み)
└── description.md     # このファイル
```
