# 比較実験リスト

デフォルト設定: `BATCH_SIZE=256`（GRAM削減のため512→256）

DBリビルドが必要な設定変更: `CONTEXT_WINDOW`, `NUM_CHEM_FEATURES`, `MAX_SEQ_LEN`, `MAX_CO_OCCURRENCE`

---

## グループ A: 最優先

| # | 実験 | 変更設定 | 選択肢 | DBリビルド |
|---|---|---|---|---|
| A1 | ホスト適応24特徴量 | `ABLATION_MASKS`: `HOST_LOG_RATIO_DIFF`〜`HOST_RSCU_RATIO_DIFF` を一括 | 全 `False`(あり) / 全 `True`(なし) | 不要 |

---

## グループ B: 特徴量アブレーション

| # | 実験 | 変更設定 | 選択肢 | DBリビルド |
|---|---|---|---|---|
| B1 | 累積変異カウント | `ABLATION_MASKS`: `CUM_SYN`, `CUM_NONSYN` | 両 `False`(あり) / 両 `True`(なし) / 個別 | 不要 |
| B2 | コンテキスト塩基（一括） | `ABLATION_MASKS`: `MUTATION_CONTEXT` | `False` / `True` | 不要 |
| B3 | コンテキスト窓幅 | `CONTEXT_WINDOW` | `3` / `5` | **必要** |
| B4 | ホスト適応グループ別 | RSCU系 / CAI系 / CpG系 / Transition系 などグループ単位でOFF | グループ別 | 不要 |

---

->グループA,Bはなしで。損失重みやモデル内部重みで評価。E3優先。

## グループ C: アーキテクチャ

| # | 実験 | 変更設定 | 選択肢 | DBリビルド |
|---|---|---|---|---|
| C1 | Temporal Pooling | `TEMPORAL_POOLING` | `'last'` / `'mean'` / `'cls'` | 不要 |
| C2 | Shared Trunk | `USE_SHARED_TRUNK` | `False` / `True` | 不要 |
| C3 | 相対位置エンコーディング（ALiBi） | `USE_RPE` | `False` / `True` | 不要 |
| C4 | Co-Attn Heads | `CO_ATTN_N_HEADS` | `4` / `8` | 不要 |
| C5 | Co-Attn 層数 | `CO_ATTN_N_LAYERS` | `1` / `2` | 不要 |
| C6 | Co-Attn 内部次元 | `CO_ATTN_DIM` | `256` / `512` | 不要 |
| C7 | 正規化位置 | `NORM_FIRST` | `True`(Pre-Norm) / `False`(Post-Norm) | 不要 |
| C8 | FFN 倍率 | `FFN_RATIO` | `2` / `4` / `8` | 不要 |
| C9 | 活性化関数 | `ACTIVATION` | `'gelu'` / `'relu'` / `'silu'` | 不要 |
| C9 | Substitution Head | `USE_SUBSTITUTION_HEAD` | `False` / `True` | 不要 |
| C10 | Autoregressive Decoder | `USE_AUTOREGRESSIVE_DECODER` | `False` / `True` | 不要 |
| C11 | ESM-2 タンパク質言語モデル特徴量 | `USE_ESM2` | `False` / `True` | **必要**（埋め込み抽出も別途必要） |

---

## グループ D: 損失関数

| # | 実験 | 変更設定 | 選択肢 | DBリビルド |
|---|---|---|---|---|
| D1 | Biologically Informed Loss | `USE_BIO_INFORMED_LOSS` | `False` / `True` | 不要 |
| D2 | Bio Informed σ | `BIO_INFORMED_SIGMA` | `1.0` / `2.0` / `5.0` | 不要 |
| D3 | Strength 損失タイプ | `STRENGTH_LOSS_TYPE` | `'mse'` / `'mae'` / `'huber'` | 不要 |
| D4 | Supervised Contrastive Loss | `USE_SUPCON` | `False` / `True` | 不要 |
| D5 | Label Smoothing | `USE_LABEL_SMOOTHING`, `LABEL_SMOOTHING_FACTOR` | OFF / `0.05` / `0.1` / `0.2` | 不要 |

---

## グループ E: 学習設定

| # | 実験 | 変更設定 | 選択肢 | DBリビルド |
|---|---|---|---|---|
| E1 | Curriculum Learning | `CURRICULUM_LEARNING` | `False` / `True` | 不要 |
| E2 | TTA（推論時Dropout） | `USE_TTA`, `TTA_N_PASSES` | OFF / 3回 / 5回 | 不要 |
| E3 | 事前学習（MLM/CLM） | `USE_PRETRAINING`, `PRETRAINING_MODE` | OFF / `'mlm'` / `'clm'` | 不要（別途 pretrain.py 実行が必要） |

---

## 結果の確認方法

各実験後に `run_summary.json` を比較する。主要指標:

```
val / test:
  macro_recall_region           # リージョン別マクロRecall
  macro_recall_position         # 塩基位置マクロRecall
  top1/3/5_region_hit_rate_pct
  top1/3/5_position_hit_rate_pct
  top1/3/5_aa_pos_hit_rate_pct
```
