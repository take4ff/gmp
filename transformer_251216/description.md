# transformer_251216

## 概要

transformer_251209をベースに、**Origin Attention（原点参照アテンション）**を追加したバージョン。
モデルが「現在の変異」と「原点（Wuhan株）」の関係を常に直結で参照できるようになり、長期的な予測精度の向上を目指す。

---

## v1209からの変更点

### 1. 新機能: Origin Attention

| 項目 | 説明 |
|------|------|
| **目的** | 原点（Wuhan株）からの距離をモデルに常に意識させる |
| **実装** | Cross-Attention層 + 学習可能なOrigin埋め込みをモデルに組み込み |
| **メリット** | 時系列が長くなっても原点を直結で参照可能 |

#### 設定（config.py）
```python
USE_ORIGIN_ATTENTION = True      # Origin Attentionの有無
ORIGIN_ATTENTION_HEADS = 4       # Attentionのヘッド数
```

#### 学習可能なOrigin埋め込み（model.py）
```python
# 「変異なし」の原点を表す専用ベクトル
# データセットから渡す必要がなく、モデルが「原点の意味」を自動学習
self.origin_embedding = nn.Parameter(
    torch.randn(1, 1, config.FEATURE_DIM) * 0.02
)
```

---

### 2. アーキテクチャ比較

#### v1209
```
Input → Embedding → CoOccurrence → Conv1D → Transformer → Prediction
```

#### v1216
```
Input → Embedding → CoOccurrence → Conv1D → Origin Attention → Transformer → Prediction
                                              ↑
                                    origin_embedding（学習可能）
```

---

### 3. Origin Attention の処理フロー

```
x_combined (Conv1D後の局所文脈)
    │
    ├─→ Query として Attention に入力
    │         │
    │         ↓
    │    attn(Q=x_combined, K=origin_embedding, V=origin_embedding)
    │         │
    │         ↓
    │    origin_context (原点との比較情報)
    │         │
    └─────────┼─→ x_combined + origin_context (残差結合)
              │
              ↓
         Transformer Encoder へ
```

---

### 5. 変更ファイル一覧

| ファイル | 変更内容 |
|---------|----------|
| **config.py** | `USE_ORIGIN_ATTENTION`, `ORIGIN_ATTENTION_HEADS` 追加 |
| **model.py** | `OriginAttention`クラス追加、学習可能な`origin_embedding`追加 |

※ dataset.py、train.py、evaluate.py は変更不要（学習可能な埋め込みを使用するため）

---

## Ablation Study用フラグ

| フラグ | デフォルト | 説明 |
|--------|-----------|------|
| `USE_LOCAL_CONV1D` | True | Conv1D局所特徴抽出の有無 |
| `USE_ORIGIN_ATTENTION` | True | 原点参照Attentionの有無 |

これらのフラグを切り替えることで、各コンポーネントの効果を比較実験できます。

---

## 実行方法

```bash
cd /mnt/ssd1/home3/aiba/gmp
conda activate gvp25-05
python -m transformer_251216.main
```