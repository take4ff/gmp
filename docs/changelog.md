# 🐛 既知の不具合・修正履歴

[← README に戻る](../README.md)

### `USE_SUBSTITUTION_HEAD` の `base_after` ラベル付与バグ（修正済み）

**症状**: `base_after_accuracy` が常時 98〜99% と異常に高い値を示す。

**原因**: `db/dataset.py` の `_maybe_extend_labels` において、ラベルの `base_after_token` 付与に2つの誤りがあった。

1. **タイムステップのズレ**: ターゲット（T+1）の変異ではなく、入力の最終タイムステップ（T）の第1変異から `base_after` を取得していた。
2. **共起変異の無視**: T+1 に複数の共起変異があっても、全ターゲットタプルに同一の `base_after_token` を付与していた。

このため `target_base_set` が常に1要素（入力 T の base_after）となり、モデルは入力から答えをコピーするだけで高精度になっていた。残り約 70% は SARS-CoV-2 の C→T 転換優位によるクラス不均衡が寄与。

**修正内容**: `raw_path` の最終セグメント（T+1 の各変異）を `nucpos` でマッチングし、各ターゲット変異に個別の `base_after_token` を付与するよう修正。

```python
# db/dataset.py
def _parse_target_base_map(raw_path):
    last_step = raw_path.split('>')[-1]
    pos_to_token = {}
    for mut in last_step.split(','):
        m = _MUT_NUC_RE.search(mut.strip())
        if m:
            nucpos = int(m.group(1))
            pos_to_token[nucpos] = config.BASE_VOCABS.get(m.group(2), 0)
    return pos_to_token
```

---

### `aa_after` ラベルが PAD(0) 固定だった問題（修正済み・2026-07）

**症状**: `aa_after_accuracy` が 1.000 と表示される（学習・比較で退化に見えた）。

**原因**: `db/dataset.py` の `_maybe_extend_labels` で **`aa_after` ラベルが全サンプル PAD(0) 固定**の暫定実装だった。モデルは定数 0 を当てるだけで accuracy=1.0 になり、加えて損失に無意味な項が加算されていた。

**修正内容**: `reference/codon/codon_mutation4.csv` の置換後コドン列（`>A/>T/>G/>C`）を `db/feature.py:DNA2Protein` で AA に翻訳し `config.AA_VOCABS` の token へ変換した参照表を `db/dataset.py:_get_aa_after_lut()` として1プロセス1回キャッシュ構築。`base_after` と同じく `raw_path` の最終セグメントを `nucpos` でマッチングし、各ターゲット変異に本物の `aa_after` を dataload 時に付与する。**DB 再構築は不要**（例: 266A>T→L, 269A>T→\*（停止コドン）, 非コーディング→n）。

---

### `walk_forward` 実行時の `split_type_wf` 列欠落（修正済み・2026-07）

**症状**: `SPLIT_MODE='walk_forward'` で Fold 1 の最初に `BinderException: Referenced update column split_type_wf not found in table!` で停止。

**原因**: `split_type_wf` 列が追加される前に構築された古いスキーマの DuckDB には同列が無く、`db/queries.py:assign_wf_splits` が `ALTER TABLE` を持たず存在前提で `UPDATE` していた。

**修正内容**: `assign_wf_splits` の冒頭で `PRAGMA table_info('samples')` により列の有無を確認し、無ければ `ALTER TABLE samples ADD COLUMN split_type_wf INTEGER DEFAULT -1` で自動追加する（DuckDB バージョン非依存・**再構築不要**、ALTER はメタデータ操作で一瞬）。

---

### `CO_ATTN_N_LAYERS > 1` での NaN 伝播バグ（修正済み）

**症状**: `CO_ATTN_N_LAYERS=2` に設定すると Epoch 1 から `Val Loss: nan` となり、Early Stopping が即時発動する。Train Loss は正常に低下するが、モデルは実質的に学習されない状態で終了する。

**原因**: Self-Attention 層（`n_layers - 1` 層）において、あるタイムステップの全変異が PAD の場合に `key_padding_mask` が全て `True` となり `softmax([-inf, -inf, ...]) = nan` が発生。nan が LayerNorm → Cross-Attention へ伝播し、Loss 計算が崩壊する。`CO_ATTN_N_LAYERS=1` では Self-Attention 層が存在しないため問題が顕在化しない。

**修正内容**: `CoOccurrenceAttention` の集約出力（Cross-Attention 後）に `torch.nan_to_num` を適用し、全 PAD タイムステップの寄与をゼロに置換。集約の最終出力を直接ガードすることで、Self-Attention 層を持たない `CO_ATTN_N_LAYERS=1` を含む全構成を1箇所で保護する。

```python
# model.py  CoOccurrenceAttention.forward()
if self.out_proj is not None:
    output = self.out_proj(output)

# 全PADタイムステップ（kpm 行が全 True）の集約は不定値になりうるため 0 で埋める。
output = torch.nan_to_num(output, nan=0.0)
```
