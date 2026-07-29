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

---

### `walk_forward` の R-Precision 評価フェーズでの OOM（修正済み・2026-07-29、transformer_260723）

**症状**: `walk_forward` 実行時、fold_1・fold_2 は完走するが、より重い fold（例: fold_3）で `run_final_evaluation` 内の R-Precision 評価（`evaluate_topk`）の途中で DataLoader worker が Tracebook なしに消え、プロセスごと OOM-Kill される。単独 fold で再実行しても、Test評価自体は通過するのに直後の R-Precision フェーズで同様に落ちることを確認（=fold間の蓄積ではなく、この設計自体の問題と判明）。

**原因**: `run_final_evaluation` は Validation/Test 各1回の `evaluate()` の後、**同じ** val_loader・test_loader（学習開始以降 `persistent_workers=True` で生存し続けている）をそのまま使って `evaluate_topk` をもう1周走らせていた。評価用ローダーは学習用と同じ `NUM_DATALOADER_WORKERS`（既定8）を使うため、val+test 合計で最大16 workerが、学習相当の長時間にわたり同時生存し続ける設計になっていた。

**修正内容**:
1. `config.EVAL_NUM_DATALOADER_WORKERS`（既定 `NUM_DATALOADER_WORKERS` の半分）を新設し、val/test（評価用）ローダーはこちらを使うよう `db/dataset.py:create_db_dataloader` に `num_workers_override` を追加。
2. `run_final_evaluation` から R-Precision（`evaluate_topk`）部分を `run_topk_evaluation` として分離。`main()` は可視化・Ensemble評価で val/test_loader を使い終えた後、明示的に `del val_loader, test_loader; gc.collect()` してから `make_val_loader()`/`make_test_loader()` で新規（かつ半減された worker 数の）ローダーを作り直し、R-Precision 用に渡す。

```python
# main.py
del val_loader, test_loader
gc.collect()
val_loader = make_val_loader()
test_loader = make_test_loader()
run_topk_evaluation(model, val_loader, test_loader, run_output_dir,
                    val_metrics, test_metrics, val_loss, test_loss)
```
fold_3 を単独プロセスで再検証し、修正前に OOM していた R-Precision フェーズを完走することを確認済み。

---

### `save_config_copy()` が実行時オーバーライドを反映しない問題（修正済み・2026-07-29、transformer_260723）

**症状**: `walk_forward`（`_wf_patch_config()` が `USE_POINT_IN_TIME_FREQ=True` を実行時に強制）で学習したチェックポイントの `config_snapshot.py` を読み込むと、`USE_POINT_IN_TIME_FREQ` が既定値の `False` に戻っている。`evaluate_only.py` や XAI/分析スクリプト（`_xai_common.load_config_snapshot`）がこのスナップショットを再読込すると、学習時とは異なる設定で動いてしまう。

**原因**: `utils/io.py:save_config_copy()` が `config.py` ファイルそのものを `shutil.copy2` していただけで、実行時に上書きされた値（`_wf_patch_config()` や `ABLATION_MASKS` 相当の動的オーバーライド）を一切反映していなかった。

**修正内容**: `config` モジュールのランタイム属性値を走査し、モジュール・関数などシリアライズ不可能な値を除いて `NAME = repr(value)` 形式で書き出す方式に変更（再読込側は既存通り `hasattr(config, name)` でフォールバック）。

---

### walk-forward 分析スクリプトでの `USE_POINT_IN_TIME_FREQ` 揮発によるリーク再混入（修正済み・2026-07-29、transformer_260723・260707）

**症状**: `feature_importance.py` で walk-forward チェックポイント（例: fold_3, Omicron early BA.1/BA.2）を分析すると、`codon_freq`（現 `mutation_recurrence_freq`、`x_num[...,0]`）の重要度が他特徴量を2桁近く上回る。

**原因**: 上記 `save_config_copy()` のバグにより、学習時は `USE_POINT_IN_TIME_FREQ=True`（fold の `split_date` 時点までの変異頻度のみ使用）で正しくリーク対策されていたにもかかわらず、`feature_importance.py` は checkpoint 同梱の `config_snapshot.py` を再読込するだけで `_xai_common.assign_fold_test_window()` を呼んでいなかった（同関数を呼ぶ他の XAI スクリプト群は `TEMPORAL_SPLIT_DATE` 等は補正していたが `USE_POINT_IN_TIME_FREQ` は補正対象に含めていなかった）。結果、分析時は静的 `FREQ_CSV`（データセット全期間で1回だけ計算、test窓より未来の頻度情報を含む）由来の値に揮発し、時系列リークが分析時に再混入していた。

**修正内容**:
1. `_xai_common.py:assign_fold_test_window()` に `config.USE_POINT_IN_TIME_FREQ = True` の強制設定を追加（この関数を呼ぶ全 walk-forward 分析スクリプトに波及）。
2. `feature_importance.py` に `--checkpoint` パスから `fold_N` を自動検出し `assign_fold_test_window()` を呼ぶロジックを追加（従来はこの自動検出・呼び出し自体が無かった）。
3. `codon_freq` は「コドン使用頻度」ではなく塩基位置×置換パターン単位の変異再発頻度であるため、表示名を `mutation_recurrence_freq` に変更（`utils/codon_freq.py` という無関係な別モジュールと同名だった点も解消）。

fold_3 で再実行した結果、重要度は 0.000946→0.000212（約4.5倍減）に低下したが、依然として1位（2位比 2.4倍→2.0倍）。リークを除いても一定の正当な予測シグナル（ホモプラシー変異の再発しやすさ）が残ることを示唆。
