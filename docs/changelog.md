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

---

### 巨大共起グループ（Omicron期）による学習中 OOM（修正済み・2026-08-03、transformer_260723）

**症状**: `walk_forward` 学習中、特定 fold（fold_3・fold_4 等）で長時間経過後にプロセスが OOM-Kill される。前項（2026-07-29）の R-Precision 評価フェーズ分離後も、学習フェーズ自体で再発。

**原因**: 2つの経路が特定された。いずれも Omicron 系統で祖先枝1本に大量サンプルがぶら下がる巨大な共起グループ（同一 `input_path_str` を共有し any-of-set 統合されるサンプル群。fold_3 test で最大14,684件、fold_4 train で最大13,871件を実測、2026-07-30/31）が起因。

1. **`_build_group_label_cache` の Copy-on-Write 崩壊**: `db/dataset.py` のグループ別ラベルキャッシュはメインプロセスで1回だけ構築し fork 後の全 worker に Copy-on-Write で共有される設計だが、CPython は読み取りだけでも参照カウントの書き込みを伴うため、`DATALOADER_PERSISTENT_WORKERS=True` で全 epoch（15epoch×数百〜千バッチ）生存する worker がバッチ処理のたびに少しずつページを複製し、長時間かけて実質的に worker 数分重複する。巨大グループを含む split では 1 worker あたり約6GBまで肥大化し、8 worker 合計・評価用待機 worker との合算で OOM した（2026-07-31実測）。
2. **R-Precision 動的 K の瞬間的巨大確保**: `evaluate_topk` の動的 K はバッチ内の最大 `target_set` 長（any-of-set 評価でグループ全メンバーのターゲットを合算するため、グループサイズがそのまま長さになる）を `need_k` として全サンプル共通で使うため、バッチにたった1サンプルでも巨大グループが混入すると、バッチ全体に対し `torch.topk(..., k=need_k)` と CPU化（`.tolist()`）が走り、`k×batch_size` 相当のテンソル/Pythonリストを瞬間的に確保して OOM する（fold_3、Omicron早期BA.1/BA.2で最大14,684人を実測、2026-07-30）。

**修正内容**（`config.py` にコメント付きで記録）:
1. `MAX_GROUP_MEMBERS_FOR_CACHE = 4000` を新設。これを超えるグループはメンバーを決定的に（代表サンプルIDをseedに）ランダムサブサンプリングし、キャッシュ・Soft Target分配・any-of-set評価の対象から間引く（代表サンプル自身は常に残す。`None`で無効化）。
2. `TRAIN_NUM_DATALOADER_WORKERS = NUM_DATALOADER_WORKERS // 2` を新設し、学習用ローダーの worker 数を絞って Copy-on-Write 複製の総量を抑制。
3. `MAX_R_PRECISION_K = 4000` を新設し、R-Precision評価の `need_k` に上限を設定。`group_label_count_histogram.py`（新規追加）でfold別のグループ内ユニークラベル数を実測した結果、fold_3のみ突出（最大12,751）で他fold（1,2,4,5,6,7）は最大でも3,510（fold_2）だったため、4000に設定すればfold_3以外は打ち切りなしで厳密な評価を維持しつつfold_3の外れ値のみ打ち切れる。
4. `train.py`／`evaluate.py`／`main.py` にトレーニング中のメモリ使用量を追跡するロギングを追加。`plot_group_count_by_month.py`・`group_label_count_histogram.py`を新規追加し、共起グループサイズの分布を可視化できるようにした。

**未検証**: `MAX_GROUP_MEMBERS_FOR_CACHE`・`MAX_R_PRECISION_K` の打ち切りが fold_3 の評価指標（Recall@K・R-Precision等）に与える影響の定量評価は未実施。間引き後もfold_3が他foldと同じオーダーの値を示すかの確認が今後の課題。

---

### `gc.freeze()` によるCopy-on-Write崩壊の根本対策（対処済み・2026-10-05、transformer_260817）

**背景**: 上記（2026-08-03）の対策は、巨大キャッシュによるCOW崩壊の「複製される量」を `MAX_GROUP_MEMBERS_FOR_CACHE` 等で削る対策であり、「複製が起きる頻度」自体は止めていなかった。

**メカニズムの深掘り**: COW崩壊は、CPythonの循環参照GC（世代0は約700オブジェクト生成ごとに自動起動）が `_group_label_cache`（巨大なネストしたdict/list/tuple）を定期的に走査することが引き金になっている。GCは生存確認のため各オブジェクトの参照カウントを読むだけだが、CPythonの参照カウントはオブジェクトヘッダに格納されているため「読むだけ」でも参照カウントの書き込みが発生し、forkされたworkerプロセスではこれがCopy-on-Writeのページ複製トリガーになる。`DATALOADER_PERSISTENT_WORKERS=True` でworkerが全epochに渡り生存し続けるため、GCが走るたびに複製が進行し、最終的にworker数分の実質コピーが溜まってOOMする。

**対処内容**: `db/dataset.py:create_db_dataloader()` で `DBIterableDataset` 構築直後（`_group_label_cache` 構築完了後、DataLoaderがworkerをforkする前）に `gc.freeze()` を呼ぶよう追加。これはその時点までに生成済みの全オブジェクトを「永続世代」に移し、以降の循環GCの走査対象から除外する。複製が起きる経路自体を断つため、既存の `MAX_GROUP_MEMBERS_FOR_CACHE` 等（量を削る対策）とは独立かつ併用可能。train/valid/test全ローダー経路（`make_train_loader`等によるepoch毎の再構築含む）が `create_db_dataloader()` を通るため、1箇所の変更で全てカバーされる。

```python
# db/dataset.py:create_db_dataloader()
dataset = DBIterableDataset(...)
gc.freeze()  # _group_label_cache構築済み・worker fork前
```

**未検証**: 実際のwalk_forward実行（特にfold_3）でのRSS推移の改善効果は、既存の `_log_rss()` ロギング（main.py）を使った before/after 比較がまだ未実施。次の実行で確認する。

---

### 不正な `collection_date`（`'2022/2024'` 217件）が fold4 の test・fold5 の train に混入（修正済み・2026-10-08、transformer_261008）

**症状**: `reference/sequences-241017_2.csv` 由来の217件（29株、ジンバブエ・Harare、release_date=2025-07-18、元データの入力ミス）が、walk_forward の fold4 の test と fold5 の train に混入していた。表示側（月別集計）で YYYY-MM 形式でない行を除外する応急対応のみで、DB の割当は未修正だった（2026-07-17）。

**原因**: split 割当は `RPAD(collection_date,10,'-01-01')` の**文字列比較**。`'/'`(0x2F) が `'-'`(0x2D) より大きいため、`'2022/2024'` が `2022-07-01 <= x < 2023-01-01` の範囲に偶然入る。

**修正内容**（案A: 割当SQLにガード。DBのデータは書き換えない・再前処理不要）:
1. `db/queries.py` に `VALID_DATE_REGEX`（`YYYY` / `YYYY-MM` / `YYYY-MM-DD`）と `valid_date_sql()` を追加し、`assign_wf_splits`（Fold1の train 含む全UPDATE）・`assign_date_splits`（不正形式は -1）の述語に AND。
2. 同じ述語を使う全箇所に適用: `_xai_common.assign_fold_test_window`、`attention_by_month.py`、`majority_baseline.py`、`lineage_accuracy_drivers.py`、`verify_batch_local_grouping.py`、`generate_point_in_time_freq.py`（割当側だけ直すと分析側の test が学習時とずれるため）。
3. `preflight_check.py` の参照実装 `expected_wf_split` を更新（不正形式は常に -1）。不正形式の検知は fail → warn（割当から除外済み）。

**実DBでの影響（読み取り専用で確認）**: fold4 の test（917,905→917,688）と fold5 の train（同）から**ちょうど217件**が除外。他の fold は不変。fold4 の test に占める割合は約0.04%で、数値への影響は小さい。過去の run（fold4・fold5）は再実行しない。

**テスト**: `tests/test_assign_wf_splits.py` に、不正形式8種×4つのfold設定、dateモード、軽量割当と全割当の一致、`valid_date_sql` の形式判定を追加（既知問題の `xfail(strict)` は解消）。ガード削除・正規表現緩和・Fold1側の削除などの変異テストで検出を確認。

**未対応（別件）**: `collection_date` が空文字の710件は従来仕様（Fold1 は train、Fold N は除外）のまま。

---

### 学習が黙って失敗・再現不能になる経路の対策（対処済み・2026-10-08、transformer_261008）

AIコーディングで入りうる静かな不具合を減らすため、プロジェクト全体を調査して「学習前に必須」の4点を入れた。

1. **空splitの検知**: trainが0件でも、DataLoaderは0バッチ・`train_one_epoch`は例外なく loss=0 を返して学習が「成功」していた（再現済み）。`create_db_dataloader`が0件で `EmptySplitError`（`allow_empty=True`で許可）、`train_one_epoch`が1バッチも処理しなければ `RuntimeError`。
2. **実行の来歴の保存**: run出力に `provenance.json`（git commit・branch・dirty・変更/未追跡ファイル・python/torch/numpy/duckdbの版・GPU・DBのサイズと更新時刻）と、dirtyなら `code_changes.patch`（追跡ファイルの`git diff HEAD`）を保存（`utils/provenance.py`）。これまではconfigだけで、コードの版が残らなかった。**未追跡ファイルはpatchに入らないため、実行前にコミットしておくこと**。
3. **起動前ゲート**: `scripts/eval/walk_forward.py`が、起動時に`pytest -x`（合成データのみ、数秒）と実DBのプリフライトを自動実行し、失敗したら学習を始めない（`--skip_gate`で省略可）。
4. **主貢献の性質テスト**: Co-occurrence Attentionの出力が、同一タイムステップ内の共起変異の順序・スロット配置に依らない（差は1e-6、数値誤差）こと、timestep順には敏感であること、PADの中身が出力に影響しないことを、Broadcast-back／Region条件付けON時も含めてテスト化。

テストは156件。変異テスト（空splitガード削除、来歴のdirty判定、スロット位置依存の導入、共起マスク無効化、ゲートの無効化など）で検出を確認した。

**調査で見つかった残課題**（B・C項目として別途）: `evaluate.py`のカバレッジ4%（報告される指標そのもの）、`feature.py`/`preprocess.py`は0%、`CLAUDE.md`が`.gitignore`対象、`getattr(config,X,default)`が274か所でフラグ名のタイプミスが黙って無視される、定義済み248フラグのうち12個が未配線、`mutation_freq.py`に`__main__`ガードが無くimportで実行される。

---

### テストの拡充（B項目）と、テストで見つかった問題（対処済み・2026-10-08、transformer_261008）

プロジェクト全体の調査（カバレッジ19%、`evaluate.py` 4%、`feature.py`/`preprocess.py` 0%）を受けて追加。テストは202件（カバレッジ 19%→28%、`evaluate.py` 4%→76%、`feature.py` 0%→23%、`preprocess.py` 0%→23%）。

- **B1 報告される指標**: `evaluate`/`evaluate_topk`（Top-K の hit/precision/recall、R-Precision、位置の許容誤差、階層マスク）と集計関数（`calculate_metrics`・weighted/macro recall）を、偽モデル＋合成DBで手計算の期待値と照合。`evaluate`のサンプル別hitと集計値の整合も確認。
- **B2 モデル入力・ラベル**: 12塩基の極小ゲノムで特徴量（コンテキスト塩基・同義判定・累積カウント・状態復元）を手計算と照合。前処理の中核（`process_strain_features_core_chunked`）で、ラベルが最終ステップから作られること、`raw_path`が切り詰められない/設定時は末尾を保持すること、除外統計、サンプル間で参照ゲノムが持ち越されないことを確認。
- **B3 import**: 全107モジュールがimportでき（約1.5秒）、`scripts/`直下に実行文が無く、`def main`には`__main__`ガードがあること。
- **B4 設定**: コードが参照する`config.X`/`getattr(config,'X')`が定義済みであること、未配線フラグは許可リスト（現在12件）で管理、CLAUDE.mdの既定値表がconfigと一致。
- **B5 数値回帰**: 固定seedの順伝播と1エポック学習（loss・タスク重み・更新量）を`tests/golden/numerics.json`と照合（更新は`UPDATE_GOLDEN=1`）。
- **B6 既定値変更の互換**: 既定Trueでパラメータを追加するフラグは、旧checkpoint互換の登録が必須（挙動ベースで検出）。

**テストで見つかった問題（修正済み）**
1. `PLOT_TOP_N_LINEAGES`がconfigに未定義で、`getattr(...,30)`により設定不能のまま黙って30固定だった → configに追加（値は同じ30）。
2. `USE_SUBSTITUTION_HEAD`（既定True、パラメータ追加）が旧checkpoint互換に未登録だった → `_OFF_IF_ABSENT_FROM_SNAPSHOT`に追加。
3. `scripts/analysis/mutation_freq.py`が、モジュール直下で全データを読む解析を実行し、importで固まっていた → `main()`化し`__main__`ガードを追加（内容は不変）。

**調査で分かった仕様（テストで記録、変更なし）**
- **同一コドン内の共起変異の特徴量は、記載順に適用され順序依存**（後の変異のaa_before/afterが前の変異の影響を受ける）。モデルの共起集約は順序不変だが、特徴量生成の段階では順序依存が残る。論文では「共起集約が順序の不確実性に対処する」主張の範囲（同一コドン内の稀なケースで特徴量が順序に依存する）に注意。
- `evaluate`の`detailed_results['hit_*']`は、集計の hit_rate とは別経路で計算される（整合性テストで一致を確認）。

変異テストは累計で約50種（指標・特徴量・前処理・設定・importの各変異を含む）。いずれも該当テストが失敗することを確認。

---

### 運用面の対策（C項目）: 割当状態の記録・フォールバックの厳格化・AIの変更ルール（対処済み・2026-10-08、transformer_261008）

1. **C4 DB割当の状態を記録・検査**: `split_type_wf`（共有DBの可変な状態）を書く処理が、割当の内容を`split_state`テーブル（kind=full/test_only、fold窓、valid比率、seed、件数、時刻）に記録する（`assign_wf_splits`・`assign_fold_test_window`・`attention_by_month`）。読む側（`DBIterableDataset`）は、①test専用割当のままtrain/validを読む、②`full`割当の窓が現在の設定と一致しない（別foldの割当が残っている）、のとき`StaleSplitError`で止まる。記録が無い（旧DB）場合は検査しない。`preflight_check`が現在の割当状態を表示する。窓の一致は`full`のみ検査する（分析スクリプトは「割当→checkpoint読込で窓が上書き→再割当」の順序に依存するため）。
2. **C5 フォールバックの厳格化**: `config.STRICT_FALLBACKS=True`（既定）で、`utils/logging.py::fallback_or_raise`が、これまで警告のみで続行していた次の箇所を`FallbackError`で止める — 事前学習checkpoint無し（ランダム初期化）／ホモプラシーCSV無し・読込失敗（機能が無効化）／**point-in-time頻度表が無い（codon_freqが0になり、リーク対策の設計が黙って崩れる）**／kNNデータストア無し・空／config_snapshot無し（現行configで続行）／前処理の入力CSV無し・読めないデータファイル（サンプルが欠落）。`False`で従来の警告のみに緩和できる。全foldのpoint-in-time表(2021-01-01〜2024-01-01)が存在することをテストで確認。
3. **C1/C2**: `CLAUDE.md`を`.gitignore`から外して追跡対象に。「AI（Claude）が変更を入れるときのルール」12項目を追記。

**未対応（棚卸し結果）**: `except Exception`が主要経路にまだ約60か所ある。このうち学習・評価の結果に影響しうるもの（`db/queries.py`の統計計算、`utils/io.py`のキャッシュ読み書き、`preprocess.py`のキャッシュ・メモリ監視など）は個別に精査していない。ログのみで継続する箇所を今後`fallback_or_raise`か明示的な例外へ置き換える候補。

---

### パッケージ名を固定名 `transformer` に改名（対処済み・2026-10-08）

日付付きディレクトリのコピー運用（`transformer_260702` → … → `transformer_261008`）を廃止し、パッケージ名を固定の `transformer` にした。版はgitのタグ（`git tag run-<日付>-<名前>`）と、run出力の`provenance.json`（commit hash・`git describe`）で管理する。

- **理由**: 版を上げるたびに全参照（コード・テスト・`petra/`・docs・出力パス、今回は122ファイル・626か所）を書き換える必要があり、AIの一括置換のリスクが毎回かかる。旧版を`old_file/`（`.gitignore`対象）に退避する運用では、手元のコピーにしか残らない。旧checkpointは`_OFF_IF_ABSENT_FROM_SNAPSHOT`と互換テストで現行コードのまま読めるため、旧コードを残す必要も小さい。結果のrunがまだ新名で1つも無い（旧名と新名にまたがらない）今が、最も影響が小さい時期だった。
- **実施**: `git mv transformer_261008 transformer`、`transformer_261008`の完全一致置換（コード・テスト・`petra/`・出力パス`outputs/transformer/`）。履歴を含むdocsは置換せず、CLAUDE.mdのバージョン節を書き換え。`transformers`（HuggingFace）とは別名で衝突しない。
- **テストで見つかった取り残し（過去の版更新で更新されていなかった旧名）**: `utils/plotting.py`の関数内に `from transformer_260416 import config`（存在しないモジュール。該当機能の実行時に`ImportError`）と、出力パスの既定値 `outputs/transformer_260416/...`、各scriptsのUsage例の旧run dir。すべて新名へ修正。importの確認では関数の内側は検知できないため、`tests/test_package_name.py`（旧名のimport・`-m`実行・出力パス・日付付きディレクトリの再作成を検知）を追加。
- **`provenance.json`に`git describe`を追加**（直近のタグ。無ければ短縮hash、未コミット変更があれば`-dirty`）。
- 中断された起動が残した空のrun（`outputs/transformer_261008/.../walk_forward_meta.json`のみ）と小さなログを削除。

---

### 共起変異の特徴量生成を順序非依存にするオプション（実装済み・2026-10-08、`FEATURE_ORDER_INDEPENDENT`、既定False）

**背景**: 従来の特徴量生成は、1ステップ内の共起変異を位置昇順に1つずつ共有のゲノム状態へ適用しながら作るため、同一コドン内の後の変異のaa_before/after・同義判定・置換スコア・ホスト適応が「存在しない中間アミノ酸」に基づき、±5bp以内の変異があると前後文脈にも先の変異の塩基が入る。実データ（無作為30万パス・約1,010万ステップ）で、共起ステップは全体の24.1%（100%が位置昇順で記載）、うち同一コドン内に2変異以上が13.7%（全ステップの3.3%）、±5bp以内に別変異が21.7%。同一コドンのステップでは、ほぼ全てで特徴量が、8.6%で同義/非同義ラベルが順序に依存するが、予測対象（最終ステップ）では全パスの0.07%と小さい。

**実装**（`db/feature.py::Step_features_order_independent`、`Mutation_features_fast`が`config.FEATURE_ORDER_INDEPENDENT`で切り替え）:
- 全変異を**そのステップ開始時のゲノム状態**に対して独立に評価する。
- 同一コドン内で適用される変異は、そのコドンに対する**全変異を合成した後のコドン**を共有する。コドン単位の特徴量（aa_before/after・同義判定・置換スコア・ホスト適応）が全メンバーで揃う。位置・塩基・codon_pos・再発頻度は変異固有のまま。
- 前後文脈の塩基はステップ開始時のゲノムから取る。評価後に、適用可能な全変異をゲノム状態へ適用（順序に依らず同じ最終状態）。累積カウント（cum_syn/nonsyn）は変異ごとに数える。

**DB・キャッシュ**: `get_feature_config_hash`・`get_config_hash`はFalseで従来と同一（現行DB`features_b516bca4`・株キャッシュ`d9434d3c`をそのまま使う。テストで固定）、Trueのときだけ別ハッシュ（`features_7bba0728`・`2b600965`）になる。構築: `python -m transformer.preprocess --order_independent`（前回実績で約2.6時間、約730GB）。学習でこのDBを使うには`config.FEATURE_ORDER_INDEPENDENT=True`にする。

**テスト**（`tests/test_feature_order_independent.py`、14件）: OFFで従来と同一（DB名・ハッシュ）、相互作用の無い300通りのランダムなステップで従来の逐次適用と全フィールド一致、順序の入れ替えで各変異の特徴量・最終状態が同一（300通り、相互作用ありを100超含む）、同一コドンのコドン単位特徴量が揃う（AGT→A1T,G2C→TCT: 従来は両方非同義、新は両方同義）、文脈はステップ開始時、全変異適用後の状態、状態の復元、前処理ラベルへの反映。変異テスト6種で検出を確認。

**位置づけ**: 現行の結果（逐次適用の規約）は変えない。順序非依存版との比較は、論文のアブレーション候補（11/10頃の判断点）。論文では、現行の特徴量が「位置昇順に逐次適用する規約」で生成されていることを明記する。

