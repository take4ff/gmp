# CLAUDE.md — gmp (Viral Genome Mutation Prediction)

## プロジェクト概要

SARS-CoV-2のゲノム変異発生を予測する時系列マルチタスク Transformer モデル。
先行研究 PETra に対し、**Co-occurrence Attention**（共起変異の順序非依存集約）と
**物理化学特徴量**（電荷・疎水性・コドン頻度等）を追加した提案アーキテクチャを検証する研究プロジェクト。

---

## 実行環境

```bash
conda activate gvp25-05   # PyTorch / DuckDB / W&B が入った環境
```

作業ディレクトリはリポジトリルート (`/mnt/ssd1/home3/aiba/gmp`) から実行すること。

---

## AI（Claude）が変更を入れるときのルール

過去に実際に起きた「クラッシュせず、もっともらしい値を出し続ける」不具合（raw_path切り詰め、グルーピング分断、fold間checkpoint
取り違え、`'2022/2024'`の混入、旧checkpointが読めない、空のtrainで学習が進む等）を防ぐための運用。**テストが通ることを変更の完了条件とする。**

1. **変更の前後で `python -m pytest -q` を実行する**（約13秒）。失敗を残したままコミットしない。`walk_forward`は起動時にも自動実行する。
2. **一括置換・広範囲の編集のあとは、`py_compile`だけで済ませない。** import smokeテスト（全モジュールのimport）と差分確認まで行う。
3. **新しい検証・不変条件は、意図的に壊して失敗することを確認する（変異テスト）。** 通ったテストは「検出できる」ことの証明にならない
   （実際に、変異の挿入位置を誤り、何も壊さないまま「通った」ことがあった）。
4. **新しいフラグ**: 既定OFF（挙動不変）、「OFF同値」と「ON時の性質」のテスト、コードからの参照（定義だけの空フラグは`test_config_health`が検知）。
   既定ONでパラメータ/バッファを追加するなら`_xai_common._OFF_IF_ABSENT_FROM_SNAPSHOT`に登録（旧checkpoint互換）。
   **新しい設定を`getattr(config,'X',既定)`で読まない**（タイプミスが黙って無視される）。`config.py`に定義して`config.X`で読む。
5. **「無い/読めない→警告で続行」を書かない。** `utils/logging.py::fallback_or_raise`を使う（`STRICT_FALLBACKS=True`で止まる）。
6. **日付範囲のSQLは必ず`db/queries.py::valid_date_sql()`をANDする**（RPAD文字列比較は不正形式を範囲に入れる）。
7. **共有DBの`split_type_wf`は可変な状態**: train/validを読む処理は`assign_wf_splits`（`assign_fold_test_window`はtest専用で他を-1にする）。
   割当は`split_state`テーブルに記録され、不一致なら`StaleSplitError`で止まる。
8. **指標・特徴量・ラベルを変えるときは、手計算できる期待値のテストを足す。** `tests/golden/numerics.json`は意図した変更のときだけ
   `UPDATE_GOLDEN=1`で更新し、差分をレビューしてからコミットする。
9. **学習・評価の実行中は、`transformer/`のコードを書き換えない**（関数内の遅延importで挙動が変わる）。テストと文書のみ。
10. **結果に使うrunは、コミット済みのコードで起動し、そのコミットにタグを付ける**（`git tag run-<日付>-<名前>`。`provenance.json`のcommit hashと`git describe`が再現の鍵。未追跡ファイルは`code_changes.patch`に入らない）。
11. **数値を報告・記録するときは、出典（runのディレクトリ、fold、フラグ、サンプル数n）を併記し、旧版・旧設定のrunと混ぜない。**
12. **コミットは論理単位で分ける**（機能・修正／検証スクリプト／テスト／計画メモ）。CLAUDE.mdの既定値表は`test_config_health`が検証するので、
    既定値を変えたらこのファイルも更新する。

---

## バージョン（重要）

**正典パッケージは `transformer`**（固定名）。修正・実装はここに入れる。
2026-10-08に `transformer_261008` から日付なしの固定名へ改名した（日付付きディレクトリのコピー運用を廃止）。
**版は日付付きディレクトリではなく、gitのタグと`provenance.json`で管理する**:
- 実験の区切り（結果を出す学習の前）に、そのコミットへタグを付ける: `git tag run-<日付>-<名前>`（例 `run-20261008-baseline`）。
- run出力の`provenance.json`に commit hash・`git describe`（直近のタグ）・dirty・環境が残る。結果は「どのタグのコードか」で特定する。
- 旧コードを開く: `git worktree add ../gmp_old <タグ>`。**新しい日付付きディレクトリ（`transformer_YYMMDD`）を作らない**（`tests/test_package_name.py`が検知）。
- 旧版 `transformer_260817` / `transformer_260723` / `transformer_260707` / `transformer_260702` / `transformer_260625` は
  `old_file/` 配下に退避済み（git管理外）で「前実装をすぐ参照する」ための残置として触らない（履歴はgitに残っている）。
DB は特徴量設定ハッシュで命名されるため（`db/features_<hash>.duckdb`）、特徴量構成が同じなら
旧版（260702〜261008）と`transformer`は同一 DB を共有し、再前処理は不要。

以下の「261008」「260817」等の見出しは、**当時の版名での変更履歴**（現在の`transformer`の履歴でもある）。

261008 で 260817 から入れた変更（2026-10-08、現在の `transformer`）:
- ディレクトリ名変更（`transformer_260817` → `transformer_261008`、のち固定名 `transformer` へ。`petra/`・`tests/`の参照も追随）。
  旧260817は`old_file/transformer_260817`へ退避。以降の「260817で追加」記述は260817時点の変更履歴。
- **不正な`collection_date`（`'2022/2024'`217件）をsplit割当から除外**: `db/queries.py`の`valid_date_sql()`
  （`YYYY`/`YYYY-MM`/`YYYY-MM-DD`のみ有効）を、日付範囲で割当・抽出する全SQL（`assign_wf_splits`・`assign_date_splits`・
  `assign_fold_test_window`・`attention_by_month`・一部分析スクリプト）にANDした。fold4のtestとfold5のtrainから
  ちょうど217件が除外（他foldは不変、約0.04%）。DBデータは未変更・再前処理不要。**日付範囲のSQLを新規に書くときは
  必ず`valid_date_sql()`をANDすること**（RPAD文字列比較は不正形式を範囲に入れてしまう）。詳細は docs/changelog.md。
- **黙って失敗する経路の対策**: 空split（train等が0件）は`EmptySplitError`/`RuntimeError`で止まる。
  run出力に`provenance.json`（git commit・dirty・環境・DB fingerprint）と`code_changes.patch`を保存
  （**未追跡ファイルはpatchに入らないので実行前にコミット**）。`walk_forward`は起動時に`pytest`＋DBプリフライトを
  自動実行し失敗なら中断（`--skip_gate`で省略）。主貢献の性質（共起スロット順への不変性）をテスト化。詳細は docs/changelog.md。
- **テストの拡充（B項目）**: 報告される指標（`evaluate`/`evaluate_topk`/集計関数）を手計算の期待値で、特徴量・前処理を
  極小ゲノムで、全モジュールのimport、設定の健全性（未定義名・未配線フラグ・CLAUDE.md既定値表との一致・既定値変更時の
  旧checkpoint互換）、数値回帰（`tests/golden/numerics.json`、更新は`UPDATE_GOLDEN=1`）を追加。
  テストで`PLOT_TOP_N_LINEAGES`の未定義、`USE_SUBSTITUTION_HEAD`の旧checkpoint互換漏れ、`mutation_freq.py`のimport副作用を発見・修正。
  **同一コドン内の共起変異の特徴量は記載順に依存する**（モデルの共起集約は順序不変）点は仕様として記録。
- **運用面の対策（C項目）**: 共有DBの割当状態を`split_state`に記録し、test専用割当のままtrain読込・別foldの割当のままの学習を
  `StaleSplitError`で止める。`STRICT_FALLBACKS=True`（既定）で、「リソース無し→警告で続行→結果が静かに劣化」する箇所
  （事前学習ckpt・ホモプラシーCSV・**point-in-time頻度表**・kNN・config_snapshot・前処理の入力）を`FallbackError`で止める。
  CLAUDE.mdを追跡対象にし、「AI（Claude）が変更を入れるときのルール」を追記。テスト223件。

260817 で 260723 から入れた変更（2026-08-17）:
- ディレクトリ名変更のみ（コード内容・機能は260723と完全に同一、diffで確認済み）。
  `petra/` 側の `transformer_260707` 参照も `transformer_260817` に追随済み
  （コミット「バージョン更新」）。

260723 で 260707 から入れた変更（2026-07-23）:
- **Co-occurrence Attentionへの頻度ペナルティ**（`USE_COATTN_FREQUENCY_PENALTY`, 既定 `False`）:
  大きな共起グループ（オミクロン系統等）で、ほぼ全系統に共通する創始変異（nsp12:P323L等）に
  Attentionが機械的に収束する現象への対策。`HOMOPLASY_CSV`のホモプラシー再発回数が高い変異ほど
  Cross-Attentionスコアへ学習可能スケール付きの負バイアスを加算する（`USE_HOMOPLASY_PRIOR`と
  対称的な設計）。既定Offのため無効時の挙動は260707と完全に同一（回帰確認済み）。
  発見の経緯は `plot_timetree_model.py`（系統樹への学習信号オーバーレイ）→
  `scripts/analysis/verify/verify_coattn_frequency_confound.py`（頻度交絡検証）→
  `verify_position_embedding_norm_confound.py`（embeddingノルム仮説の検証・否定）を参照。
- 上記に伴い `attention_by_month.py::_attach_attention_capture` /
  `feature_importance.py::collect_coattn_weights` のmonkey-patchラッパを `**kwargs` 対応に拡張
  （`freq_penalty`引数の素通し、既定Off時は無関係）。

260723 で入れた変更（2026-07-29）:
- **walk_forward の R-Precision 評価フェーズでの OOM 修正**: `run_final_evaluation` が
  val/test_loader を使い終えた後も同じ（学習開始以来生存し続ける）ローダーで
  `evaluate_topk` を回していたのを `run_topk_evaluation` として分離し、明示的に破棄・
  `EVAL_NUM_DATALOADER_WORKERS`（既定 `NUM_DATALOADER_WORKERS` の半分）採用の新規ローダーで
  作り直すよう変更。fold_3 単独再実行で修正前に OOM していたフェーズの完走を確認済み。
- **`save_config_copy()` がランタイムのconfig上書きを反映しない不具合を修正**: 静的
  `config.py` の `shutil.copy2` から、実行時属性値をシリアライズする方式に変更。
  `_wf_patch_config()` 等による `USE_POINT_IN_TIME_FREQ` 等の上書きが `config_snapshot.py`
  に正しく反映されるようになった。
- **walk-forward分析スクリプトでの `USE_POINT_IN_TIME_FREQ` 揮発によるリーク再混入を修正**:
  上記バグにより `feature_importance.py` 等の事後分析が静的 `FREQ_CSV`（未来情報混入）に
  揮発していた。`_xai_common.assign_fold_test_window()` に `USE_POINT_IN_TIME_FREQ=True` の
  強制を追加し、`feature_importance.py` に `fold_N` 自動検出によるこの関数の呼び出しを追加
  （transformer_260707側にも同修正を適用しfeature_importance再実行）。
  `codon_freq` 表示名を実体（塩基位置×置換パターン単位の変異再発頻度）に即した
  `mutation_recurrence_freq` に変更。詳細は [docs/changelog.md](docs/changelog.md) 参照。
- **Broadcast-back Cross-Attention追加**（`USE_BROADCAST_BACK_ATTENTION`, 既定 `False`）:
  Co-occurrence Attentionでの集約により失われる「個々の変異×他タイムステップ変異」間の
  個別粒度Attentionを補うため、集約前の変異embeddingが時系列Encoder適用後の代表ベクトル列へ
  Cross-Attentionし、再集約した結果を学習可能ゲート（0初期化）付き残差として加算する
  （`model.py:BroadcastBackAttention`）。既定Offのため無効時の挙動は完全に同一。
  詳細は [docs/features.md](docs/features.md) 参照。

260723 で入れた変更（2026-08-03）:
- **巨大共起グループ（Omicron期）による学習中OOMの修正**: Omicron系統で祖先枝1本に
  大量サンプルがぶら下がる巨大グループ（fold_3 test最大14,684件実測）が、
  (1) `_build_group_label_cache`のCopy-on-Write崩壊（persistent worker長時間生存で
  1 workerあたり約6GBまで肥大化）と (2) R-Precision動的Kの瞬間的巨大確保
  （`torch.topk(..., k=need_k)`がバッチ全体に波及）の2経路でOOMを引き起こしていた。
  `MAX_GROUP_MEMBERS_FOR_CACHE=4000`（超過グループは決定的ランダムサブサンプリングで
  間引き）・`TRAIN_NUM_DATALOADER_WORKERS`（学習用workerを半減）・`MAX_R_PRECISION_K=4000`
  を新設して対処。学習中のメモリ使用量ロギング・グループサイズ分布の可視化スクリプト
  （`group_label_count_histogram.py`等）も追加。**fold_3の評価指標への影響は未検証**
  （今後の課題）。詳細は [docs/changelog.md](docs/changelog.md) 参照。

260817 で追加調査・対処した変更（2026-10-05〜10-08、コードは261008へ引き継ぎ）:
- **`gc.freeze()` によるCOW崩壊の根本対策**（`db/dataset.py:create_db_dataloader()`）:
  上記のCOW崩壊は、巨大キャッシュを「量で削る」対策（`MAX_GROUP_MEMBERS_FOR_CACHE`等）
  だけでは、CPython循環GCがキャッシュを定期走査して複製を誘発する「頻度」自体は止まら
  ない。`_group_label_cache`構築直後・worker fork前に `gc.freeze()` を呼び、構築済みの
  全オブジェクトを永続世代へ移して以降のGC走査対象から除外することで、複製が起きる
  経路自体を断つ。train/valid/test全ローダー経路（epoch毎の再構築含む）を通る
  `create_db_dataloader()` 1箇所に実装。既存のcap策とは併用可能で相反しない。
  **実際のwalk_forward実行での効果検証（`_log_rss`によるRSS比較）は未実施**（今後の課題）。
- **`MultiTaskLoss`の学習済みタスク重みをログ出力**（`main.py:run_training`）: `USE_MULTITASK_LOSS=True`
  では `LOSS_WEIGHT_*`（Position=0.7等の手動優先度）は参照されず、Kendall式自動重みのみが
  有効だが、`get_weights()`が未呼び出しで実際の配分が不明だった。各epoch末に
  `[MultiTaskLoss weights]`を出力し、`training_log.csv`に`weight_<task>`列、W&Bに`loss_weight/*`を記録。
  次回学習で配分が手動優先度と乖離していないか確認すること。
- **`USE_HIERARCHICAL_PREDICTION` を既定 `True` に変更**（再学習不要・評価時のみ）:
  `USE_REGION_CONDITIONED_POSITION=True`で学習したfold_3 checkpointでも追加効果を確認
  （position_hit_rate 5.1026→5.4709%、+0.368pt、region不変。`hierarchical_prediction_check.py`）。
  現行config下での全fold平均は未確認（当該学習がfold_3のみ）。
  `USE_HOMOPLASY_PRIOR`は後付け簡易チェック（`homoplasy_prior_posthoc_check.py`、scale=1.0固定）で
  fold_3 position_hit_rate 4.0084%（基準5.4709%、−1.46pt）と**方向性は負**（未学習スケールでの参考値）。
  `USE_KNN_OUTPUT`: 配線完了（`evaluate()`、`scripts/eval/build_knn_datastore.py`、疎CSR形式、`faiss-cpu`導入済み。
  旧`build_datastore`はSoft Targetで空になる問題があり修正）。**パイロット（fold_3、datastore 30万群、test10万群、k=16）で
  λ=0.1/0.25/0.5のposition_hit_rateは5.373/5.242/4.682%とλ=0(5.401%)より単調に悪化。現設定では不採用**
  （`evaluate_topk`には未配線のまま）。※train分割を読む分析は`assign_fold_test_window`(test専用)ではなく
  `assign_wf_splits`を使うこと（前者はtest以外を-1にするためtrainが空になる）。
- **旧checkpoint読込の不具合を修正**: `USE_REGION_CONDITIONED_POSITION`が既定Trueになった後、当該フラグが
  無い時代のcheckpointを`_xai_common.load_config_snapshot`経由で読むとstate_dictのキー不足で失敗した。
  スナップショットに無ければFalseに戻す（`tests/test_load_config_snapshot.py`）。
- **テスト導入**（2026-10-06）: `tests/`（pytest、DB/GPU不要、`pytest.ini`で`pythonpath=.`・`slow`既定除外）。
  グルーピング・cap・Soft Target・MultiTaskLoss・fold checkpoint解決・config snapshot・
  ホモプラシーバイアス・階層マスキング・`gc.freeze()`配線・kNN・旧checkpoint互換（層A）に加え、
  合成DuckDBでのsplit割当・DataLoader E2E・1エポック学習スモーク・checkpoint往復（層B）、
  実DB向けプリフライト検査`scripts/inspect/preflight_check.py`と低速テスト（層C）。223件（カバレッジ約30%、evaluate.py 76%）、
  約60種の変異テストで検出力を確認済み。実DBに`collection_date='2022/2024'`が217件残るが割当から除外済み（preflightはwarn）。
  テスト容易化のため挙動不変で切り出し: `main.py:_wf_resolve_prev_checkpoint`、
  `evaluate.py:build_allowed_position_mask`。計画・経緯はgit履歴（コミット00732b6の`PLAN_pipeline_self_verification.md`）を参照。
- **`feature_importance.py`に特徴量×recency(timestep)の同時マップを追加**:
  `feature_by_recency_importance.csv` / `feature_by_recency_heatmap.png`（既存出力は不変）。

260707 で 260702 から入れた変更（2026-07-07）:
- 損失計算のベクトル化（Soft Target の per-sample ループ廃止）・評価の GPU→CPU 転送集約（高速化）
- collate_fn の `DBBatch` NamedTuple 化 / 評価集計の共通ヘルパ化（保守性）
- MLM の span masking（`PRETRAINING_MASK_MODE='span'`）
- 高確信サブセット Recall の出力（`SAVE_CONFIDENT_SUBSET`, `{prefix}_confident_subset.csv`）

---

## 主要コマンド

```bash
# 0. UShER 出力 → TSV 変換（初回のみ・preprocess より先に実行）
python -m transformer.scripts.preprocess.data_format

# 1. DuckDB 前処理（初回 or DB再構築時）
nice -n 19 python -m transformer.preprocess

# 2. 学習・評価
nohup python -m transformer.main > nohup0.out 2>&1 &

# 3. バックグラウンド進捗確認
tail -f nohup0.out

# 事前学習
python -m transformer.scripts.train.pretrain

# 推論のみ（学習済みチェックポイントから評価再実行）
python -m transformer.scripts.eval.evaluate_only \
    --checkpoint outputs/transformer/results/<timestamp>/models/best_model.pth

# 半年次 Walk-forward 検証（全 7 フォールド）
# config.py で SPLIT_MODE='walk_forward' にしておくこと
nohup python -m transformer.scripts.eval.walk_forward > nohup_wf.out 2>&1 &
# 特定フォールドのみ
python -m transformer.scripts.eval.walk_forward --folds 2 6

# walk_forward 全fold横断プロット一括実行（全fold学習完了後に実行すること）
python -m transformer.scripts.analysis.walk_forward.run_all_walk_forward_plots \
    --walk_forward_dir outputs/transformer/results/walk_forward/<timestamp>

# 特徴量重要度分析（学習済みモデル）
python -m transformer.scripts.analysis.feature_importance \
    --checkpoint outputs/transformer/results/<timestamp>/models/best_model.pth

# DB・データ確認
python -m transformer.scripts.inspect.inspect_duckdb     # DB スキーマ確認
python -m transformer.scripts.inspect.view_one_sample    # サンプル内容確認

# テスト（DB/GPU不要・数秒。コード変更後、長時間学習の起動前に実行する）
python -m pytest -q                      # 既定は slow マーカーを除外（223件、約13秒）
python -m pytest -m slow                 # 実DB・読み取り専用（約10秒。DBが他プロセスに開かれていればskip）
# 起動前のプリフライト検査（実DB、読み取り専用、約10秒。--fold Nは割当済みのsplitと日付規則を突合）
python -m transformer.scripts.inspect.preflight_check --output stage_verification.json

# 集計・分析
python -m transformer.scripts.analysis.aggregate_strains    # 株別集計
python -m transformer.scripts.analysis.aggregate_lineages   # 系統別集計
python -m transformer.scripts.analysis.aggregate_variants   # 月別変異株集計
```

出力先: `outputs/transformer/results/<EXPERIMENT_NAME>/<timestamp>/`
（`EXPERIMENT_NAME = ''` のときは `results/<timestamp>/` に直接保存）
スクリプト出力: `outputs/transformer/scripts/`

---

## ディレクトリ構成

```
gmp/
├── transformer/       # メインパッケージ
│   ├── config.py             # 全設定フラグ（ここを変えて実験する）
│   ├── model.py              # HierarchicalTransformer / MultiTaskLoss
│   ├── main.py               # エントリポイント（学習→評価→保存）
│   ├── train.py              # train_one_epoch()
│   ├── evaluate.py           # evaluate() / evaluate_topk()
│   ├── preprocess.py         # DuckDB構築・特徴量生成
│   ├── db/                   # DuckDB接続・クエリ・Dataset・特徴量計算
│   │   ├── connection.py
│   │   ├── dataset.py
│   │   ├── queries.py
│   │   └── feature.py        # 変異パス読み込み・特徴量計算（preprocess.py が使用）
│   ├── utils/
│   │   ├── losses.py         # build_loss_fn() (cbce / focal / ce 等)
│   │   ├── plotting.py       # 全プロット関数
│   │   ├── io.py             # CSV・JSON・モデル保存
│   │   ├── logging.py        # force_print / W&B init
│   │   ├── bio_smooth.py     # Biologically Informed Loss
│   │   └── codon_freq.py     # コドン頻度計算
│   └── scripts/              # ユーティリティ・分析スクリプト（サブパッケージ構成）
│       ├── preprocess/               # データ準備
│       │   ├── data_format.py        # UShER→TSV 変換（preprocess.py より先に実行）
│       │   ├── generate_host_adaptation_features.py
│       │   ├── generate_homoplasy_prior.py       # ホモプラシー再発回数CSV生成
│       │   └── generate_point_in_time_freq.py    # point-in-time頻度CSV生成（リーク対策）
│       ├── train/                    # 学習
│       │   └── pretrain.py           # MLM/CLM 事前学習
│       ├── eval/                     # 評価・推論
│       │   ├── evaluate_only.py      # 学習済みモデルで評価のみ実行
│       │   ├── walk_forward.py       # 半年次7fold walk-forward（FOLDS定義もここ）
│       │   ├── build_knn_datastore.py  # kNN用データストア構築（要checkpoint。--max_groupsで無作為抽出）
│       │   ├── multi_seed.py         # 複数seed実行（mean±std）
│       │   ├── optuna_search.py      # Optunaハイパラ探索
│       │   └── petra_recall.py       # PETRA式Recall@K（representativeness重み）
│       ├── inspect/                  # DB確認・デバッグ
│       │   ├── preflight_check.py    # 起動前プリフライト検査（切り詰め・日付不正・重複・欠落）→stage_verification.json
│       │   ├── inspect_duckdb.py     # DB スキーマ・先頭データ表示
│       │   ├── view_one_sample.py    # 1サンプルの特徴量を人間可読形式で表示
│       │   └── test_dataloader_filter.py  # dataloader フィルタ効果検証
│       ├── analysis/                 # 集計・分析
│       │   ├── aggregate_variants.py       # 月別・変異株別出現数集計
│       │   ├── aggregate_lineages.py       # 系統別サンプル数集計
│       │   ├── aggregate_strains.py        # DuckDB 株別サンプル数集計
│       │   ├── mutation_freq.py            # 変異頻度・ヒートマップ集計
│       │   ├── analyze_lineage_difficulty.py   # 系統別予測難易度分析
│       │   ├── analyze_timestep_by_lineage.py  # タイムステップ×系統分析
│       │   ├── feature_importance.py       # Gradient-based 特徴量重要度分析
│       │   ├── permutation_importance.py   # Permutation feature importance
│       │   ├── majority_baseline.py        # 一様ランダム・多数決ベースライン
│       │   ├── verify/                # 一回性の検証・診断スクリプト（読み取り専用）
│       │   │   ├── verify_point_in_time_freq.py       # point-in-time頻度リーク検証
│       │   │   ├── branching_cooccurrence_frequency.py # raw_pathの分岐(K)・共起(M)頻度測定
│       │   │   └── verify_batch_local_grouping.py     # 分岐グルーピングのバッチ分断検証（修正済みバグの回帰記録）
│       │   ├── walk_forward/          # walk_forward全fold横断のプロット群（run_all_walk_forward_plots.py で一括実行）
│       │   │   ├── run_all_walk_forward_plots.py   # 対象8本をwalk_forward_dir1つに対し一括実行
│       │   │   ├── position_tolerance_monthly.py   # 許容誤差付き位置予測Top-1評価（月別・全fold、要checkpoint/DB）
│       │   │   ├── plot_position_tolerance_comparison.py  # 上記とPETRA側の月別比較プロット（PETRA側CSV要・一括対象外）
│       │   │   ├── entropy_accuracy_analysis.py    # 系統別エントロピー vs 精度、Fano理論上限
│       │   │   ├── overlay_diversity_folds.py      # 系統別entropy/unique_ratio/simpson vs 精度、全fold重畳
│       │   │   ├── overlay_nll_folds.py            # タスク別held-out NLLのfold横断トレンド
│       │   │   ├── overlay_region_folds.py         # 遺伝子別recallのfold横断トレンド
│       │   │   ├── overlay_strength_folds.py       # 流行度カテゴリ別精度のfold横断トレンド
│       │   │   ├── plot_position_tolerance_all_tol.py  # 許容誤差0/5/10/50を1枚に
│       │   │   ├── predicted_position_by_month.py  # 月別の予測位置分布
│       │   │   ├── entropy_fano_by_month.py        # 月別エントロピー・Fano上限
│       │   │   ├── fano_gap_multivariate.py        # Fanoギャップの多変量分析
│       │   │   ├── fano_gap_sample_size_curve.py   # Fanoギャップ vs サンプル数
│       │   │   ├── lineage_accuracy_drivers.py     # 系統別精度の要因分析
│       │   │   ├── group_label_count_histogram.py  # 共起グループ内ユニークラベル数分布（OOM対策の根拠）
│       │   │   ├── hierarchical_prediction_check.py  # 階層的予測(Regionマスキング)の効果検証（再学習不要）
│       │   │   ├── homoplasy_prior_posthoc_check.py  # USE_HOMOPLASY_PRIORの後付け簡易チェック
│       │   │   ├── overlay_confident_subset_folds.py  # 確信度ベースcoverage曲線のfold横断重ね描き
│       │   │   └── monthly_diversity_accuracy.py   # 月別精度×多様度3種の色分け＋無色版、全fold接続（要checkpoint/DB）
│       │   └── xai/                  # XAI・生物学的帰属分析（run_all_xai.py で一括実行）
│       │       ├── run_all_xai.py            # ①②③④⑤ を1 checkpointに対し一括実行
│       │       ├── run_all_xai_walk_forward.py  # 上記をwalk_forward全foldに対し実行・fold横断集計
│       │       ├── _xai_common.py            # fold探索・checkpoint読込等の共通ヘルパ（walk_forward/ 配下からも利用）
│       │       ├── genome_track.py           # ①位置/遺伝子別重要度→ゲノム地図
│       │       ├── homoplasy_alignment.py    # ②重要度 vs homoplasy相関＋負コントロール
│       │       ├── cooccurrence_constellation.py  # ③共予測ペア vs 実共起の復元検証
│       │       ├── local_explain.py          # ④Integrated Gradientsによる局所説明
│       │       ├── probe_representation.py   # ⑤内部表現の線形プロービング
│       │       ├── timestep_attention.py     # ⑥timestep方向Attention重み分析
│       │       ├── timestep_importance.py    # ⑤'タイムステップ別予測寄与度（IG）
│       │       ├── mutation_hotspot_report.py    # ⑦遺伝子別・位置別変異蓄積傾向レポート
│       │       ├── attention_by_month.py     # Co-occurrence Attention注目度の月別プロット
│       │       ├── lineage_attention_compare.py  # 系統別 ground truth vs Attention比較
│       │       ├── lineage_evolution_track.py    # 系統代表サンプルの変異獲得タイムライン
│       │       └── plot_xai_outputs.py       # 上記XAI出力群の可視化
│       └── visualization/            # 可視化（XAI以外の汎用プロット）
│           ├── _fold_annotate.py                   # fold境界・時代注釈の共通ヘルパ
│           ├── plot_data_diversity_by_month.py     # 月別データ多様度
│           ├── plot_group_count_by_month.py        # 月別の共起グループ数
│           ├── plot_target_position_by_month_normalized.py  # 月×NucPos（正規化版）
│           ├── plot_collection_date_comparison.py  # 収集日分布の比較プロット
│           ├── plot_cooccurrence_distribution.py   # 共起変異数分布プロット
│           ├── plot_lineage_timeline.py            # 系統タイムライン
│           ├── plot_target_position_by_month.py    # 月×NucPos サンプル数ヒートマップ
│           ├── plot_timestep_by_month.py           # タイムステップ月別プロット
│           └── plot_timetree.py                    # 系統樹プロット
├── petra/                     # 先行研究PETRA比較用の軽量再実装パッケージ
│   ├── config.py / dataset.py / model.py / tokenizer.py / train.py / main.py
│   └── eval/                  # 評価・walk-forward・月別プロット
│       ├── evaluate.py            # Recall@K（per-sequence macro-average + Weighted）
│       ├── weights.py             # representativeness重み（Weighted Recall用）
│       ├── eval_tail_by_daterange.py  # tail-onlyRecall@Kを日付範囲・leaked/non_leaked別に集計
│       ├── walk_forward.py        # 本体と同一フォールドでのwalk-forward学習
│       ├── plot_monthly_hitrate.py            # 月別tail-only Hit Rateプロット
│       └── plot_monthly_position_tolerance.py # 月別・許容誤差付き位置Hit Rateプロット
├── tests/                    # pytest（conftest.py: cfg/_restore_config/synthetic_db。real_data/は実DB・slow）
├── pytest.ini                # pythonpath=. ／ slowマーカーは既定除外
├── PLAN_accuracy_improvement.md       # 精度向上の検証キュー
├── PLAN_thesis_schedule.md            # 修論までのスケジュールと報告内容
├── PAPER_outline.md                   # 修論の構成・論点・図表リスト
├── reference/                # 参照データ（CSV・FASTA等）
│   ├── aa_properties/        # アミノ酸特性（PAM250・dissimilarity）
│   ├── codon/                # コドン頻度・変異テーブル
│   ├── genome/               # 参照ゲノム（NC_045512.2.fasta・VCF）
│   └── phylogeny/            # 系統樹（.pb ファイル、git管理外）
├── scratch/                  # 旧スクリプト置き場（scripts/ に移行済み）
├── db/                       # DuckDB ファイル（大容量・git管理外）
├── cache/                    # 前処理キャッシュ（git管理外）
├── outputs/                  # 学習結果（git管理外）
│   ├── transformer/
│   │   ├── results/          # main.py の実験出力
│   │   │   └── <EXPERIMENT_NAME>/<timestamp>/
│   │   └── scripts/          # スクリプト出力
│   └── archive/              # 旧バージョンの実験結果
└── wandb/                    # W&B ログ（git管理外）
```

---

## アーキテクチャ

```
Input → Embedding (Base/Pos/AA/Region/CodonPos/Synonymous/Context±5)   # CONTEXT_WINDOW=5
      → Co-occurrence Attention   # 共起変異を集合として集約（順序非依存）
      → [Causal Conv1d]           # USE_LOCAL_CONV1D=True のとき
      → [Origin Attention]        # USE_ORIGIN_ATTENTION=True のとき（武漢株参照）
      → Transformer Encoder (N_LAYERS=4, N_HEADS=4, FEATURE_DIM=256)
      → Multi-task Heads:
          Region (37) / NucPos (~30K) / AAPos (~10K) / CodonPos (3) / Synonymous (2) / Strength (回帰)
          + base_after / aa_after（USE_SUBSTITUTION_HEAD=True）
      ※ USE_REGION_CONDITIONED_POSITION=True: Positionロジットに region_head の予測確率
         (position→region写像経由のlog確率×学習可能スケール) を加算（学習時のRegion条件付け）
      ※ 評価時 USE_HIERARCHICAL_PREDICTION=True: 予測Region上位3個に属さない位置を除外（再学習不要）
```

---

## 主要設定フラグ (config.py)

### 実験切り替え
| フラグ | デフォルト | 説明 |
|---|---|---|
| `HYBRID_ALPHA` | `1.0` | 1.0=Soft Target, 0.0=Hard Target |
| `STRENGTH_SOURCE` | `'usher'` | `'ncbi'` / `'usher'` で流行度ソース切り替え |
| `SPLIT_MODE` | `'walk_forward'` | `'timestep'` / `'date'` / `'walk_forward'` でデータ分割方式切り替え |
| `USE_REGION_CONDITIONED_POSITION` | `True` | 学習時のRegion条件付きPosition予測（fold_3のみ検証済み） |
| `USE_HIERARCHICAL_PREDICTION` | `True` | 評価時のRegionマスキング（fold_3で+0.368pt、全foldは検証中） |
| `USE_MULTITASK_LOSS` | `True` | Kendall自動重み。`LOSS_WEIGHT_*`は無視される（重みはepoch毎にログ出力） |
| `USE_LOCAL_CONV1D` | `False` | Conv1d局所特徴抽出の有効化 |
| `USE_ORIGIN_ATTENTION` | `False` | 武漢株参照 Cross-Attention |
| `TEMPORAL_POOLING` | `'last'` | `'last'` / `'mean'` / `'cls'` |
| `LOSS_FUNCTION_TYPE` | `'ce'` | `'ce'` / `'cbce'` / `'focal'` / `'cb_focal'` |

### アブレーション
`ABLATION_MASKS` 内の各フラグを `True` にすると、その特徴量を0マスクして学習できる。
再前処理不要で動的適用される。

### データ量
| フラグ | デフォルト | 説明 |
|---|---|---|
| `MAX_STRAIN_NUM` | `10000` | 使用する上位株数 |
| `SAMPLING_MODE` | `'proportional'` | `'proportional'` / `'fixed_per_strain'` |
| `MAX_NUM` | `10000000` | 全体サンプル数上限 |

---

## データパイプライン

1. **UShER** で `public-2024-10-17.all.masked.pb` を系統解析 → `clades.txt`
2. `preprocess.py` が変異パスを抽出し特徴量を付与して DuckDB へ書き込む
3. `db/dataset.py` がバッチ単位でストリーム読み込みして学習に渡す

DuckDBファイル名はハッシュ付き (`features_<hash>.duckdb`)。
`config.DB_FILE = None` のとき設定ハッシュから自動決定される。

---

## 評価指標

- **Hit Rate** — Top-K 予測に正解共起変異が1つでも含まれる割合
- **Weighted Recall@K / Macro Recall@K** — 遺伝子位置のクラス別リコール
- **Strength Category 別精度** — Low / Medium / High 流行度カテゴリごとの精度
- **ECE** (期待キャリブレーション誤差) — `SAVE_ECE=True` のとき計算

---

## 比較実験モデル構成

| モデル | 共起の扱い | 特徴量 | 用途 |
|---|---|---|---|
| ① PETra相当 | 直列入力 | なし | Baseline |
| ② 中間 | 直列入力 | 物理化学あり | 特徴量効果の純粋検証 |
| ③ 提案 | Co-occurrence Attention | 物理化学あり | アーキテクチャ効果の検証 |

---

## 注意事項

- `db/`, `cache/`, `outputs/`, `wandb/` は合計400GB超のため git 管理外
- 学習は長時間かかるため `nohup` + `nice -n 19` を推奨
- `config.FORCE_REPROCESS = True` にすると DB を強制再構築する
- W&B はオフラインモード (`WANDB_OFFLINE=True`) がデフォルト
