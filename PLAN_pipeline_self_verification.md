# パイプライン検証・テストプラン（新規実装のバグ検出）

初版: 2026-07-19 / **改訂: 2026-10-06 / 完了: 2026-10-08** / 対象バージョン: `transformer_260817`（正典）

## 状態: ✅完了（2026-10-08）

| フェーズ | 内容 | 状態 |
|---|---|---|
| 1 層A 単体テスト | A1〜A12 | ✅ |
| 2 層B 合成DuckDB | B1〜B5 | ✅ |
| 3 層C・実行時チェック | プリフライト検査＋stage_verification.json／実DB低速テスト／（RSS閾値監視は見送り） | ✅（見送りあり、下記） |

実行: `python -m pytest -q`（既定・約5秒・**96件パス＋xfail1件**）、`python -m pytest -m slow`（実DB・読み取り専用・約10秒）、
プリフライト: `python -m transformer_260817.scripts.inspect.preflight_check [--fold N]`。
検出力は計16種の変異テストで確認済み（いずれも該当テストが失敗）。

## 目的（改訂）

新しい機能・修正を入れたとき、**入れた変更が壊していないこと／意図通り動くこと**を
コマンド1つで確認できるテストを用意する。「バグが一切ないことの証明」ではなく、
過去に実際に起きた種類のバグを、再発前に機械的に検知することを狙う。

## 背景: 過去の「静かなバグ」と、テストで捕まえられたか

いずれもクラッシュせず、もっともらしい値を出し続けた。

| バグ | 発見 | 単体テストで捕捉可能か | 必要なテストの種類 |
|---|---|---|---|
| `raw_path`切り詰め（先頭500文字） | 2026-07-10 | △ 切り詰め有無の検査はできるが、データ依存 | 実DB検査（後述 層C） |
| 分岐グルーピングがバッチ内に分断（`7c87286`） | 2026-07-18 | ○ 小さな入力で「同一履歴が別バッチでも1グループ」を検証できる | 合成データの単体テスト |
| fold間checkpoint引き継ぎ（`idx==0` vs `fold_id==1`、`8d3301f`） | 2026-07-18 | ○ 引数の組合せをテーブル化して検証できる | 単体テスト |
| `save_config_copy`が実行時overrideを反映しない | 2026-07-29 | ○ configを書き換えて保存→再読込で一致を検証 | 単体テスト |
| OOM（COW崩壊・R-Precision動的K） | 2026-07-28〜 | ✕ 規模依存。ただし「cap適用が決定的・代表が残る」等の性質は検証可 | 性質テスト＋実行時RSS監視 |
| 学習は動くがタスク重みが偏っていても気付けない（`get_weights()`未呼び出し） | 2026-10-05 | ✕ 観測の問題 | ログ（実装済み） |

結論: **「入力が小さく、期待値が手で書ける」純関数・小規模ロジックは単体テストで十分に
捕まえられる。** データ分布・実行フロー依存のものだけ実DB検査と実行時チェックに回す。

## 3層構成（改訂）

| 層 | 内容 | 実行時間 | 実行タイミング |
|---|---|---|---|
| **A. 単体テスト（pytest・DB/GPU不要）** | 合成入力＋手計算の期待値。純関数と、`__new__`で`__init__`を迂回できるメソッド | 数秒 | コード変更のたび（常時） |
| **B. 結合テスト（極小の合成DuckDB）** | メモリ上のDuckDBに数十行の`samples`/`labels`を作り、split割当・グルーピング・DataLoader・数ステップの学習を通す | 数十秒 | 機能追加のたび・walk_forward起動前 |
| **C. 実データ検査（既存`verify/`の格上げ）** | 実DBで分断率・切り詰め率・リーク等を集計しassert | 数分〜数十分 | 長時間学習の起動前に手動（任意） |

旧プランの「②埋め込み型ランタイムassert（`stage_verification.json`）」は、A/Bが整った後の
**フェーズ3**に後ろ倒しする（本体コードへの変更を伴い、まずテスト資産を作る方が費用対効果が高い）。

## テスト対象の一覧（フェーズ1: 層A）

配置: リポジトリ直下 `tests/`（`conftest.py`で合成データ・config差し替えfixtureを共有）。
実行: `conda activate gvp25-05 && pytest tests/ -q`。

| # | 対象 | 検証する性質（例） | 関連バグ／変更 |
|---|---|---|---|
| A1 | `DBIterableDataset._build_groups` | 同一`input_path_str`は**全入力順序・全shuffleで**同一グループ。代表=`min(sample_id)`。分断率0 | 分岐グルーピング分断 |
| A2 | `_capped_member_ids` | cap以下は恒等／超過時は長さ=cap・**代表が必ず残る**・同じ`repr_id`で**何度呼んでも同一結果**（決定的）・`None`で無効 | OOM対策（260723） |
| A3 | `_build_soft_target` | 各タスクの確率和=1・`(1/K)(1/M_k)`の分配・重複変異の加算・範囲外idは無視・T≠1で再正規化・K=0で零ベクトル | Soft Target |
| A4 | `MultiTaskLoss` | 初期重み=1（`log_vars=0`）・`get_weights()`が`exp(-log_vars)`と一致・勾配が`log_vars`へ流れる | 重みログ追加（2026-10-05） |
| A5 | fold checkpoint解決（`run_walk_forward`の分岐ロジックを関数に切り出して検証） | `fold_id==1`→None／連続→直前best／非連続→override優先→自動探索→見つからなければ`RuntimeError`。`--folds 2`単独・`--folds 2 6`で誤って事前学習に落ちない | fold間引き継ぎ |
| A6 | `_wf_find_prev_fold_checkpoint` | 複数候補から最終更新が最新のものを返す／無ければ`None`（tmpディレクトリで検証） | 同上 |
| A7 | `save_config_copy`（実行時overrideの反映） | `config.X`を書き換え→保存→`config_snapshot`再読込で値が一致、シリアライズ不可な値は除外 | 2026-07-29修正 |
| A8 | Hierarchical masking（`evaluate.py`の該当部を純関数に抽出して検証） | 予測Region上位R個**外**の位置は`-inf`・全位置が除外される行はマスク解除（NaN回避）・`region`の予測自体は不変 | `USE_HIERARCHICAL_PREDICTION`既定ON（2026-10-05） |
| A9 | Homoplasyバイアス（`_load_homoplasy_bias`） | `log1p(rec)`・範囲外`position_id`は無視・CSV欠損時は`None`で無効化 | `USE_HOMOPLASY_PRIOR` |
| A10 | `gc.freeze()`の配線 | `create_db_dataloader`呼出し後に`gc.get_freeze_count()>0`（配線の回帰防止） | 2026-10-05 |

### フェーズ1の進捗（2026-10-06）

実行: `conda activate gvp25-05 && python -m pytest -q`（リポジトリ直下。`pytest.ini`で`pythonpath=.`、`slow`マーカーは既定で除外）。

| # | 状態 | ファイル | 備考 |
|---|---|---|---|
| A1 | ✅ | `tests/test_grouping.py` | 順序非依存・代表=min・全サンプルがちょうど1グループ。単一ステップ履歴は`''`グループに集まる現行仕様を記録 |
| A2 | ✅ | `tests/test_group_cap.py` | 恒等／None無効／代表が先頭に残る／決定的 |
| A3 | ✅ | `tests/test_soft_target.py` | 和=1・(1/K)(1/M)・重複加算・範囲外無視・温度 |
| A4 | ✅ | `tests/test_multitask_loss.py` | 初期重み=1・式・勾配 |
| A5 | ✅ | `tests/test_walkforward_resolve.py` | `_wf_resolve_prev_checkpoint`として挙動不変で切り出し済み |
| A6 | ✅ | `tests/test_walkforward_checkpoint.py` | |
| A7 | ✅ | `tests/test_config_snapshot.py` | |
| A8 | ✅ | `tests/test_hierarchical_mask.py` | `build_allowed_position_mask`として切り出し済み。元のインライン実装との同値テストを含む |
| A9 | ✅ | `tests/test_homoplasy_bias.py` | |
| A10 | ✅ | `tests/test_gc_freeze_wiring.py` | |
| A11 | ✅ | `tests/test_knn_output.py` | kNN（Soft/Hard対応・faiss/torch一致・マスク尊重・保存読込）13件。追加機能 |
| A12 | ✅ | `tests/test_load_config_snapshot.py` | 旧checkpointのスナップショット互換（下記バグの回帰防止） |

**テストで見つかった実害のあるバグ（2026-10-06）**: `USE_REGION_CONDITIONED_POSITION`が既定Trueになった後、
このフラグが無い時代のcheckpoint（例 `20260803_115144`）を`_xai_common.load_model_and_loader`で読むと
`Missing key(s): region_hier_scale, pos_region_map`で失敗していた（階層的予測の全fold評価・特徴量重要度が
全foldで失敗して発覚）。`load_config_snapshot`が、スナップショットに無い当該フラグをFalseに戻すよう修正。

**検出力の確認（変異テスト）**: 代表をmin→max、`gc.freeze()`削除、cap時に代表を落とす、fold1でprevを返す、
raiseしない、全除外行のフォールバック削除、上位R個の数、kNNのマスク無視／近傍平均／λ取り違え／Soft無視、
の計11変異で、いずれも該当テストが失敗することを確認（ソースは都度復元、現在64件パス）。

**共通ルール（新機能を入れるときの必須2点）**: 本プロジェクトは「既定Offで挙動不変」を
原則としているため、新フラグごとに次の2本を必ず追加する。
1. **OFF同値テスト**: フラグOFFのとき、出力が変更前と**数値的に同一**であること。
2. **性質テスト**: フラグONで期待する性質（上表のような不変条件）が成り立つこと。

## フェーズ2: 層B（極小の合成DuckDB）— ✅完了

`tests/conftest.py`の`synthetic_db`（`init_db`と同じスキーマに8サンプル・3グループ＋testを投入）と、
全テスト後に`config`を自動復元する`_restore_config`（`setattr`直書きによる汚染防止）を共通化。

| # | 状態 | ファイル | 検証内容 |
|---|---|---|---|
| B1 | ✅ | `tests/test_assign_wf_splits.py`（9件） | 日付境界（半開区間）、RPADによる不完全日付、NULL/空日付（Fold1はtrain・Fold Nは除外）、train/test非重複、valid比率、前fold割当の上書き、**独立実装(`expected_wf_split`)との全サンプル一致**（3種のfold設定）。不正日付`'2022/2024'`の除外は**xfail(strict)**で既知問題として記録 |
| B2 | ✅ | `tests/test_preflight_checks.py`（10件） | train/test間の完全一致`raw_path`の件数・比率、split未割当時は「評価不能(warn)」と明示、不正日付・切り詰め・ラベル/特徴量欠落の検知 |
| B3 | ✅ | `tests/test_dataloader_e2e.py`（6件） | 全グループがちょうど1回出る、**num_workers=0/2で同一内容**、`group_targets_list`が全メンバーのラベルを含む、split間で混ざらない、左パディング・テンソル形状、Soft Target和=1 |
| B4 | ✅ | `tests/test_training_smoke.py`（6件） | 1エポックで損失が有限・パラメータが更新・NaNなし、**全パラメータに勾配が流れる**（既定で未使用の`cls_token`のみ許容）、`USE_REGION_CONDITIONED_POSITION`/`USE_HOMOPLASY_PRIOR`ON時の追加パラメータに勾配、`cls`プーリング時の`cls_token`、MultiTaskLoss重みの更新 |
| B5 | ✅ | `tests/test_checkpoint_roundtrip.py`（2件） | `save_config_copy`→checkpoint→`load_model_and_loader`で出力が一致（Noneヘッドも一致）、読み戻し側のconfigを変えても学習時のフラグへ復元、snapshot経由のloaderが指定splitを読む |

## フェーズ3: 層C・実行時チェック — ✅完了（一部見送り）

| 項目 | 状態 | 内容 |
|---|---|---|
| プリフライト検査 | ✅ | `scripts/inspect/preflight_check.py`: raw_path切り詰め／collection_date不正形式／ラベル・特徴量欠落／train-test完全一致raw_path／（`--fold N`でsplit割当と日付規則の突合）。読み取り専用・実DBで約10秒。各チェックは pass/warn/fail |
| `stage_verification.json` | ✅ | プリフライトの出力形式（`--output`）。failがあれば終了コード1。**main.py/walk_forwardへの自動組込みはしない**（本体への変更を避け、起動前の手動実行とする） |
| 実DB低速テスト | ✅ | `tests/real_data/test_real_db_preflight.py`（`@pytest.mark.slow`、`pytest -m slow`）。DBが他プロセスに開かれていれば自動skip |
| 既存`verify/`の整理 | ✅（方針） | `verify_*.py`は一回性の診断としてそのまま保持（CLAUDE.mdのツリーに記載済み）。常時チェックに値するものはプリフライトへ移した |
| RSS閾値の警告 | ⏸見送り | 既存の`[MEM]`ログ（`_log_rss`）で足りると判断。閾値は実測（`gc.freeze()`効果の確認後）に基づくべきで、現時点では根拠が無い |

**実DBのプリフライト結果（2026-10-08）**

| チェック | 結果 | 内容 |
|---|---|---|
| raw_path切り詰め | ✅pass | 904万行中、長さちょうど500は681行（0.0075%、自然な範囲）。`RAW_PATH_TRUNCATE_LEN=None` |
| ラベル・特徴量 | ✅pass | 欠落0 |
| collection_date | ❌**fail** | 不正形式**217件**（`'2022/2024'`）、NULL/空710件。**既知のデータ品質問題がDB上に残存**（fold4のtestに誤混入しうる）。DB割当側で除外する修正が未実施 |
| train/test重複 | ⚠️評価不能 | 直前の実行でtestのみ割当済み（trainが空）。`assign_wf_splits`後に再実行が必要 |

## 実装上の注意（調査で判明した点）

- 本体のテストは導入前は0本だった（`petra/PETra/`配下は同梱ライブラリのテスト）。現在は`tests/`に96件＋xfail1件。
- 多くのロジックは`DBIterableDataset.__init__`がDBアクセスを伴うため、メソッド単体を
  テストするには `DBIterableDataset.__new__(DBIterableDataset)` で`__init__`を迂回して必要な属性だけ
  設定するか、ロジックを純関数に切り出す。**テスト容易性のための小さな切り出しリファクタ
  （A5・A8）は挙動を変えないことをOFF同値テストで担保する。**
- `evaluate()`内のマスキング等、長い関数の途中にあるロジックはテスト困難。切り出すのが先。
- `config`はモジュールグローバルなので、fixtureで書き換え→**必ず元に戻す**（`monkeypatch.setattr`）。
- 実DB・実checkpointに依存するテストは`@pytest.mark.slow`で既定実行から除外する。

## 決定事項（ユーザー合意、2026-10-06〜）

1. 着手順: フェーズ1から（A1・A5・A2を優先）。→ 実施済み
2. リファクタ許容: A5・A8を挙動不変で関数に切り出す。→ 許容、実施済み（`_wf_resolve_prev_checkpoint`、`build_allowed_position_mask`）
3. 運用: 学習・`walk_forward`の起動前に手動で`python -m pytest -q`（必要なら`preflight_check`）を流す。自動フック化はしない
4. 層C: 今回のスコープに含めた（プリフライト＋低速テスト）。ランタイム自動組込みとRSS閾値は見送り

## テストで見つかった実害のある問題（総括）

| 問題 | 見つけた層 | 状態 |
|---|---|---|
| 旧checkpointが`USE_REGION_CONDITIONED_POSITION`の既定変更で読めない（キー不足） | A12（評価失敗で発覚） | ✅修正済み |
| `build_datastore`がSoft Targetで常に空 | A11 | ✅修正済み |
| `assign_fold_test_window`はtest以外を-1にするため、train分割を読む分析が空になる | 実行時に発覚 | ✅`build_knn_datastore.py`は修正済み。**他のtrainを読む分析は注意** |
| `collection_date='2022/2024'`が217件DBに残存、fold4のtestに誤混入しうる | B1(xfail)・層C | ⏳**未修正**（xfailで記録。DB割当側で除外すれば解消） |

## 旧プランからの変更点

- ゴールを「各ステージの自己診断ログ」から「新規実装のバグ検出テスト」へ再定義。
- 6ステージ分解は残すが、各ステージに**具体的なテスト項目**（A/B表）を対応づけた。
  ステージ5（fold引き継ぎ）は実装済みの`RuntimeError`ロジックを**テストで固定**する形に。
- ②ランタイムassertを後回しにし、①プリフライト相当を層B/Cに、③ユニットテストを最優先（層A）に。
- 対象バージョンを`transformer_260707`→`transformer_260817`に更新。

## 関連

- 修正履歴: [docs/changelog.md](docs/changelog.md)
- 精度向上の検証キュー: [PLAN_accuracy_improvement.md](PLAN_accuracy_improvement.md)
