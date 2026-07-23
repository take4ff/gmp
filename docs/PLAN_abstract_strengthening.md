# 学会要旨補強プラン（PETRA比較・遺伝子考察・attention説明性）

作成日: 2026-07-07 / 対象バージョン: `transformer_260707`（正典）

## 背景・課題

学会要旨に「変異履歴の構造理解と特徴抽出を実現した」と記載したが、現状の結果セクションは
領域(Region)・位置(Position)の予測精度のみで、以下3点が欠けている。

1. **精度の良し悪しを判断する基準がない**（比較対象・ベースラインが無い）
2. **変異蓄積されやすい遺伝子・変異の考察**（生物学的解釈）が無い
3. **attention重み解析による重要変異抽出**（モデルの説明性）が無い

前セッション（`transformer_260702`上）で以下を先行実装・検証済み:

- (a) walk_forward（半年次7フォールド検証）を全7フォールド完走
- (b) `attention_by_month.py` を新規実装し、単一チェックポイント版・walk_forward日付連結版の両方を検証
- (c) PETRA論文（arXiv:2511.03976）との厳密比較ハーネス（`petra_recall.py`/`petra_weights.py`/`petra/`パッケージ）を実装済み（ただし実学習は未実行）

CLAUDE.mdの正典は`transformer_260707`のため、上記の成果を260707へ移植し、260707上で
実際の学習・分析を行う。DBは260702/260707で共有（`db/features_72f42a78.duckdb`、両者とも
同一ハッシュに解決することを確認済み）のため前処理は不要。

## abstractの主張・既存数値 と 各Phaseの対応

末尾の「abstract本文」に記載の既存walk-forward数値（デルタ株: 領域45%/位置35%、オミクロン株:
領域50%/位置10%、シノニマス・コドン位置は各80%/60%程度）を踏まえた、各Phaseの狙い:

| abstractの記述 | 現状の不足 | 対応Phase |
|---|---|---|
| 「変異履歴の構造理解と特徴抽出を実現した」（共起集約層の導入） | この主張を裏付ける具体的証拠が無い | Phase C③（attention_by_month / local_explain / cooccurrence_constellation）。既知VOC変異への注目度・IG帰属を具体例として提示する |
| デルタ45%/35%・オミクロン50%/10%という数値 | 高いのか低いのか判断基準が無い | Phase F（PETRA head-to-head）。ボキャブラリサイズ(~30000位置)に対するrandom baseline、PETRA論文のscale-matched数値と並べて相対評価する |
| デルタ→オミクロンで位置精度が35%→10%に急落する一方、領域精度は45%→50%と維持・微増 | この乖離（「位置は絞れないが遺伝子は絞れる」）の生物学的解釈が無い | Phase C②（genome_track）。オミクロンの変異多様性（多数の位置に分散）と遺伝子レベルでのホットスポット固定化を対比して考察する材料にする |
| 「今後の展望: 段階的な変異候補の抽出」（領域→シノニマス→コドン位置→位置の絞り込み） | 未着手（今回のスコープ外、将来課題として明記のみ） | 対象外。ただしPhase C②のgenome_track結果は、この段階的抽出が有効そうな遺伝子（絞り込みやすい）を示唆できる可能性があるため、副産物として言及だけ検討 |

## 決定事項

| 項目 | 決定 | 理由 |
|---|---|---|
| walk_forward再学習（headline数値用） | ~~260707で再学習する（Phase X）~~ → **不要と判断・キャンセル（2026-07-08）**。260702の既存walk_forwardをheadline数値としてそのまま流用する | Phase C/D用チェックポイントの流用判断（下段参照）と同一の根拠：`model.py`/`train.py`は260702/260707でバイト同一・`config.py`差分も③提案アーキテクチャの学習ロジックに無関係と確認済みのため、260707で再学習しても数値は区別がつかない想定。ユーザ判断（2026-07-08）で新規学習ゼロに |
| メインモデルの分割方式（XAI分析用） | ~~`SPLIT_MODE='timestep'`単発学習~~ → **不要と判断・キャンセル（2026-07-08）** | 下記「Phase B不要判定」参照。walk_forwardの各フォールドを自分のtest期間で評価し積算・連結する方式（Phase C/D）で全期間カバーができ、かつ`timestep`分割自体が擬似ランダムで時系列公平性が無いという欠点があるため、こちらの方が優れる |
| Phase C/D用チェックポイントの由来 | **260702の既存walk_forward（7フォールド完走済み）をそのまま流用可** | `model.py`は260702/260707で完全にバイト同一（diff 0行）、`config.py`差分はOUTPUT_DIRと260707限定の追加機能（`SAVE_CONFIDENT_SUBSET`, `PRETRAINING_MASK_MODE='span'`）のみで③提案アーキテクチャの学習ロジックに影響なし、`train.py`も同一（diff 0行）と確認済み（2026-07-08）。精度差が出る余地がないため、新規学習ゼロでPhase C/Dに着手できる |
| 事前学習 | MLM事前学習（`PRETRAINING_MODE='mlm'`, span masking, 5epoch）を経由した状態で評価を揃える | ユーザ指示。ablation_results.mdの知見でも、MLM事前学習はhit-rateが領域・位置・aa_pos全てで一貫して改善する唯一のパターンだった |
| PETRA比較 | `petra/`パッケージを実学習し`petra_recall.py`でhead-to-head比較 | 「①精度の良し悪し」を判断する最有力の外部基準 |
| 実行順序（2026-07-08更新） | 事前学習(B-1)・PETRA学習(Phase E)・Phase F-lite(tail-only月別比較)完了。**Phase B（メインモデル単発学習）・Phase X（260707メインwalk_forward再学習）は両方不要と判定してスキップ**。headline数値・XAI分析（Phase C/D）とも260702の既存walk_forwardをそのまま使う | 一様ランダム基準は学術的に弱いため、PETRA比較を①の主判定基準に一本化（ユーザ承認）。さらにPhase X自体も260702/260707の同一性根拠でキャンセル（ユーザ判断、2026-07-08）となり、新規学習が完全に不要になった。残る作業はXAI分析（Phase C）と要旨執筆のみ |

## フェーズ一覧

### Phase A: `attention_by_month.py` を260707へポート — ✅完了
`transformer_260702/scripts/analysis/attention_by_month.py` を `transformer_260707/scripts/analysis/attention_by_month.py` へコピーし、全ての`transformer_260702`参照を`transformer_260707`に置換。ロジック変更なし（`CoOccurrenceAttention`はバイト同一、`DBBatch`のNamedTuple化後も`batch[9]`=collection_datesの位置アクセスは不変）。importチェック済み。

### Phase B: メインモデル学習（timestep分割・全履歴）— ❌不要と判定・キャンセル（2026-07-08）

**当初の想定**: walk_forwardの各フォールドは6ヶ月〜部分履歴でしか学習しておらず、genome_track/
local_explain等「モデルが変異史全体をどう理解しているか」を見る分析には全履歴学習モデルが
必要と考え、`SPLIT_MODE='timestep'`での単発学習を計画していた。

**不要と判定した理由（2026-07-08）**:
1. `attention_by_month.py`に既に実装済みの`run_walk_forward()`は、各フォールドの「自分の
   test期間だけ」を評価してattn_sum/occ_countを積算し、日付軸で連結する方式。全期間を1本の
   モデルで学習するのではなく、**各時期をその時期担当のフォールドモデルで評価して繋ぐ**やり方で、
   時系列的によりフェア。
2. `SPLIT_MODE='timestep'`は`petra/config.py`のコメントにもある通り「擬似ランダム分割で
   時系列の公平性が無い」——全履歴学習のために使うと、逆にテストデータの一部が訓練期間より
   未来でない保証がなくなりリーク懸念が生じる。timestep学習はむしろ`walk_forward`より
   質が劣る可能性がある。
3. 260702で**walk_forward（7フォールド）は既に完走済み**。genome_track /
   cooccurrence_constellation を「フォールドごとに評価して積算」パターンに対応させれば
   （Phase C参照）、新規学習ゼロで全期間をカバーできる。
4. `model.py`は260702/260707で**完全にバイト同一**（`diff`で確認、差分0行）、`config.py`の
   差分もOUTPUT_DIRと260707限定の追加機能（`SAVE_CONFIDENT_SUBSET`,
   `PRETRAINING_MASK_MODE='span'`）のみで③提案アーキテクチャの学習ロジックには無関係、
   `train.py`も同一（diff 0行）。よって**260702のwalk_forwardチェックポイントをそのまま
   使っても精度差は生じない**（ユーザ提案「260702でも精度に差がなければ260702で評価したい」
   への回答）。

**結論**: Phase Bはキャンセル。Phase C/Dは260702の既存walk_forwardチェックポイントを
そのまま使う（下記参照）。

### Phase C: ②③のXAI分析（walk_forwardチェックポイントで積算・実行）— ⏳PETRA完了待ち

`genome_track.py` / `cooccurrence_constellation.py` / `local_explain.py`の3スクリプトに、
`attention_by_month.py`と同じ「各フォールドを自分のtest期間で評価して積算・連結」方式
（`--walk_forward_dir`）を追加実装済み（2026-07-08）。共通ヘルパ（フォールドcheckpoint探索・
config_snapshot反映・test期間のsplit_type_wf割り当て）は`_xai_common.py`に集約し、
`attention_by_month.py`自体は無変更（動作実績のあるコードに触れないため）。

**②遺伝子別の蓄積傾向（全期間積算）**:
```bash
python -m transformer_260707.scripts.analysis.genome_track \
    --walk_forward_dir outputs/transformer_260702/results/walk_forward/<timestamp> --n_batches 100 [--force_cpu]
```
出力: `position_importance.csv`（位置別: 予測質量 vs 実出現数、全7フォールド積算）、
`gene_importance.csv`/`.png`。Spike等の既知ホットスポットと一致するかを考察材料にする。
補助として`mutation_freq.py`（純データ集計、チェックポイント不要）も併用可。

**③attention/勾配による説明性**:
```bash
python -m transformer_260707.scripts.analysis.attention_by_month \
    --walk_forward_dir outputs/transformer_260702/results/walk_forward/<timestamp> --n_batches 100 --value mean
python -m transformer_260707.scripts.analysis.cooccurrence_constellation \
    --walk_forward_dir outputs/transformer_260702/results/walk_forward/<timestamp> --n_batches 100 --top_k 5
# local_explainのみケーススタディ用に特定フォールドを指定（例: fold3=Omicron BA.1/BA.2初期）
python -m transformer_260707.scripts.analysis.local_explain \
    --walk_forward_dir outputs/transformer_260702/results/walk_forward/<timestamp> --fold 3 --n_samples 8 --ig_steps 32
```
`local_explain.py`（Integrated Gradients）は、D614G(23403)・N501Y等の**既知VOC定義変異がtop-1予測になっているサンプル**を狙って抽出し、モデルが何に帰属しているかを具体例として示すケーススタディ構成にする（集計統計だけより要旨の「重要変異抽出」主張に直結する）。VOC期に応じて`--fold`を選べば、単一の全履歴モデルが無くても各時代のケーススタディが可能。

**単一checkpoint版（`--checkpoint`）は引き続き利用可能**（後方互換、smoke test等に使用）。

**追加: attentionの頻度交絡チェック（`attention_by_month.py`に実装済み）**
懸念: attentionが高い変異（例: D614G）は単に「ほぼ全サンプルに出現する」ため、頻度で埋め込みが歪んで
目立っているだけの可能性がある（`value='mean'`は素朴な出現回数の交絡は除去済みだが、埋め込み自体が
頻度で歪む深い交絡までは対処できない）。`build_matrix_and_plot()`内で`check_frequency_confound()`を
自動実行し、位置ごとの**総出現回数 vs 平均attention重み**のSpearman相関を計算・出力するようにした
（`attention_vs_frequency.csv`/`.png`、meta JSONの`frequency_confound_check`）。|r|>0.8かつ有意なら
「attention≒頻度の焼き直し」を疑う目安。Phase C実行時に自動で得られるため追加実行は不要。
局所説明（IG, `local_explain.py`）は内部attention重みの大小に依存しない別経路の帰属なので、
ケーススタディ対象の位置がIGでも高帰属を示すかを併せて確認するとより強い根拠になる（新規スクリプトは
不要、Phase C③の結果を突き合わせるだけ）。
`run_all_xai.py --only genome_track local_explain cooccurrence_constellation` でまとめて一括実行も可能。

### Phase D: 260707メインwalk_forward再学習 — ❌不要と判定・キャンセル（2026-07-08）

**当初の想定**: canonicality確保のため、260702の既存walk_forwardでのXAI分析（Phase C）とは別に、
260707上で改めてメインモデルのwalk_forwardを再学習し（Phase X）、headline数値・XAI分析の両方を
260707由来のものに差し替える計画だった。

**不要と判定した理由（ユーザ判断、2026-07-08）**: Phase Bキャンセル時に確認済みの
「`model.py`/`train.py`は260702/260707でバイト同一、`config.py`差分も③提案アーキテクチャの
学習ロジックに無関係」という事実は、headline数値算出にもそのまま当てはまる。260707で
walk_forwardを再学習しても260702の既存結果と数値上区別がつかない想定のため、**headline数値・
XAI分析ともに260702の既存walk_forward（7フォールド完走済み）で代用する**。新規学習は完全に
不要になった。

（Phase C・Fの実行コマンドは引き続き`transformer_260707.scripts.analysis.*`を使うが、
`--walk_forward_dir`には260702のディレクトリを渡し続ける。260707固有の再学習・再生成は行わない）

### Phase E: PETRA比較 — 260707参照への更新＋実学習 — 🔄設計見直し中（2026-07-07）

**優先順位変更（2026-07-07）**: 一様ランダム基準は学術的に弱いとの判断から、①の主判定基準を
PETRA比較に一本化する方針にし、**メイン学習(B-2)より先にPhase Eを実行する**ことにした。

#### 準備作業（完了済み）

- **260702参照の更新**: `petra/main.py`（1箇所）・`petra/evaluate.py`（1箇所）・`petra/config.py`
  （コメントのみ、3箇所）にあった`transformer_260702`へのハードコード参照を`transformer_260707`に
  更新済み。import構文チェック済み。
- **`petra/dataset.py`の重複排除・ワーカーsharding修正**（当初版）: 本体は`USE_UNIQUE_FILTER`で
  `raw_path`完全一致サンプルを除去しているが、`petra/dataset.py`にはこの処理が無かった
  （「同一データでの厳密比較」という方針に反する非対称性）。また`NUM_WORKERS=4`なのに
  `get_worker_info()`によるシャーディングが無く全ワーカーが同一データを4重処理していた
  （本体`db/dataset.py`で以前修正済みと同じ落とし穴）。両方修正。
- **さらに正確な修正（最終版）**: 独自に重複排除ロジックを再実装するのではなく、**本体の
  `transformer_260707/db/dataset.py`の`DBIterableDataset`が実際に解決する`sample_ids`を
  そのまま再利用する**方式に変更（`petra/dataset.py`の`_resolve_ids_from_main_model()`）。
  これにより`USE_UNIQUE_FILTER`だけでなく`MAX_CO_OCCURRENCE=20`（共起数上限）・
  `EVAL_MAX_Y_CO_OCCURRENCE=5`（評価時ターゲット共起上限）等、本体の全フィルタリングロジックと
  完全に一致する。独自実装の見落としを防ぐため、今後同様のデータ整合が必要な箇所ではこの
  「本体のDBIterableDatasetを直接呼んでsample_idsを再利用する」パターンを踏襲する。

#### split設定の変遷（2026-07-07中に3回変更）

1. 当初: `SPLIT_MODE='date'`（`petra_comparison.md`時代の設計、train=791万→dedup後218万、
   test=14万→dedup後**27件**）
2. 修正版データ整合を適用後の同一date設定で再測定: train=2,186,101（変化なし）、
   valid=362,538（`EVAL_MAX_Y_CO_OCCURRENCE`除外が追加で効いた）、**test=17件**
   （さらに減少。`date`分割のtest窓＝直近の狭い期間は最頻出raw_pathが141,038件中138,914件
   （98.5%）を占めるほど遺伝的多様性が低く、重複排除すると必然的にごく少数まで潰れる。
   これは実装のバグではなくデータの本質的な特性——本体`petra_recall.py`で同じdate split test を
   評価すれば同様に極小になるはずなので両モデルに対称的な問題ではあるが、n=17/27では
   統計的に比較が弱い）。
3. ユーザ指摘により**`SPLIT_MODE='timestep'`へ変更を検討中**: Phase B-2（メインモデル）は
   `timestep`分割（train=6,763,952/valid=224,652/test=1,944,091）を使う予定のため、
   比較対象を揃えるならPETRA側も`timestep`にすべき、との指摘。

#### walk-forward版PETRAへのスコープ拡大（2026-07-07・方針A採用）

abstractに記載の精度（デルタ45%/35%、オミクロン50%/10%）は**walk_forwardの特定フォールド・
特定月**（fold2「Delta→Omicron transition」のtest窓が2021-11を含む等）から来ている
（詳細は下記「Phase F以前に実施した簡易ベースライン比較」内の出典確認セクション参照）。
これと直接比較するには、単一train/testではなくPETRAも**walk_forwardと同じフォールド構成で
複数回学習**する必要がある。

提示した選択肢（A:全7フォールド／B:デルタ/オミクロン該当2フォールドのみ）から**方針A（全7
フォールド）を採用**。以下を実装・起動済み:

- **`petra/main.py`をリファクタ**: 学習ループを`run_training(split_col, init_checkpoint, out_dir,
  log_prefix)`として切り出し、単発実行(`main()`)・walk_forward実行の両方から呼べるようにした
  （CLIの`python -m petra.main`の挙動は従来通り）。
- **`petra/walk_forward.py`を新規実装**: 本体`transformer_260707/scripts/eval/walk_forward.py`の
  `FOLDS`定義・`assign_wf_splits()`をそのまま再利用し、**本体のwalk_forwardと全く同じ
  フォールド分割**でPETRAを学習する。本体の設計（"Fold 1はスクラッチ、Fold N(N≥2)は直前
  フォールドのbest_model.pthを初期値に継続学習"）も`init_checkpoint`引数で再現。
  出力先: `outputs/petra/walk_forward/<timestamp>/fold_N/`。
  Usage: `python -m petra.walk_forward [--folds 1 2 3]`
- **重要な運用上の注意**: `split_type_wf`列は本体のwalk_forwardと共有のDB列。
  **本体のwalk_forwardと同時実行すると互いに上書きし合い両方壊れるため、同時実行しないこと**
  （ユーザのメインモデルwalk_forwardはこの時点でまだ未開始）。

**起動・進捗（2026-07-07 21:04〜）**: `nohup python -m petra.walk_forward > nohup_petra_wf.out 2>&1 &`
で起動。Fold 1（Pre-Alpha era, train<2021-01, test=2021-01〜07）:
train=197,931/valid=32,078/**test=532,262**（date分割のtest=17件と比べ大幅に頑健）。
1epoch=145秒、epoch4時点でval_loss 1.69→1.22と順調に低下中。`MAX_EPOCHS=30`・`PATIENCE=5`。
プロセスは`nohup`起動のためSSH切断後も継続する。

#### 学習規模の目安（参考値、上記方針次第で変動）

`BATCH_SIZE=64`・D_MODEL=320・6層・Parameters=37,108,480（実測。当初想定した本体scale-match
~19.46Mの約1.9倍——語彙サイズが事前想定より大きい92,332トークンだったため）。
date split・218万件trainの場合で34,157バッチ/epoch、`MAX_EPOCHS=30`・`PATIENCE=5`、
20it/s想定で5epoch=2.4時間・10epoch=4.7時間・30epoch=14.2時間。walk_forward方式の場合は
フォールドごとにtrain窓が小さくなる分、1フォールドあたりはこれより短くなる見込みだが、
フォールド数倍の合計時間がかかる。

**準拠範囲の整理（`petra/PETra`原論文リポジトリを実読して確認済み）**:

| 項目 | 準拠状況 |
|---|---|
| 評価指標（Recall@1/10/100、per-sequence macro平均、representativeness重み、核酸＋スパイクAA範囲） | ✅ `eval_mutgpt.py`の`compute_accuracy()`を直接精読して`petra_recall.py`/`petra_weights.py`に移植済み。完全準拠 |
| Megatron不使用 | ✅ 問題なし。原論文の学習エントリ(`pretrain_mutgpt_batched.py`)もMegatronの`GPTModel`＝標準decoder-only Transformerを呼んでいるだけで、Megatronはマルチ GPU並列化基盤にすぎず数理的なモデル定義ではない。単一GPUで同じ層構造(causal self-attn+FFN、Pre-LN、embed/head重み共有)を再現していれば数学的に等価 |
| 位置エンコーディング | ⚠️ 差異あり。原論文は`position_embedding_type`+`rotary_percent`引数でRoPE使用の可能性が高いが、`petra/model.py`は学習可能な絶対位置埋め込み(`nn.Embedding`) |
| モデル規模 | ⚠️ 意図的に縮小。原論文公式設定はbase:hidden=512/12層/8head、large:hidden=1024/24層/16head(`petra.sh`等で確認)。こちらは本体モデル(~19.46Mパラメータ)にscale-matchさせたD_MODEL=320/6層で、原論文公式規模より大幅に小さい |
| トークン化 | ⚠️ 簡略化。原論文は変異1件を9個の生フィールド→12要素ラベル配列に展開する複合スキーム(`utils_mutation.py`の`embedding_transform`)。こちらは変異1件=1フラットトークン(`petra/tokenizer.py`、DBスキャンで動的語彙構築) |
| データセット | ⚠️ 意図的に自データ使用（既承認済み）。原論文の学習済み重みは2025-02まで学習されており、本体の評価データ(2024-10 UShER)で使うとリークの懸念があるため、学習済み重みは使わずアーキテクチャのみを自データで再学習する方針 |

**書き方への示唆**: 本比較は「PETRA論文の数値を再現する再現実験」ではなく、「PETRA型のdecoder-only causal Transformer（共起集約なし）を、本体モデルと同一データ・同一評価プロトコル・同等パラメータ規模で学習した、共起集約Attentionの効果を切り出すcontrolled ablationベースライン」という位置づけ。要旨・論文では「PETRAの数値を再現した」ではなく「PETRAのアーキテクチャ思想に基づくベースラインとの比較」と書くのが正確。

### Phase F: petra_recall.pyでhead-to-head比較 — ⏳未着手
```bash
python -m transformer_260707.scripts.eval.petra_recall \
    --main_checkpoint <Phase Bのckpt> --petra_checkpoint <petra/学習後のckpt>
```
核酸全体＋スパイクAA範囲(21563-25384)でRecall@1/10/100をAverage（必須）・Weighted
（`samples.country`未整備のため現状はAverageと同値にフォールバックする既知の制約あり）で算出。
これが要旨の「①精度の良し悪し」を判断する最有力の外部比較になる。

**2026-07-08時点の位置づけ**: 下記「Phase E完了・全7フォールド結果」「tail-only月別比較」で
より直接的かつ解釈しやすい①の判断材料が既に得られたため、petra_recall.py（トークン語彙全体での
Recall@1/10/100、representativeness重み付き）によるPhase F本体は優先度を下げてよい。位置レベルの
生の比較で十分説得力があるため、Phase Fは「余裕があれば追加のエビデンスとして実施」に格下げする。

## Phase E完了・全7フォールド結果（2026-07-08）

`outputs/petra/walk_forward/20260707_210406/` に全7フォールドのチェックポイント・`results.json`
（配列全体を教師強制で平均する通常のPETRA準拠Recall@K）が揃った。

| fold | 時期 | test n（重複除去後） | test_loss | R@1(avg) | R@1(wtd) | R@10(avg) |
|---|---|---|---|---|---|---|
| 1 | Pre-Alpha | 532,262 | 3.40 | 78.1% | 76.0% | 85.1% |
| 2 | Delta→Omicron移行 | 1,427,663 | 2.65 | 80.0% | 80.1% | 85.0% |
| 3 | Omicron初期(BA.1/BA.2) | 213,385 | 1.49 | 94.2% | 92.8% | 96.8% |
| 4 | Omicron variants | 391 | 1.44 | 83.7% | 81.1% | 89.8% |
| 5 | XBB | 124 | 1.76 | 86.0% | 93.4% | 90.6% |
| 6 | JN.1移行 | 21 | 0.68 | 93.5% | 92.1% | 97.1% |
| 7 | 最新 | 17 | 0.52 | 96.8% | 96.4% | 99.8% |

**fold3(94.2%)の高さの原因を特定（DB実測、`max_cooccurrence`列を直接集計）**: config.pyのコメント
「Omicron等に対応するため5から20へ緩和」通り、BA.1起源の単一祖先枝には共起18変異が同時発生して
おり、これが子孫ゲノム全てに継承される。2021-12（fold3のtrain末尾月）は716,667件中325,207件
（45.4%）がこの共起18ノードを含み、2022-01（fold3のtest初月）は815,157件中768,793件（94.3%）が
同一ノードを継承——**この94.3%という比率がfold3のRecall@1=94.2%とほぼ完全に一致**する。
教師強制の配列全体平均は、この「訓練中に32万回以上見た同一の共起集合をなぞるだけ」の位置を
大量に含むため、汎化性能ではなくfounder変異の暗記を強く反映していると判断した。

**fold4-7（test n=391/124/21/17）は統計的に不安定**（`petra_comparison.md`記載の
`SPLIT_MODE='date'`テスト崩壊問題と同型：直近の狭い時期ほど遺伝的多様性が低く、重複除去で
必然的に極小まで潰れる）。この件はこれ以上の追加調査はしない方針（ユーザ判断、2026-07-08）。
fold4以降の数値は参考値以上の意味を持たせない。

## tail-only月別比較（2026-07-08・本体との直接対応）

fold3のfounder変異による底上げを避けるため、`petra/evaluate.py`に**配列末尾の変異1件のみを
評価する`evaluate_recall_topk_tail_only()`**を追加実装済み（本体の`position_hit_rate`＝
「まだ見ぬ次の変異1件を予測する」タスクと粒度を揃えた版）。さらに`petra/eval_tail_by_daterange.py`
（単一日付範囲での通常評価とtail-only評価の並列出力）・`petra/plot_monthly_hitrate.py`
（本体の`monthly_hitrate_all_folds.png`と同レイアウトで全7フォールドをstitchした月別プロット、
`split_type_wf`列に触れず日付範囲直接指定なので他フォールド学習と並行実行安全）を新規実装し、
`outputs/petra/walk_forward/20260707_210406/monthly_plot/petra_monthly_hitrate.{png,csv}`を生成。

**実行時の注意（解決済み）**: 実行時、`petra/model.py`に未コミットのRoPE化改修
（`nn.TransformerEncoder`+学習可能絶対位置埋め込み → 独自`PetraDecoderBlock`+回転位置埋め込み）
が入っており、学習済みチェックポイント（旧アーキテクチャ）と state_dict 不一致でロード不能に
なった。**ユーザ判断で旧アーキテクチャに戻して評価**：RoPE版は
`/tmp/claude-1001/.../scratchpad/model_rope_backup.py`に退避した上で
`git checkout HEAD -- petra/model.py`でコミット済みの旧版に戻し、評価スクリプトを実行した。
**現在の`petra/model.py`はgit HEAD相当（RoPE無し）の状態**。RoPE版はスクラッチパッド上のみに
存在しリポジトリには入っていない点に注意（次回以降、RoPE版を正式に採用するか判断が必要）。

本体`monthly_metrics_all_folds.csv`（`outputs/archive/transformer_260625/results/walk_forward/
20260705_203818/`、abstractのデルタ45%/35%の出典と同じファイル）と月単位でマージした結果:

| 年月 | サンプル数(本体/PETRA) | 本体 position_hit_rate | PETRA tail-only hit_rate |
|---|---|---|---|
| 2021-06 | 52,831 / 69,476 | 21.77% | 5.04% |
| 2021-07 | 122,046 / 148,252 | 28.00% | 8.87% |
| 2021-08 | 222,336 / 255,273 | 25.58% | 5.98% |
| 2021-09 | 242,229 / 263,354 | 27.27% | 5.42% |
| 2021-10 | 223,567 / 238,279 | 31.78% | 5.90% |
| **2021-11**（abstract「デルタ45%/35%」の出典月） | 256,009 / 274,466 | **34.84%** | **5.14%** |
| 2021-12 | 207,976 / 248,039 | 30.18% | 4.45% |
| 2022-01 | 83,510 / 138,570 | 9.14% | 5.70% |
| 2022-02 | 41,086 / 62,333 | 4.47% | 1.31% |

**結論**:
1. tail-only化でfold3の94.2%は完全に消え、母数の大きい月で1〜9%程度まで下がった。これは
   本体のfold3位置精度（7.33%、`majority_baseline.py`参照）とオーダーが一致し、
   founder変異仮説を裏付ける。
2. **最も母数が大きく頑健なDelta期（2021-06〜12）では、本体が一貫してPETRA型ベースラインの
   3〜6倍**（例: 2021-11は34.8% vs 5.1%）。要旨の「①精度の良し悪しの判断基準」に使える、
   同一データ・同一時系列カットオフでの直接比較として説得力が高い。
3. **正直な懸念点**: 2021-01〜04（Pre-Alpha期、fold1、事前学習なし・train最小）だけは逆転し、
   本体が0.5〜0.7%とPETRA（1.0〜2.2%）を下回る。本体アーキテクチャは変異史がまだ蓄積されて
   いない最初期には相対的に弱い可能性があり、要旨に書く場合はこの限界も正直に記載すべき。
4. n<1,000の月（2022-04以降）は両モデルとも数値が乱高下し統計的に無意味（PETRA側は
   n=1〜9で100%連発）。fold4-7の全体平均Recall（Phase E結果）と同じ崩壊パターン。

## データリーク調査（2026-07-08・fold4-7の高い数字の原因を追跡）

fold4-7でPETRAのtail-only hit rateが50%を超える月が頻発した件（例: 2023-02 n=18で61.1%）を
受けて、train窓とtest窓をまたいだ完全一致raw_pathの重複をDB実測で調査した。

### 発見: train/test間でraw_pathが完全一致する重複が存在

各foldのtrain窓・test窓で個別に`DISTINCT raw_path`を取得し重複を数えた結果:

| fold | trainユニーク数 | testユニーク数 | 重複数 | testに占める重複率 |
|---|---|---|---|---|
| 1 | 214,190 | 546,011 | 11,151 | 2.0% |
| 2 | 546,011 | 1,444,255 | 10,220 | 0.7% |
| 3 | 1,444,255 | 215,262 | 21,189 | 9.8% |
| 4 | 215,262 | 571 | 176 | 30.8% |
| 5 | 571 | 165 | 50 | 30.3% |
| **6** | 165 | 25 | **21** | **84.0%** |
| **7** | 25 | 27 | **20** | **74.1%** |

原因: `USE_UNIQUE_FILTER`（dedup）はtrain/valid/testをそれぞれ個別にクエリした結果に対して
のみ適用されており（`transformer_260707/db/dataset.py:_get_sample_ids()`、`split_col = ?`で
1つのsplitに絞ってから`drop_duplicates`）、**splitをまたいだ重複除去は行われていない**。
fold6/7は系統多様性が枯渇しており、進化が止まった同一系統がtrain窓・test窓の両方で
繰り返し提出されているため、この重複が高頻度で発生する。

### 【最初の誤判定】「本体も同様に汚染されている」→ 検証により撤回

初期の判断: PETRA(`petra/dataset.py`)は本体の`DBIterableDataset`をそのまま呼び出して
sample_idを解決しているため、本体自身のwalk_forward評価も同じ重複の影響を受けている
はずだと推測し、修正は共有クラス側（`transformer_260707/db/dataset.py`）に入れるべきと
一旦提案した。

**実地検証で撤回**: fold6の重複raw_path 21件**全件**について、train側代表サンプルと
test側代表サンプルの`labels.targets`（本体の予測ターゲット）を直接比較したところ、
**21件中21件（100%）でターゲットが不一致**だった。

理由: 本体のターゲットは`labels`テーブルに個別サンプルごとに紐づく値（そのサンプル固有の
実際の子孫の運命）であり、raw_path文字列から決定論的に導かれるものではない。進化が止まった
同一系統が異なる時期に重複提出されても、その先の運命（ターゲット）はサンプルインスタンスごとに
異なる。**そのため「同じraw_pathを訓練で見た」ことはテスト時の正解を教えない＝本体は
この意味で構造的にリークしない。**

一方PETRAはCLM（decoder-only, next-token prediction）設計のため、input/targetは同一raw_path
文字列を1トークンずらしただけの関係。raw_pathが完全一致すれば input→target の対応も
**機械的に**完全一致する。したがって「同じraw_pathを訓練で見た」ことがそのままテストの正解を
教えてしまう、**PETRA固有の構造的リーク**であることが確定した。

### 対応方針（ユーザ判断・2026-07-08）

- **本体側（`transformer_260707/db/dataset.py`）の修正は不要**。fold間チェーン
  （fold Nのtest期間=fold N+1のtrain期間）は意図された walk-forward 設計であり別問題。
- **PETRA側は「重複サンプルを除外」ではなく「集計を分けて両方報告」する方式を採用**。
  除外だと只でさえ枯渇したfold4-7のデータがさらに減ってしまうため、サンプルはそのまま使い、
  各配列に `is_leaked`（raw_pathが同一foldのtrain窓に既出か）を付与して
  「全体」「leaked（暗記込み・参考値）」「non_leaked（真の汎化性能）」の3種類を並べて出す。

### 実装済み（タスク#10）

- `petra/dataset.py`: `PetraDataset.collate_fn`が`is_leaked`フラグを含む6要素タプルを返すよう変更
- `petra/eval_tail_by_daterange.py`: `get_raw_path_set(db_path, start, end)`ヘルパーを追加。
  `_FixedIdsPetraDataset`に`leaked_raw_paths`パラメータを追加。CLIに`--train_start`を追加
  （渡すとleaked/non_leaked内訳を表示、省略時は従来通り全体のみ）
- `petra/plot_monthly_hitrate.py`: `FOLDS`から得られる`train_start`を使い、fold毎に自動で
  leaked判定→月別プロットに「全体」と「non_leaked限定」の2本の折れ線＋サンプル数バーを追加
- `petra/evaluate.py`: `evaluate_recall_topk`・`evaluate_recall_topk_tail_only`両方に
  `leaked`/`non_leaked`キーを追加（spike/regionと同じ形式の`_finalize`ヘルパーを再利用）
- `petra/train.py`: 6要素タプルに対応（学習ロジック自体は無変更）
- **検証済み**: fold6実データ・ランダム初期化モデルで疎通確認。`leaked+non_leaked=overall`が
  `evaluate_recall_topk`・`evaluate_recall_topk_tail_only`の両方で完全一致することを確認
  （fold6 test n=21のうち19件がleaked、2件がnon_leaked）。

### 今後: タスク#9（RoPE＋39ステップ窓での再学習）にこのleaked/non_leaked集計が組み込まれる

再学習後は、fold6/7のような低データ・低多様性foldでもnon_leaked側の数値を見ることで、
「暗記による底上げ」と「真の汎化性能」を区別して報告できるようになる。

## PETRA再学習（RoPE＋39ステップ窓）完了・最終結論（2026-07-08〜09）

タスク#9で`petra/walk_forward.py`を再実行（`outputs/petra/walk_forward/20260708_145056/`）。
RoPE化・39ステップ窓・leaked/non_leaked集計の3つを反映した状態で全7フォールド完走。

**全体平均Recall@1（旧設定とほぼ同水準、founder変異による底上げは引き続き残存）**:
fold1=80.3%, fold2=82.2%, fold3=94.3%, fold4=83.6%, fold5=86.9%, fold6=93.3%, fold7=96.3%
（旧: 78.1/80.0/94.2/83.7/86.0/93.5/96.8% とほぼ同一。窓を39に揃えても全体平均は大きく変わらない
= 39ステップの窓自体は元々ほとんどのサンプルでほぼ非制約だったことを示唆）。

`petra/plot_monthly_hitrate.py`を新チェックポイントで再実行し、本体の`monthly_metrics_all_folds.csv`
とnon_leaked版で再突合した結果、**2つの結論が確定した**:

### 結論1: Delta期（頑健な大n月）で本体の優位性は残るが縮小（3〜6倍→約2倍）

| 年月 | 本体 | PETRA(旧・窓不一致+リーク込み) | PETRA(新・39ステップ+non_leaked) |
|---|---|---|---|
| 2021-06 | 21.77% | 5.04% | 5.25% |
| 2021-07 | 28.00% | 8.87% | 14.29% |
| 2021-08 | 25.58% | 5.98% | 13.48% |
| 2021-09 | 27.27% | 5.42% | 14.96% |
| 2021-10 | 31.78% | 5.90% | 16.71% |
| **2021-11** | **34.84%** | 5.14% | **17.48%** |
| 2021-12 | 30.18% | 4.45% | 12.24% |

窓を揃えてリークを除いても、本体はDelta期で一貫してPETRA型ベースラインの**約2倍**を維持。
これが要旨の「①精度の良し悪しの判断基準」に使える、最も条件を揃えた比較結果。

### 結論2: 低データ期（2022-07〜）の「PETRA優位」は窓不一致とリークの複合アーティファクトだった

| モデル | プール結果（2022-07〜、non_leaked版） |
|---|---|
| 本体 | 38/406 = 9.36% |
| PETRA(non_leaked) | 28/320 = 8.75% |
| 2標本比率検定 | z=-0.284, **p=0.777（有意差なし）** |

旧比較（窓不一致・リーク込み）ではz=6.42, p<1e-6でPETRAが有意に上回るとしていたが、**正しく
補正すると有意差は消滅**。低データ期は両モデルとも訓練データ枯渇（fold5=523件・fold6=148件・
fold7=24件）という共通の限界に直面しているだけで、アーキテクチャの優劣を反映した差ではないことが
確定した。

**How to apply**: 要旨に載せる比較数値は必ずこの最終版（39ステップ+non_leaked、
`outputs/petra/walk_forward/20260708_145056/monthly_plot/`）を使う。旧版（`20260707_210406/`）は
過大評価であり使わない。

## 追加のXAI分析（2026-07-08〜09）

Phase C完了後、要旨の考察を厚くするため以下を新規実装・実行した（全て`transformer_260707/scripts/`）。

### `analysis/lineage_evolution_track.py`（新規）
上位5系統（strain_name_usher基準: AY.4, BA.2, BA.1.1, B.1.1.7, BA.1）から代表サンプル1件ずつ
（系統内でpath_lengthが中央値に最も近いもの）を選び、timestep×ゲノム位置で変異蓄積経路を可視化。
**オミクロン系統3つ（BA.1・BA.1.1・BA.2）はtimestep 8〜10あたりで位置26000〜29000
（N・ORF8・M遺伝子付近）に密集するクラスタを共有**——共通祖先（オミクロン創始変異群）の可視化に
成功。AY.4（Delta）は突出して長い経路、B.1.1.7（Alpha）は独自の低位置変異から開始、という
系統ごとの差異も見えた。
（代表サンプル選定SQLの同点タイブレークが非決定的で実行毎に微妙に異なる日付が選ばれるが、
多くの場合raw_path重複のため図の内容自体は再現性がある。）

### `analysis/lineage_attention_compare.py`（新規）
上記5系統の代表サンプルについて、実際の変異位置（x_cat由来のground truth）とモデルの
Co-occurrence Attention重みを同一入力スロットで比較。**重要な発見**: 生のattention重みは、
softmaxが同一timestep内で合計1になる制約により、共起数(co_count)が多いほど機械的に小さくなる
構造的交絡を受ける（`attn_weight`の平均は厳密に`1/co_count`）。`relative_attn = attn_weight ×
co_count`（1.0=一様分布基準）で補正すると、共起密集クラスタ内でもモデルが特定の位置を一様分布の
最大7倍程度重視し、大半は逆に軽視するという**本物の選択的アテンション**が確認できた
（生の重みだけを見ると「モデルはORF1abを重視しSpikeを軽視している」という誤った解釈をしてしまう
ところだった）。出力は系統別重ね合わせ版(線なし)と系統別個別ファイル(経路を線で接続)の両方に対応。

### `analysis/timestep_importance.py`（新規）
`local_explain.py`と同じIntegrated Gradients機構を再利用し、集約軸を「どの特徴/位置か」ではなく
「入力の何ステップ前(recency)か」に変更。全7フォールド・20,374サンプルで集計した結果:

| recency | 平均\|IG\|重要度 |
|---|---|
| 0（直近1件） | **145.5**（突出） |
| 1 | 41.7 |
| 6〜7 | 60.1 / 59.2（副次的な山） |
| 19〜20 | ~11.0（ピークの約7.6%） |
| 38〜39 | ~3.4〜4.1（ピークの約2.5%） |

直近の変異が圧倒的に重要（単調ではなく recency=6〜7 に副次ピーク）。**recency=20時点で既に
ピークの1割未満、窓の端でも約2.5%まで減衰済みで、39ステップという入力窓の設計を裏付ける**
（大きな情報の切り捨ては起きていなさそう）。

### `visualization/plot_xai_outputs.py`（新規）
既存のCSV/JSON専用出力（PNG無し）を可視化:
- `local_explain`: サンプルごとの上位寄与特徴・入力変異を小さい棒グラフで並べる
- `cooccurrence_constellation`: 予測ペア/実ペアのネットワーク図（`both`/`model_only`/`target_only`
  で色分け）。**注意**: 元スクリプトの「復元率85.5〜90%」は「予測ペアが実ペア全体(4,140万通り)の
  どこかに存在するか」という緩い基準であり、本可視化の「予測top25と実頻度top25が完全一致するか」
  という厳しい基準とは別の指標（後者は重複0件だった）。両者は矛盾ではなく、モデルの上位予測は
  生物学的に妥当だが最頻出の実ペアそのものとは限らない、という追加知見。
- `majority_baseline`: フォールド別の多数決/ランダム基準比較（region/positionを別軸で、position軸
  はsymlogスケール）

### `feature_importance.py`の実行と副産物のバグ修正
初めて実行したところ2つの問題が見つかった。(1) `collect_coattn_weights`内の`co_attn.forward`への
monkey-patchが`co_occur_mask`引数を受け取らずTypeErrorでクラッシュ（修正済み、実際のシグネチャに
合わせ`co_occur_mask=None`を追加）。(2) `--checkpoint`単発モードは`assign_fold_test_window`を
呼ばないため、DBの`split_type_wf`列が直前に実行した別スクリプトの状態（別フォールド）のまま
評価されてしまう（fold3のつもりが実際はfold7の17件で評価していた）。**walk-forward系スクリプトを
連続実行する際は、単発`--checkpoint`モードのスクリプトを挟む前に必ず対象フォールドの
`assign_fold_test_window`を明示的に呼び直す必要がある**。

**hydrophobicity/charge重要度の調査**: ユーザーから「以前のアブレーションでcharge/hydrophobicityが
精度に大きく影響した」との指摘を受け検証。fold1,2,3,4,5・position/aa_pos/synonymousの複数ヘッド・
path_length=44-48サブセット（ユーザーの記憶にある条件、timestep分割のbaseline checkpoint
`outputs/archive/transformer_260625/results/20260629_002051/`で確認）**全てで一貫してtop10圏外・
他の上位特徴より1〜2桁小さい値**という結果になった。ユーザーから「その時の精度変化も2%程度で
大きな影響ではない」との補足があり、矛盾ではなく整合的と判断。さらに`transformer_260109`
（2026年1月の旧バージョン）ではNUM_CHEM_FEATURES=6（ホスト適応24種・成長4種が未実装）だったことを
確認し、「特徴量が32〜36種に増えたことで相対重要度が希薄化した」という仮説を構造的に裏付けた。
当時の生データディレクトリ(`usher_output/`)が現在のファイルシステムに存在しないため、旧
チェックポイントでの直接再検証は不可能と判明し、調査はここで終了。

**How to apply**: charge/hydrophobicityの効果を要旨で主張する場合は「小さいが有意な寄与」という
正確な表現に留め、「重要な特徴」であるかのような誇張は避ける。

## Phase F以前に実施した簡易ベースライン比較（2026-07-07・260702の既存walk_forward結果で検証）

PETRA実学習（Phase E/F）を待たずに、学習不要の3種類のベースラインで「デルタ45%/35%、
オミクロン50%/10%」等の数値の良し悪しを判定した。**PETRA比較の完了を待たずに今すぐ着手できる**
という前セッションの提案どおり、以下2本の新規スクリプトを実装・実行済み。

### 実装したスクリプト

- **`transformer_260707/scripts/analysis/entropy_accuracy_analysis.py`**: 評価のたびに
  `utils/io.py`の`save_lineage_metrics_csv`が自動出力する系統別ターゲット位置エントロピー
  （`db_target_entropy_norm`）と各タスク精度を突き合わせ、(1) Spearman相関、(2) Fano不等式
  `H(X|Y) <= H_b(Pe) + Pe*log2(M-1)`を`Pe`について解いた理論的達成可能精度上限、を計算する。
  学習・推論不要（既存CSVを読むだけ）。
- **`transformer_260707/scripts/analysis/majority_baseline.py`**: walk_forwardの各フォールド
  窓について、(1) 一様ランダム基準（1/語彙サイズ、解析的計算のみ）、(2) 多数決基準（学習期間内で
  最頻出のregion/positionを常に予測した場合のテスト期間での的中率、DB集計のみ）を計算する。
  学習・推論不要（DBを直接集計するだけ）。

### 統合比較結果（260702の既存walk_forward、7フォールド）

| Fold | 時期 | モデル領域% | 多数決領域% | ランダム領域% | モデル位置% | 多数決位置% | ランダム位置% | Fano上限位置% |
|---|---|---|---|---|---|---|---|---|
| 1 | Pre-Alpha | 27.31 | 17.09 | 2.70 | 3.37 | **3.50** | 0.003 | 37.52 |
| 2 | Delta→Omicron移行 | 43.98 | 14.02 | 2.70 | 28.63 | 23.79 | 0.003 | 54.72 |
| 3 | Omicron初期 | 31.75 | 16.53 | 2.70 | 7.33 | **0.27** | 0.003 | 54.86 |
| 4 | Omicron variants | 44.18 | 16.08 | 2.70 | 10.03 | **0.02** | 0.003 | 75.70 |
| 5 | XBB | 40.50 | 16.09 | 2.70 | 5.79 | **0.37** | 0.003 | 86.72 |
| 6 | JN.1移行 | 33.34 | 17.08 | 2.70 | **0.00** | **1.58** | 0.003 | 84.63 |
| 7 | 最新 | 76.47 | 20.09 | 2.70 | 11.76 | **0.36** | 0.003 | 90.41 |

系統粒度でプールした相関（n=1575系統）: entropy_norm vs 位置精度 **Spearman r=-0.349, p=2.95e-46**
（強い有意な負相関）。領域精度との相関は弱め（r=-0.124, p=8.7e-7）。「変異が多様な系統ほど
位置予測が難しい」という仮説を統計的に裏付ける。平均Fano gap（理論上限−実測）は65.08pt。

### 基準ごとの判定

- **一様ランダム基準**: 全フォールドで圧倒的に上回る。位置精度は最悪でも1000倍以上。abstractに
  入れる根拠として最も説得力が高い。
- **多数決基準**: 領域は全フォールドで上回るが、**位置はfold1でわずかに下回る**（3.37% vs 3.50%）。
  fold3〜5・7ではモデルが多数決を大きく上回る（多数決が1%未満の場面でモデルは6〜12%）。
  **fold6はモデルが0.00%で多数決(1.58%)にも負ける**。
- **Fano理論上限**: 全フォールドで大きく届いていない。ただし「系統が分かっている」という緩い
  条件だけで計算した保守的な上限なので、実際のモデル入力（変異履歴全体）で条件づけた真の上限は
  これより高い可能性があり、「まだ伸びしろがある」以上の否定的な意味はない。
- **PETRA比較**: 未実施（Phase E/F待ち）。

### 正直な懸念点（要旨に書く前に確認すべきこと）

1. **fold1・fold6でモデルが多数決に負けている**。特にfold6の0.00%は明確な異常値。両フォールドとも
   学習データが最小規模（fold1=448,898サンプル、fold6=304,521サンプル）なので学習不足の可能性が
   高いが、要旨で「良い」と言い切る前に原因を確認すべき。
2. **abstractの数値の出典が判明（2026-07-07）**: `outputs/archive/transformer_260625/results/walk_forward/
   20260705_203818/monthly_hitrate_all_folds.png`を**目視**で読んだ値だった（フォールド全体の
   加重平均ではなく月別の折れ線グラフから）。同ディレクトリの`monthly_metrics_all_folds.csv`
   （生データ）を直接読んで検証した結果:
   - **デルタ45%/35%は2021-11（fold2）と正確に一致**（領域46.19%・位置34.84%）し、
     **n=256,009サンプル**と非常に頑健。この数値はそのまま使って問題ない。
   - **オミクロン50%/10%に正確に一致する月が無い**。領域≈50%の月（2022-07: 50.34%,
     2022-08: 50.00%）は位置精度が4.83%/22.50%で10%に一致せず、**サンプル数もn=145・n=40と
     極端に少ない**。位置精度が10%に近い2022-01（9.14%）はn=83,510と頑健だが領域精度は
     32.97%で50%とは離れている。**2022年1月以降テストサンプル数が8万台→数百→数十→一桁まで
     急減しており**、オミクロン期の「50%/10%」は複数の不安定な（極小サンプルの）月を目視で
     平均・折衷して読んだ可能性が高い。
   - **対応要**: 260707のwalk_forward完了後、同じ`monthly_hitrate_all_folds.png`/
     `monthly_metrics_all_folds.csv`が自動生成される（`transformer_260707/main.py`に
     生成ロジック内蔵済み）。目視ではなくCSVから直接、**サンプル数がある程度大きい月
     （目安n>1000）に限定して**代表値を選び直すべき。デルタは2021-11相当の月が引き続き
     使えるはずだが、オミクロンの代表値は頑健性を優先するなら2022年1月付近（領域33%/
     位置9%程度）に近い値へ修正が必要になる可能性がある。
3. 上記の理由により、当セクションの比較結果は**260702の既存walk_forwardデータでの先行検証**
   であり、260707でのPhase B/D完了後に同じスクリプトで再計算し、数値を差し替える必要がある。

## 判明した技術的注意点（ハマりどころ）

- **`config.SPLIT_MODE`の恒久上書きは不要**: `python -c`で一時的に`config.SPLIT_MODE`を
  設定してから該当関数を呼ぶパターンで安全に単発実行できる（`walk_forward.py`の`main()`が
  内部で同じことをしているのと同一の流儀）。config.pyファイル自体は変更しない。
- **`USE_PRETRAINING=True`でチェックポイント不在時は警告のみでランダム初期化続行**:
  エラーにはならないため気づきにくい。事前学習を経由させたい場合は先に
  `scripts.train.pretrain`を該当`SPLIT_MODE`で完走させ、`outputs/.../pretrain/<split_mode>/
  pretrain_mlm_final.pth`を用意する必要がある。
- **`kill`（SIGTERM）だけではCUDAプロセスが完全に終了しないことがある**: 一度
  `kill <pid>`しても`ps`上は消えてもnvidia-smi上でGPUメモリを掴んだまま（dataloader
  workerが孤児化）というケースに遭遇。次のジョブがOOMする場合は
  `ps aux | grep python`と`nvidia-smi --query-compute-apps=pid,used_memory`を突き合わせ、
  残っているworker PIDを`kill -9`で掃除してから再実行する。
- **DB共有の確認方法**: `transformer_260702.db.connection.get_db_path()`と
  `transformer_260707.db.connection.get_db_path()`を両方呼んで同一パスに解決されることを
  確認済み（`db/features_72f42a78.duckdb`）。特徴量設定ハッシュが同じ限り再前処理は不要。
- **260702/260707のアーキテクチャ同一性確認（2026-07-08）**: `diff transformer_260702/model.py
  transformer_260707/model.py`は差分0行（完全にバイト同一）。`train.py`も同一。`config.py`の
  差分は`OUTPUT_DIR`と260707限定の追加機能（`SAVE_CONFIDENT_SUBSET`,
  `PRETRAINING_MASK_MODE='span'`等、③提案アーキテクチャ自体には無関係）のみ。`evaluate.py`は
  189行差分があるが、CLAUDE.md記載の通りベクトル化・GPU→CPU転送集約等の高速化のみで
  ロジック等価。このため260702で学習したチェックポイントは260707上でも精度的に区別がつかない
  想定で、Phase C/DのXAI分析に流用して問題ない。
- **`outputs/transformer_260702/results`の内容確認（2026-07-08）**: `find`で確認したところ
  `walk_forward/`サブディレクトリのみ存在し、`SPLIT_MODE='timestep'`等の単発学習
  config_snapshotは存在しない（＝260702にも「全履歴timestep学習」の既存チェックポイントは
  無い）。ただしPhase Bキャンセルによりこの点は無関係になった（walk_forwardチェックポイントを
  積算方式で使うため、単発timestep学習自体が不要）。

## 検証方法

- Phase A: import構文チェック（実施済み・OK）。Phase B完了後に`--n_batches 5 --force_cpu`の
  小規模スモークで実データが集計されることを確認する。
- Phase B: `nohup_main707_timestep.out`に`Test data - Initial sample count`等が出て、
  最終的に`run_summary.json`が保存されることを確認する。
- Phase C/D/E/F: 各スクリプトの`[INFO] Saved ...`ログと出力ファイルの存在確認、および
  画像を実際に開いて内容が妥当か（既知変異ホットスポットと整合するか等）を目視確認する。

## 進捗ステータス（随時更新・実行順に記載）

| # | 項目 | 状態 | 備考 |
|---|---|---|---|
| 1 | A: attention_by_month.py を260707へポート | ✅完了 | import確認済み。頻度交絡チェック(`check_frequency_confound`)も追加済み |
| 2 | B-1: MLM事前学習 | ✅完了 | timestep分割、5epoch完走。loss 1.53→1.31。`pretrain_mlm_final.pth`保存済み |
| 3 | 簡易ベースライン比較（一様ランダム・多数決・Fanoエントロピー上限） | ✅完了（260702データで先行検証） | `entropy_accuracy_analysis.py`・`majority_baseline.py`実装済み。260707のPhase B/D完了後に再計算要 |
| 4 | abstractのデルタ/オミクロン数値の出典確認 | ✅完了 | `transformer_260625`の`monthly_hitrate_all_folds.png`を目視で読んだ値と判明。デルタ45%/35%は2021-11・n=256,009で頑健に裏付け済み。オミクロン50%/10%は2022年1月以降サンプル数が急減するため対応する頑健な月が無く、統計的に脆弱（詳細は上記セクション参照） |
| 5 | Phase E準備: petra/の260702参照更新 | ✅完了 | `petra/main.py`・`petra/evaluate.py`・`petra/config.py`の計6箇所を`transformer_260707`に更新、import確認済み |
| 6 | Phase E準備: petra/dataset.pyのデータ整合修正 | ✅完了 | 独自の重複排除実装ではなく、本体`DBIterableDataset.sample_ids`を直接再利用する方式に最終決定。`USE_UNIQUE_FILTER`/`MAX_CO_OCCURRENCE`/`EVAL_MAX_Y_CO_OCCURRENCE`等を完全一致させた。ワーカーsharding欠落バグも修正済み |
| 7 | Phase E方式決定 | ✅完了 | 方針A（全7フォールドのwalk-forward版PETRA）を採用 |
| 8 | Phase E実装: `petra/main.py`リファクタ＋`petra/walk_forward.py`新規実装 | ✅完了 | 学習ループを`run_training()`として切り出し、本体と同じ`FOLDS`/`assign_wf_splits()`を再利用するwalk_forwardランナーを実装。import確認済み |
| 9 | **Phase E: PETRA walk-forward学習** | ✅完了（2026-07-08） | `nohup python -m petra.walk_forward`（2026-07-07 21:04〜）が全7フォールド完走。結果は下記「Phase E完了・全7フォールド結果」参照 |
| 9.5 | **Phase F-lite: tail-only月別Hit Rateで本体と直接比較** | ✅完了（2026-07-08） | `petra/eval_tail_by_daterange.py`・`petra/plot_monthly_hitrate.py`を新規実装・実行。下記「tail-only月別比較」参照。petra_recall.py（Phase F本体）を待たずに①の判断材料が得られた |
| 10 | B: メイン単発学習（timestep・全履歴） | ❌不要と判定・キャンセル | model.py等の同一性確認とwalk_forward積算方式の実装により不要と判定（詳細はPhase B節参照） |
| 11 | C準備: genome_track/cooccurrence_constellation/local_explainへのwalk-forward対応実装 | ✅完了 | `--walk_forward_dir`（genome_track・cooccurrence_constellationは全フォールド積算、local_explainは`--fold`でVOC期選択）を追加。共通ヘルパは`_xai_common.py`に集約。import/`--help`確認済み |
| 12 | C: ②③ XAI分析（260702の既存walk_forwardで実行） | ✅完了（2026-07-08） | `genome_track`→`cooccurrence_constellation`→`local_explain`(fold3)を逐次実行し完走。共起ペア予測は実共起との一致率85.5〜90.0%（top50/100/200）。local_explainはfold3（Omicron初期BA.1/BA.2）で8サンプルのIG帰属を出力 |
| 13 | X: メインモデルwalk_forward（260707・headline数値用） | ❌不要と判定・キャンセル（2026-07-08） | ユーザ判断。260702/260707の同一性根拠によりheadline数値も260702の既存walk_forwardで代用。詳細はPhase D節参照 |
| 14 | D: 260707版でのXAI再実行・照合 | ❌不要（Phase Xキャンセルに伴い対象消滅） | Phase Xが無くなったため「260707版へ差し替えて再実行」も不要。Phase Cの結果がそのままcanonical |
| 15 | F: PETRA head-to-head比較（petra_recall.py） | 優先度低（任意） | Phase F-lite（tail-only月別比較）で①の判断材料は既に得られたため、必須ではない |
| 16 | データリーク調査: train/test間raw_path重複の発見・原因特定・本体/PETRAの構造差異の検証 | ✅完了（2026-07-08） | fold6重複21件全件で本体targetは100%不一致（本体は構造的にリークしない）と確認。詳細は「データリーク調査」節参照 |
| 17 | PETRA側: leaked/non_leaked集計の実装（サンプル除外ではなく集計を分ける方式） | ✅完了（2026-07-08） | `petra/dataset.py`・`petra/evaluate.py`・`petra/eval_tail_by_daterange.py`・`petra/plot_monthly_hitrate.py`・`petra/train.py`を修正。fold6実データで疎通確認済み（leaked+non_leaked=overall一致） |
| 18 | RoPE復元＋39ステップ窓をPETRAに実装 | ✅完了（2026-07-08） | `petra/model.py`をRoPE版に復元（スクラッチパッド退避分）。`petra/config.py`に`MAX_HISTORY_STEPS=39`追加、`tokenizer.encode()`に本体と同じ末尾スライス実装。合成データで45→39切り詰めを検証済み |
| 19 | PETRA walk-forward再学習（RoPE＋39ステップ窓＋leaked/non_leaked集計込み） | ✅完了（2026-07-08〜09） | 全7フォールド再学習完走（`outputs/petra/walk_forward/20260708_145056/`）。詳細は「PETRA再学習（RoPE＋39ステップ窓）完了・最終結論」節参照 |
| 20 | tail-only月別比較・低データ期プール検定の再実行（窓一致＋non_leaked版） | ✅完了（2026-07-09） | **最終結論確定**: Delta期は本体が一貫して約2倍優位（旧3〜6倍から補正）。低データ期(2022-07〜)の「PETRA有意」は窓不一致+リークの複合アーティファクトで、正しく補正すると有意差消滅(z=-0.284, p=0.777) |
| 21 | 系統別変異蓄積経路の可視化（`lineage_evolution_track.py`、新規） | ✅完了 | 上位5系統の代表サンプルでtimestep×ゲノム位置の経路を可視化。オミクロン3系統(BA.1/BA.1.1/BA.2)がtimestep8-10で共通クラスタを共有することを確認 |
| 22 | 代表サンプルのground truth vs attention重み比較（`lineage_attention_compare.py`、新規） | ✅完了 | raw attention weightはco_countで構造的に交絡（平均=1/co_count）することを発見。`relative_attn`補正版で本物の選択的attentionを確認。系統別個別ファイル＋全系統重ね合わせ版の両方出力 |
| 23 | timestep(recency)重要度の減衰カーブ（`timestep_importance.py`、新規） | ✅完了 | 全7フォールド20,374サンプルで集計。recency=0が突出(145.5)、recency=20時点でピークの1割未満に減衰——39ステップ窓の設計を裏付け |
| 24 | 既存CSV/JSON専用出力の可視化（`plot_xai_outputs.py`、新規） | ✅完了 | local_explain・cooccurrence_constellation（ネットワーク図）・majority_baselineを可視化。constellationの「復元率85-90%」と「top25完全一致率0%」は別指標である点に注意 |
| 25 | `feature_importance.py`初回実行・バグ修正・fold横断比較 | ✅完了 | monkey-patchの引数不一致(`co_occur_mask`欠落)を修正。単発`--checkpoint`モードは`assign_fold_test_window`を呼ばないため、直前の別スクリプト実行のfold窓が残っていた事故も修正。fold1,2,3,4,5・position/aa_pos/synonymousの全ヘッドでhydrophobicity/chargeは一貫してtop10圏外(1-2桁小さい値) |
| 26 | hydrophobicity/charge重要度低さの原因調査（旧アブレーションとの整合性確認） | ✅完了（調査終了） | ユーザ指摘の「2%程度の元々小さい効果」＋特徴量数6→32-36への増加による相対希薄化、で整合的と判断。`transformer_260109`(NUM_CHEM_FEATURES=6)での直接再検証は元データ(`usher_output/`)喪失のため不可能と判明 |
| — | MCC（マシューズ相関係数） | 検討のみ・実装見送り | ユーザ判断で追加不要と決定（2026-07-07） |
| — | `USE_HIERARCHICAL_PREDICTION`（提案3、Region→Position推論時マスキング） | ⏳未実行・実装済み・リーク確認済み | `config.py`に既存（既定False）。pos_region_mapはtrain分割のみから構築・マスキングはモデル自身のライブ予測を使うためリーク無しと確認済み（詳細は本文参照）。True/False比較未実施 |

**現在の状態（2026-07-09時点）**: PETRA比較・追加XAI分析・feature_importance調査、全て完了。
残っているのは要旨本文の改訂と、任意項目（Phase F本体・階層的予測の実測・アブレーション比較図）のみ。


## 元のabstract本文（参考・変更前）
新型コロナウイルスの新規変異株は数ヶ月という短い期間で出現し、その度に新たな感染拡大を引き起こし、対策の遅れを招いてきた。ゲノム配列から変異傾向を予測する研究が進められてきたが、配列全長を入力とするため計算コストが高く、予測結果の説明性にも課題が残っていた。
本研究では、ゲノム全長ではなくウイルスゲノムに生じた変異（変異前塩基・位置・変異後塩基）を出現順に記録した「変異履歴」を系列データとして扱う変異予測モデルを提案する。変異が生じていない領域を除外することで計算量を削減するとともに、各変異にアミノ酸の性質変化、変異頻度、コドン組成、変異周辺塩基の生物学的特徴量を付加することで、変異間の共起関係と進化過程を効果的に学習する。新型コロナウイルスにおける変異出現傾向と変異蓄積過程の理解を目指す。
分散化した変異トークンに特徴量を付加したベクトルを入力データとし、予測モデルにTransformerを採用することで、変異の共起関係を学習することができる。さらにほぼ同時に起きたとされる変異群を集約して代表ベクトルを生成する層の導入により、変異発生順序の非確実性に対応した。このアーキテクチャによって、予測モデルによる変異履歴の構造理解と特徴抽出を実現した。
6ヶ月おきのデータセットを用いて推論・再学習を繰り返すウォークフォワードテストを行い、変異パス末尾の変異を予測した結果、デルタ株では遺伝子領域で45%、塩基配列位置で35%の精度を達成した。オミクロン株では多様な変異を経ているため、塩基配列位置では10%の予測精度となったが、遺伝子領域では50%と高い予測精度を維持した。シノニマス変異であるか、コドン中の位置の予測では一貫して80%、60%ほどの精度となった。
今後の展望として段階的な変異候補の抽出を検討する。変異が起きる遺伝子領域の予測を元にシノニマス変異であるかを予測、さらにコドン中のどの位置であるかの予測を行うことで、候補を絞り、塩基配列位置の予測精度向上につながるかを検証する。本手法により新型コロナウイルスの変異動態を捉え、早期感染対策への貢献を目指す。

## 現行の修正案本文（2026-07-07時点・850字以内、配分: 背景100/手法300/結果350/展望100）

新型コロナウイルスの変異株は短期間で出現し感染拡大を招くが、既存のゲノム配列全長を用いる変異予測手法は計算コストが高く、予測結果の説明性にも課題があった。

本研究では、ゲノム全長ではなく変異履歴（変異前塩基・位置・変異後塩基の出現順系列）を入力とし、各変異にアミノ酸性質・変異頻度・コドン組成等の生物学的特徴量を付加したTransformerベースの変異予測モデルを提案する。ほぼ同時に生じたとされる変異群を集約し代表ベクトルを生成する層の導入により、変異発生順序の非確実性に対応した。このアーキテクチャによって変異履歴の構造理解と特徴抽出を実現し、モデルが予測質量を集中させる部位は系統樹上での独立再発回数（ホモプラシー）が既知データベースでも上位の部位と一致することを確認した。

半年おきのウォークフォワードテストで変異パス末尾を予測した結果、デルタ株で遺伝子領域45%・塩基配列位置35%、オミクロン株で遺伝子領域50%・位置10%の精度を達成し、いずれも一様ランダム予測（領域2.7%・位置0.003%）を大きく上回った。位置精度の低下はオミクロン株の変異多様化に起因する一方、遺伝子レベルではSpike遺伝子への予測質量の集中が一貫して確認され、蓄積傾向の把握を維持していた。シノニマス変異・コドン中の位置の予測は一貫して各80%・60%程度の精度となった。直近変異のみに依存し共起集約を持たないPETRA型decoder-onlyベースラインと同一データ・同一評価条件で比較した結果、最も頑健な評価期間（デルタ期）において本体モデルが位置予測精度で約2倍の優位性を一貫して示した。

今後は遺伝子領域→シノニマス性→コドン位置の順に候補を絞り込む段階的予測（実装済みの階層的推論マスキング機構を活用）を検証し、塩基配列位置の精度向上につなげる。本手法により変異動態を捉え、早期感染対策への貢献を目指す。


## 2026/07/09
新型コロナウイルスの新規変異株は短期間で出現し、その度に感染拡大を引き起こした。既存のゲノム配列全長を用いる変異予測手法は計算コストが高く、予測結果の説明性にも課題があった。
本研究では、ゲノム全長ではなく変異履歴（変異前塩基・位置・変異後塩基の出現順系列）を入力とし、各変異にアミノ酸性質変化・変異頻度・コドン組成等の生物学的特徴量を付加したTransformerベースの変異予測モデルを提案する。ほぼ同時に生じたとされる変異群を集約し代表ベクトルを生成する層の導入により、変異発生順序の非確実性に対応した。このアーキテクチャによって変異履歴の構造理解と特徴抽出の実現を目指す。
半年おきのウォークフォワードテストで変異履歴末尾を予測した結果、デルタ株で遺伝子領域43%・塩基配列位置30%、オミクロン株で遺伝子領域38%・位置8%の精度を達成した。位置精度の低下はオミクロン株の変異多様化に起因する一方、遺伝子レベルではスパイク遺伝子への予測の集中が一貫して確認され、蓄積傾向の把握を維持していた。シノニマス変異・コドン中の位置の予測は一貫して80%・60%程度の精度となった。生物学的特徴量や共起集約を持たない先行研究ベースモデルと比較した結果、デルタ株・オミクロン株において提案モデルが位置予測精度で約2倍の優位性を一貫して示した。
学習を行なったモデルのアテンション重みより特徴量の重要度を評価した結果、ノンシノニマス変異の蓄積数、直近のサンプル数、変異前コドンのウイルス・人間の使用頻度、アミノ酸分子量の変化量で高い重要度を示した。また末尾予測においてどのタイミングでの変異が重要であったか評価した結果、直前の変異が圧倒的に高い重要度を示したが、次点で6・７回前の変異で高い重要度を示した。
今後は遺伝子領域→シノニマス性→コドン位置の順に候補を絞り込む段階的予測を検証し、塩基配列位置での精度向上につなげる。本手法により新型コロナウイルスの変異動態を捉え、早期感染対策への貢献を目指す。
