# 学習プロセス改善・精度向上プラン（検証キュー）

作成日: 2026-10-06 / 対象バージョン: `transformer_260817`（正典）

2026-10-05の全体見直しで挙がった改善候補を、1つずつ検証していくための記録。
既にA/B済みで効果なし/悪化が確定したもの（`CO_ATTN_N_LAYERS=2`、`TEMPORAL_POOLING=mean/cls`、
`FFN_RATIO=8`、`USE_AUTOREGRESSIVE_DECODER`、`USE_LOCAL_CONV1D`、`USE_ORIGIN_ATTENTION`、
`USE_BROADCAST_BACK_ATTENTION`、`USE_COATTN_FREQUENCY_PENALTY`、`LOSS_FUNCTION_TYPE=cbce/focal`）は対象外。

## 検証キュー

| # | 項目 | 再学習 | 状態 | 結果・備考 |
|---|---|---|---|---|
| 1 | `MultiTaskLoss`の学習済みタスク重みのログ出力 | 不要（観測のみ） | ✅実装済み | 次回学習で`[MultiTaskLoss weights]`・`training_log.csv`の`weight_<task>`列を確認し、Position重みが手動優先度（0.7）と乖離していないか見る。乖離が大きければ固定重み or `log_vars`初期値の手動重み合わせを検討 |
| 2 | `USE_HIERARCHICAL_PREDICTION`（推論時Regionマスキング） | 不要 | ✅**全7fold検証済み**・既定`True` | `20260803_115144`（Region条件付けなし）全7fold: 加重平均 **+0.105pt**（21.5349→21.6398%）。fold別: f1 **−0.331**／f2 −0.010／f3 +0.309／f4 **+0.401**／f5 +0.032／f6 +0.004／f7 +0.009。効くのはOmicron期(f3・f4)、Pre-Alpha(f1)は悪化。Region条件付けあり（現行）はfold_3のみ（5.1026→5.4709%、+0.368pt）で全fold平均は未確認。出力: `outputs/transformer_260817/scripts/hierarchical_prediction_check/20261007_031029/` |
| 3 | `USE_HOMOPLASY_PRIOR`（Positionへのlog(1+再発回数)バイアス） | 本来は必要 | ✅後付け簡易チェック完了（**方向性は負**） | `homoplasy_prior_posthoc_check.py`、fold_3、scale=1.0固定（未学習）。position_hit_rate **4.0084%**（基準5.4709%、**−1.46pt**）、region 31.6153%（不変）、aa_pos 12.8801%。未学習の固定スケールで過大に効いた可能性があり、学習時ONでの公平な評価ではない。**優先度低**: 学習時ONは12月以降の余力があれば（学習でscaleが縮む可能性は残る）。scale掃引（0.1/0.3等）は各1時間 |
| 4 | `USE_KNN_OUTPUT`（kNN-LM型補間） | 不要（モデル再学習なし） | ✅パイロット完了（**効果なし・悪化**、2026-10-07実施） | 配線: `build_datastore`がSoft Target(dict)で空になる問題を修正し疎(CSR)形式に、`evaluate()`に配線（階層マスク後に許可位置内でのみ補間）、`faiss-cpu`導入、単体テスト13件。**パイロット**（fold_3、Region条件付けあり`20260809_194502`、datastore=trainから30万群・81.6万ラベル、test10万群を無作為抽出、k=16）: position_hit_rate λ=0 **5.401%**／λ=0.1 5.373%（−0.028pt）／λ=0.25 5.242%（−0.159pt）／λ=0.5 4.682%（−0.719pt）。regionは不変(31.436%)。λが大きいほど単調に悪化。**判断: 現設定では不採用、`evaluate_topk`への配線は見送り**。未検証: kの変更・datastore増量・他fold。悪化理由の仮説（fold_3は分布変化が大きくtrain近傍がtestの代表にならない）は未確認。出力: `outputs/transformer_260817/scripts/knn_output_check/`。実行時間: 構築約38分（起動9分＋順伝播30万群で約30分）、評価約1時間（起動13分＋10万群×4λで約37分） |
| 5 | `USE_CLADE_EMBEDDING`（主要クレード埋め込み） | 必要 | ⏸未着手 | fold_1（Pre-Alpha）のみ本体がPETRAに劣る既知の弱点（分布シフト）への対策候補 |
| 6 | `USE_TRAIN_ENTROPY_FILTER`（高エントロピー系統を学習から除外） | 必要 | ⏸未着手 | `TRAIN_ENTROPY_MAX=0.7`要チューニング。学習データ減とのトレードオフに注意 |

## 付随する確認事項

- **fold_3の評価指標への打ち切り影響（未検証）**: OOM対策の`MAX_GROUP_MEMBERS_FOR_CACHE=4000`・
  `MAX_R_PRECISION_K=4000`がfold_3のRecall@K・R-Precisionに与える影響は未評価。fold_3の数値を
  学会要旨等で使う前に確認する。
- **`gc.freeze()`の効果検証（未実施）**: 次のwalk_forward実行で`_log_rss`（`[MEM]`ログ）の
  RSS推移を、対策前（`outputs/.../20260803_115144`等のログ）と比較する。
- **現行config（`USE_REGION_CONDITIONED_POSITION=True`）での全7fold学習は未実施**
  （`20260809_194502`はfold_3のみ）。#2の全fold平均、#3の公平な評価、#1の重み確認は
  この全fold学習をもって一括で行うのが効率的。

- **#5・#6の配線テストと残タスク（2026-10-08）**: `USE_CLADE_EMBEDDING`・`USE_TRAIN_ENTROPY_FILTER`は
  実装済みだがテストが無かったため、`tests/test_clade_and_entropy_filter.py`（28件、変異テスト7種で検出力確認）を追加した。
  **✅`config.X`直読みへ書き換え済み（2026-10-10）**。元の残タスク: `getattr(...)`等を（CLAUDE.mdルール4。
  対象: `train.py`・`evaluate.py`・`model.py`の`USE_CLADE_EMBEDDING`、`db/dataset.py`の`USE_TRAIN_ENTROPY_FILTER`・`TRAIN_ENTROPY_MAX`）。
  DB構築・学習の実行中は`transformer/`を書き換えない（ルール9）ため、完了後に実施する。
  なお`clade_embed`はON時のみstate_dictにキーが増える（`_OFF_IF_ABSENT_FROM_SNAPSHOT`への登録は、既定をONにする場合に必要）。

## 月例報告項目「理論上限に届かないデータからの精度向上」との対応（2026-10-07）

Fano上限と達成率の分析（`fano_gap_*`・`lineage_accuracy_drivers`、7/21）は実施済み。未着手は
**ギャップが大きい（達成率が低いのにエントロピーは低い）系統・月の特定と個別の原因調査**、
および回帰が示した「分岐系統は不利」への対策（`USE_FREQ_WEIGHTED_TARGETS`等、未検証）。
サンプル数カーブのフィットはR²が負で、必要サンプル数の見積もりは信頼できない。
詳細は [PLAN_thesis_schedule.md](PLAN_thesis_schedule.md) の §7。

## 次のアクション

1. ✅階層的予測の全7fold評価と特徴量×recencyマップ（完了 2026-10-07、[PLAN_thesis_schedule.md](PLAN_thesis_schedule.md)）
2. ✅#4 kNNパイロット完了（効果なし、不採用。上表）
3. #3は優先度低（結果は負）。#5/#6の再学習系は11月の学習枠と合わせて判断
4. 現行configで全7fold walk_forwardを実行し、#1・#2・`gc.freeze()`の確認を同時に行う

## 関連

- 詳細な修正履歴: [docs/changelog.md](docs/changelog.md)
- テスト・プリフライト検査（導入済み）: CLAUDE.md の「主要コマンド」（`python -m pytest -q`、`preflight_check`）
