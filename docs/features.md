# ✨ 主要機能（v260707 搭載機能）詳細

[← README に戻る](../README.md)

- **NCBI/UShERダブル系統名・流行度システムの統合（動的切り替え）**
  - 配列メタデータ上の系統名（NCBI）と、系統樹トポロジー上の placements から得られる系統名（UShER）の双方をDBに抽出し、保持・表示。
  - UShERの `clades.txt` からNextclade名（Clade最頻値）およびサンプル数を自動集計する機構を導入。
  - `config.STRENGTH_SOURCE` の切り替え（`'ncbi'` / `'usher'`) だけで、モデル学習や評価時に参照する系統名・流行度を動的に切り替え可能。

- **追加Usher出力ディレクトリ・追加サンプル一覧CSVの取り込み**
  - `config.DATA_BASE_DIR`（デフォルト `../usher_output/`）に加え、`config.EXTRA_DATA_BASE_DIRS`（リスト）で同一の `<lineage>/<N>/` 構造を持つ追加のUsher出力ディレクトリを取り込める。`scripts/preprocess/data_format.py` と `preprocess.py` は両方とも自動的に全ディレクトリを横断して処理する（`--base-dir` 指定時は1ディレクトリのみに限定可能）。
  - 対応する追加サンプル一覧は `config.EXTRA_SEQUENCES_CSV`（リスト）で指定し、`SEQUENCES_CSV` と同一スキーマ前提で結合ロードされる。
  - 現行設定: `EXTRA_DATA_BASE_DIRS = ['../usher_output2/']` / `EXTRA_SEQUENCES_CSV = ['reference/sequences-241017_2.csv']`。

- **ホスト適応特徴量の統合（24次元）**
  - `reference/codon/SCV2_host_adaptation_features.csv` に集約された24種類のコドン組成ベース特徴量を数値入力に追加。
  - ヒト・SCV2 の RSCU・コドン頻度・CAI（相対的適応度）の変異前後の値と差分、コドン最適性フラグ（optimal_to_optimal 等）、トランジション/トランスバージョン、CpG/UpA ジヌクレオチド変化量、宿主距離対数比などを網羅。
  - `reference/codon/generate_host_adaptation_features.py` で再生成可能。

- **パス内累積変異カウント特徴量の追加**
  - 入力パスにおけるシノニマス変異・ノンシノニマス変異それぞれの累積カウントを数値特徴量（`cum_syn`/`cum_nonsyn`）として追加。
  - 各タイムステップ開始前の累積値を渡すため、同一タイムステップ内の共起変異は全て同じ値を参照する。

- **亜系統の流行ダイナミクス特徴（4次元・因果的な適応度代理）**
  - 各サンプル（亜系統 × 収集日）に、**収集日以前のデータだけ**で計算した4つの成長特徴を数値入力に追加（`USE_LINEAGE_GROWTH_FEATURES`）。適応度は「どの変異が出現・拡大するか」を駆動するため、位置予測の文脈情報になる。
  - `lineage_log_count_recent`（規模の因果版）/ `lineage_growth_rate`（直近K週の log件数の傾き＝勢い、正=拡大・負=縮小）/ `lineage_rel_growth_adv`（logit シェアの傾き＝共流行系統に対する**相対成長優位＝logistic fitness**）/ `lineage_growth_accel`（加速度＝ピークアウト検知）。
  - epidemic 件数は系列メタCSV（`Pangolin` × `Collection_Date`）から週次で構築（`db/growth_features.py`）。系統粒度は Pango 上位2階層。「縮小期の件数」等の**未来量は使わずリークを防止**し、位相は `growth_rate` の符号で表現する。
  - 有効化すると `NUM_CHEM_FEATURES` が 32→36 に増え **DB 再構築が必要**。効果は permutation importance／ablation／walk-forward で検証する（既存 `strength` との冗長性・リークの有無を確認）。

- **前後コンテキスト塩基の窓幅拡張（`CONTEXT_WINDOW` で制御）**
  - 変異位置の前後塩基を `config.CONTEXT_WINDOW`（デフォルト ±5）で動的に制御。±3/±5/±7 を config の変更だけで切り替え可能。
  - カテゴリ特徴量 = 9（固定）+ 2×CONTEXT_WINDOW（コンテキスト）。
  - アブレーションマスクも `MUTATION_CONTEXT_L1`〜`L7` / `R1`〜`R7` で1塩基単位の個別制御に対応。

- **高精度マルチタスク学習**
  - 変異の発生する「遺伝子領域（37分類）」、「ゲノム位置（~3万分類）」、「アミノ酸位置（~1万分類）」、「コドン内位置（3分類）」、「同義/非同義（2分類）」、「塩基変化先（7分類）」、「アミノ酸変化先（23分類）」、およびその変異株の「流行規模（Strength Score / 回帰）」をマルチタスクで同時に学習・予測。

- **時系列マルチタスクの学習**
  - 変異の時系列ステップ（Mutation Step）に伴う進化の遷移を学習し、未知の時間の経過に合わせて変異発生時期を高度に予測。

- **Co-occurrence Attention の多段階拡張（Omicron期大規模共起への対応）**
  - `CO_ATTN_DIM`（内部次元拡大）、`CO_ATTN_N_LAYERS`（変異間 Self-Attention の多段化）、`USE_FLAT_COATTN`（全変異を独立トークンとして Transformer に直接入力）の3フラグで共起集約の表現力をアブレーション比較可能。

- **DuckDB による超高速データベース＆バルク処理**
  - 数百万レコードに及ぶ大規模な変異パスとゲノムメタデータを DuckDB で管理し、メモリ効率的かつ高速なストリームバッチ学習を実現。

- **多角的な評価・出力システム**
  - 感染規模9区分（〜100/〜500/〜1K/〜5K/〜10K/〜50K/〜100K/〜500K/>500K）の精度ファイル分割。
  - 系統（Pango 上位2階層）単位でのHitRate・予測難易度エントロピーを集計した `{prefix}_lineage_metrics.csv` を常時出力。
  - 実験横断スクリプト比較のための `run_summary.json`（主要指標を軽量JSONで保存）。
  - `USE_R_PRECISION=True` 時の R-Precision 結果を独立ファイル `{prefix}_r_precision.csv` に保存。

- **メンテナンスフリーな動的ロギング**
  - バージョンアップに伴う日付コードの置換漏れを防ぐため、エラーログメッセージ内のパッケージ名を `{__package__}` を用いて自動取得するよう動的化。

- **実験ハーネス・高速化・分析ツール（v260707 拡張）**
  - **マルチシード検証**（mean±std 集計）・**Walk-forward 検証**（半年次7フォールド）・**Optuna ハイパラ探索**（オフライン SQLite・pruning）の3ハーネス。
  - **DataLoader 並列化**によるデータ律速の解消（実測 ~2.0x）、**プロット出力の個別トグル**、**早期終了の hit-rate 対応**。
  - **特徴量重要度**（Permutation＝精度寄与 / Gradient×Input＝感度）に加え、**生物学的 XAI スイート**（ゲノムトラック／ホモプラシー整合／コンステレーション復元／Integrated Gradients 局所説明／表現プロービング）を単独スクリプトで提供。`run_all_xai` で一括実行可。
  - 詳細は README「⚙️ 学習設定・実験ハーネス・高速化」節および [docs/xai_usage.md](xai_usage.md) を参照。
