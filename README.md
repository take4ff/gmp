# 🧬 Viral Genome Mutation Prediction Model (v260625)

本プロジェクトは、新型コロナウイルス（SARS-CoV-2）をはじめとするウイルスゲノムの変異発生予測を行う、高度な時系列マルチタスク・ディープラーニングモデル（Hierarchical Transformer）のコードベースです。
変異の時系列的な蓄積ステップ（Mutation Step）と、同時に発生した共起変異の集合を高度にモデル化し、将来のパンデミックを引き起こす変異の早期予測と進化シミュレーションを実現します。

---

## ✨ 主要機能（v260625 搭載機能）

- **NCBI/UShERダブル系統名・流行度システムの統合（動的切り替え）**
  - 配列メタデータ上の系統名（NCBI）と、系統樹トポロジー上の placements から得られる系統名（UShER）の双方をDBに抽出し、保持・表示。
  - UShERの `clades.txt` からNextclade名（Clade最頻値）およびサンプル数を自動集計する機構を導入。
  - `config.STRENGTH_SOURCE` の切り替え（`'ncbi'` / `'usher'`) だけで、モデル学習や評価時に参照する系統名・流行度を動的に切り替え可能。
- **共起変異コンテキストの高度化（前後3文字 / 計6塩基特徴量の追加）**
  - 変異が起きた位置の前後3文字（境界外は `'n'`）を新たなカテゴリカル特徴量（15次元特徴量）として追加。
  - `config.py` 内のアブレーションマスクフラグにより、学習中・推論中にこれらの周辺塩基を **1塩基単位で独立して無効化（マスク）** できる高精度アブレーション分析をサポート。
- **高精度マルチタスク学習**
  - 変異の発生する「遺伝子領域（37分類）」、「ゲノム位置（~3万分類）」、「アミノ酸位置（~1万分類）」、「コドン内位置（3分類）」、「同義/非同義（2分類）」、「塩基変化先（7分類）」、「アミノ酸変化先（23分類）」、およびその変異株の「流行規模（Strength Score / 回帰）」をマルチタスクで同時に学習・予測。
- **時系列マルチタスクの学習**
  - 変異の時系列ステップ（Mutation Step）に伴う進化の遷移を学習し、未知の時間の経過に合わせて変異発生時期を高度に予測。
- **Co-occurrence Attention の多段階拡張（Omicron期大規模共起への対応）**
  - `CO_ATTN_DIM`（内部次元拡大）、`CO_ATTN_N_LAYERS`（変異間 Self-Attention の多段化）、`USE_FLAT_COATTN`（全変異を独立トークンとして Transformer に直接入力）の3フラグで共起集約の表現力をアブレーション比較可能。
- **DuckDB による超高速データベース＆バルク処理**
  - 数百万レコードに及ぶ大規模な変異パスとゲノムメタデータを DuckDB で管理し、メモリ効率的かつ高速なストリームバッチ学習を実現。
- **メンテナンスフリーな動的ロギング**
  - バージョンアップに伴う日付コードの置換漏れを防ぐため、エラーログメッセージ内のパッケージ名を `{__package__}` を用いて自動取得するよう動的化。

---

## 🛠️ Architecture（アーキテクチャ）

同時に発生した共起変異の集合順序への依存性を解消し、位置エンコーディングの矛盾を防ぐための **Co-occurrence Attention (共起集約層)** を中核とする革新的なネットワーク構造です。

```mermaid
graph TD
    Input["Input Sequence (Mutations, Properties, Time)"] --> Embed["Embedding Layer\n(Base, Position, AA, Region, CodonPos, Synonymous,\n+/-3 Surrounding Context Bases)"]

    Embed --> FlatSwitch{USE_FLAT_COATTN?}
    FlatSwitch -->|False| CoAttn["Co-occurrence Attention\n(CO_ATTN_DIM / CO_ATTN_N_LAYERS で拡張可能)\n[B, T, C, F] → [B, T, F]"]
    FlatSwitch -->|True| FlatSeq["Flatten + 2D Positional Encoding\n[B, T, C, F] → [B, T×C, F]"]

    CoAttn --> Conv["Causal Conv1d\n(局所塩基文脈 / Optional)"]
    Conv --> OriAttn["Origin Attention\n(Wuhan参照株アテンション / Optional)"]
    OriAttn --> Transformer

    FlatSeq --> Transformer["Transformer Encoder\n(大局的時系列進化の読解)"]

    Transformer --> ARSwitch{USE_AUTOREGRESSIVE_DECODER?}
    ARSwitch -->|False| Heads{"Prediction Heads"}
    ARSwitch -->|True| ARDec["Autoregressive Decoder\n(タスク別クエリ埋め込み)"]
    ARDec --> Heads

    Heads --> Out1["Region Class (37)"]
    Heads --> Out2["Nucleotide Pos (~30K)"]
    Heads --> Out3["AA Pos (~10K)"]
    Heads --> Out4["Codon Pos (3)"]
    Heads --> Out5["Synonymous (2)"]
    Heads --> Out6["Strength Score (感染規模回帰)"]
    Heads --> Out7["Base After (7) / AA After (23)\n(USE_SUBSTITUTION_HEAD)"]
```

---

## 🚀 Usage（実行方法）

### 1. 環境構築
本プロジェクトは Linux 上で動作します。Conda 仮想環境等で必要なライブラリ（PyTorch、DuckDB、Pandas 等）をインストールしてください。

### 2. DuckDB データベースの前処理（構築）
前処理に必要な前後3塩基特徴量を含んだ高速データベースを構築します。
```bash
# 既存の重い学習プロセスを阻害しないよう、nice で優先度を下げて実行することを推奨
nice -n 19 python -m transformer_260625.preprocess
```

### 3. モデルの学習および評価の実行
```bash
python -m transformer_260625.main
```
実行完了後、`outputs/transformer_260625/results/<EXPERIMENT_NAME>/<timestamp>/` にすべてのログ、学習曲線、集計CSV、および評価プロット画像が自動保存されます。
（`config.EXPERIMENT_NAME = ''` のときは `results/<timestamp>/` に直接保存）

---

## 📊 Evaluation Metrics（評価指標）

- **Hit Rate（サンプル単位精度）**
  - 各サンプルについて「Top-K予測の中に、実際に発生した共起変異が1つでも含まれるか」を測定。
- **Weighted Recall@1 / Macro Recall@1（クラス単位評価）**
  - クラス不均衡が極めて大きい遺伝子位置予測において、高頻度クラスを重視する Weighted と、稀なクラスも公平に評価する Macro の双方でリコールを測定。
- **流行度カテゴリ（Strength Category）別比較**
  - 変異株の流行規模を `log(1 + sample_count)` スケールで Low（低流行）、Medium（中流行）、High（大流行）の3カテゴリに分類し、カテゴリごとの予測精度を多角的に分析。

---

## 🔬 Ablation Study（アブレーション実験）

`config.py` 内の以下のフラグで各特徴量やニューラルネットワーク層の有効/無効を切り替え、各コンポーネントが予測性能に与える影響度を精密に測定できます。

```python
USE_LOCAL_CONV1D = False         # Conv1d局所特徴抽出の切り替え
USE_ORIGIN_ATTENTION = False     # Origin Attention（武漢株参照）の切り替え

# Co-occurrence Attention 拡張（共起集約の表現力比較）
CO_ATTN_N_HEADS  = 4            # 共起 Attention のヘッド数（CO_ATTN_DIM で割り切れること）
CO_ATTN_N_LAYERS = 1            # 2以上で変異間 Self-Attention を多段化
CO_ATTN_DIM      = 256          # 内部次元（大きくすると集約表現力UP、最後に FEATURE_DIM へ投影）
USE_FLAT_COATTN  = False        # True: 全変異を独立トークンとして Transformer に渡す（要 BATCH_SIZE 削減）

# カテゴリ特徴量・数値特徴量の個別アブレーションマスク（再構築不要で動的適用）
ABLATION_MASKS = {
    'MUTATION_CONTEXT': False,     # 前後3文字（6塩基）を一括無効化
    'MUTATION_CONTEXT_L3': False,  # 3文字前 (P-3) を無効化
    'MUTATION_CONTEXT_L2': False,  # 2文字前 (P-2) を無効化
    'MUTATION_CONTEXT_L1': False,  # 1文字前 (P-1) を無効化
    'MUTATION_CONTEXT_R1': False,  # 1文字後 (P+1) を無効化
    'MUTATION_CONTEXT_R2': False,  # 2文字後 (P+2) を無効化
    'MUTATION_CONTEXT_R3': False,  # 3文字後 (P+3) を無効化

    # 各生化学的数値特徴量の個別マスク
    'FREQ': False,    # 変異発生頻度
    'HYDRO': False,   # 疎水性差
    'CHARGE': False,  # 電荷差
    'SIZE': False,    # サイズ差
    'BLSM': False,    # BLOSUM62スコア差
    'PAM250': False,  # PAM250スコア
    'CODON_LOG_RATIO_DIFF': False,    # コドン対数比差
    'CODON_LOG_RATIO_BEFORE': False,  # 変異前コドン対数比
    'HUMAN_CODON_RSCU_DIFF': False,   # ヒトRSCU差
    'SCV2_CODON_RSCU_DIFF': False,    # SCV2 RSCU差
}
```

---

## 🧬 Baseline Comparison（先行研究 PETra との比較実験）

[PETra](https://github.com/xz-keg/PETra) を含む先行研究に対する本提案アーキテクチャの有効性を科学的に検証するため、リファクタリング済みの同一コードベースを用い、**通常の損失関数（CrossEntropy等）のまま**、以下の3つのモデル構成を同一データセットで対比検証します。

| 比較軸 | モデル①: 先行研究ベース <br>(PETra相当) | モデル②: 中間モデル <br>(直列＋物理化学特徴量) | モデル③: 提案モデル <br>(集約層＋物理化学特徴量) |
| :--- | :---: | :---: | :---: |
| **同時変異（共起）の扱い** | 直列入力 (位置の嘘の時間差) | 直列入力 (位置の嘘の時間差) | **共起変異の集約層 (Co-occurrence Attention)** |
| **生物学的特徴量** | なし (生物学のカンペなし) | **あり** (電荷・疎水性・コドン頻度等) | **あり** (電荷・疎水性・コドン頻度等) |
| **位置エンコーディングの矛盾** | あり (直列順序に依存) | あり (直列順序に依存) | **なし** (集合としての処理により解消) |
| **検証の目的** | PETra相当の基準（Baseline）測定 | 物理化学特徴量による純粋な精度向上効果の証明 | 共起変異集約による本質的なアーキテクチャ効果の証明 |

### 比較実験のポイント
1.  **モデル① 先行研究ベース（直列入力 ＋ 特徴量なし）**:
    同時に発生した変異に対して、便宜的に直列の順序（嘘の時間差）を割り当てて Transformer に入力し、生物学的な特徴量（アミノ酸の電荷差、疎水性差など）を一切与えない構成です。これが先行研究 **PETra** 相当の基準点（Baseline）となります。
2.  **モデル② 中間モデル（直列入力 ＋ 特徴量あり）**:
    モデル①の構成に「電荷・疎水性・アミノ酸変異PAM250スコア・コドン頻度」などの生化学的特徴量を統合。これにより、「生物学的知識（カンペ）」の追加がどれほどダイレクトにモデルの予測精度を押し上げるかを純粋に実証します。
3.  **モデル③ 提案モデル（集約層あり ＋ 特徴量あり）**:
    物理化学特徴量に加え、同時に発生した共起変異を順序に依存せず集合として処理する **「共起アテンション集約層（Co-occurrence Attention）」** を適用。位置エンコーディングの矛盾が解消されることで、単なる特徴量の追加を超えた、アーキテクチャ設計による決定的な精度跳ね上がりを証明します。

---

### 📅 Future Roadmap（今後の展望）

予測精度のさらなる向上と生物学的妥当性の強化に向け、実装コスト・即効性・重要度を考慮した **3フェーズのロードマップ** に沿って段階的に機能を統合する計画です。

### 📅 Phase 1: 高優先度（Quick Wins & 評価・実運用基盤の確立）
実運用に向けた基盤の確立や、評価の妥当性向上に直結する最優先タスクです。

#### [x] 1. 系統未分類（Unclassified）サンプルの受容設計（一律 `0.0` フォールバック）
* **カテゴリ**: `【DB・データパイプライン】`
* **内容**: NCBI等の unclassified 株を一律で `"c"` に丸めず、`unclassified_[AccessionID]` として擬似系統名（Pseudo-strain）化して登録することでキャッシュ衝突を防止し、流行度を一律 `0.0`（最小値）に強制設定します。
* **ステータス**: **【完了】** (v260625) 擬似系統名による受容と、流行度最小値フォールバック処理を実装。->できていないかも、Unclassifiedの流行度が11になっている

#### [x] 2. UShERの `clades.txt` を用いた真の系統ベースの流行度集計
* **カテゴリ**: `【DB・データパイプライン】`
* **内容**: メタデータの Lineage テキスト集計ではなく、UShERの系統樹トポロジー placements が出力する `clades.txt` を直接パースして系統別のサンプル数を集計する機構。
* **ステータス**: **【完了】** (v260625) `clades.txt` をパースし、UShERサンプル数およびNextclade最頻値の集計・DB登録機構を実装。さらに、NCBI基準（メタデータ）とUShER基準（系統樹トポロジー）の動的切り替え（`config.STRENGTH_SOURCE`）に対応。

#### [x] 3. 変異シーケンスの事前学習（MLM/CLM 選択式）
* **カテゴリ**: `【アーキテクチャ・モデル】` / `【学習・訓練ロジック】`
* **内容**: 本プロジェクトの変異データ自体を活用し、双方向の **MLM（Masked Language Modeling）** または単方向の **CLM（Causal Language Modeling）** を config 経由で切り替えて事前学習を回す機構。
* **効果**: 共起関係や時系列推移のドメイン知識をあらかじめ獲得した強力な初期重みを得ることで、下流タスクのファインチューニング精度を極限まで高めます。
* **ステータス**: **【実装済み】** `config.USE_PRETRAINING = True` で有効化。`python -m transformer_260625.scripts.pretrain` で実行。MLM/CLM は `config.PRETRAINING_MODE` で切り替え。

#### [x] 4. 予測ターゲット間関係の自己回帰デコーダー（極小Decoder / アプローチB）
* **カテゴリ**: `【アーキテクチャ・モデル】`
* **内容**: 各マルチタスク予測ヘッド（Region, Pos, AA, CodonPos 等）の手前に非常にコンパクトな自己回帰デコーダー層を組み込みます。
* **効果**: 「変異位置決定 ➔ アミノ酸変異/コドン変異の決定」のように、出力ラベル間の依存関係を条件付き確率（自己回帰）で結合し、遺伝暗号表に則った生物学的に矛盾のない一貫した予測結果を実現します。
* **ステータス**: **【実装済み】** `config.USE_AUTOREGRESSIVE_DECODER = True` で有効化。層数は `AR_DECODER_LAYERS`、ヘッド数は `AR_DECODER_HEADS` で調整。

#### [x] 5. 予測対象（ターゲット）の拡張：具体的な塩基置換・アミノ酸置換の直接予測
* **カテゴリ**: `【アーキテクチャ・モデル】`
* **内容**: 変異が発生する位置情報だけでなく、具体的に「どの文字からどの文字へ変化したか」という遷移確率（4x4の塩基置換、および20x20のアミノ酸置換）自体を直接予測するヘッドを追加します。
* **効果**: 変異の「発生位置」と「置換内容」の双方を完全予測可能とし、進化シミュレータとしての実用性を飛躍的に高めます。
* **ステータス**: **【実装済み】** `config.USE_SUBSTITUTION_HEAD = True` で有効化。塩基変化先（7クラス）・アミノ酸変化先（23クラス）のヘッドを追加。

#### [x] 6. 共起変異数 $R$ に応じた動的評価指標（R-Precision）の追加
* **カテゴリ**: `【評価・検証機構】`
* **内容**: 一律の固定値 $K$ (Top-1, Top-5等) ではなく、サンプルごとに実際の変異共起数（ユニークな正解数 $R$）を $K$ とした動的な Top-$R$ 予測精度・再現率を計測する評価機構。
* **効果**: 共起数が $K$ を上回るとRecallが100%に到達できない問題を完全に解消し、モデル本来のカバー能力を正確に可視化します。
* **ステータス**: **【実装済み】** `config.USE_R_PRECISION = True` で有効化。`evaluate_topk()` が固定K結果と並列に `'r_precision'` キーで結果を出力。

#### [x] 7. 基準日ベースの時系列分割（Temporal Split）と時間軸評価（Temporal Evaluation）の統合
* **カテゴリ**: `【評価・検証機構】` / `【DB・データパイプライン】`
* **内容**: `config.py` 内の設定フラグのみで、特定の基準日（例: 2023-12-31）以前を学習データ、以降をテストデータとして自動分割し、評価グラフのX軸を「月別（Year-Month）」などの時間軸に切り替えるスマートなマルチモード評価機構。
* **効果**: 既存の `main.py` などを書き換えることなく、実用的な将来予測能力（時間の経過に伴う予測精度の推移）を安全に検証できるようになります。
* **ステータス**: **【完了】** (v260625) `config.SPLIT_MODE = 'date'` で有効化。

---

### 📅 Phase 2: 中優先度（生物学的ドメイン知識の統合）
アミノ酸の立体構造や進化的特性といった「生物学的カンペ」を明示的に与え、予測タスクの表現力を格段に引き上げるコア開発フェーズです。

#### 8. 事前学習済みPLM（Protein Language Model：ESM-2）の統合（アプローチA）
* **カテゴリ**: `【特徴量・ドメイン知識】` / `【アーキテクチャ・モデル】`
* **内容**: 軽量な **ESM-2**（例: `esm2_t6_8M_UR50D`）を用い、変異位置の周辺30〜50アミノ酸の局所配列から文脈依存の表現（Embedding）ベクトルを抽出。これを既存の `combined` 特徴量に結合します。
* **効果**: タンパク質の3次元立体構造や進化的な変異許容度（保存度）といった数億規模の事前知識を、モデル構造を大きく変えることなく安全かつ低コストに既存パイプラインへ注入できます。
* **ステータス**: **【config・モデルフック実装済み】** `config.USE_ESM2 = True` で有効化（フラグ・射影層のみ）。実際の ESM-2 特徴量抽出（`preprocess.py` との連携）は未実装。

#### 9. 立体構造特性（3D座標・SASA・B-factor）の明示的特徴量の統合
* **カテゴリ**: `【特徴量・ドメイン知識】`
* **内容**: Spike等のAlphaFold/PDB構造情報から、変異位置の **SASA（溶媒露出表面積：表面か内部か）** や **B-factor（分子のゆらぎ）**、立体中心からの距離等を直接数値特徴量としてマージします。
* **効果**: AIが塩基置換から間接的に立体構造変化を推測する学習負担を排除し、構造依存の予測性能を極限まで高めます。
* **ステータス**: **【config実装済み】** `config.USE_STRUCTURE_FEATURES = True`、`STRUCTURE_CSV` でCSVパス指定。構造データファイルの準備と `preprocess.py` 統合は未実装。

#### 10. ウイルスの進化適応度・免疫逃避能（先行研究 EVEscape）の統合
* **カテゴリ**: `【特徴量・ドメイン知識】`
* **内容**: 著名なゼロショット進化適合度モデル **EVEscape** のスコアリングロジック、あるいは変異ごとのEVEscape予測スコア自体を新たな数値特徴量としてマージします。
* **効果**: 「生化学的に発生しやすい変異（塩基置換特性）」と「環境下で生存・免疫逃避しやすい変異（Fitness）」の両面から多角的に将来変異をシミュレーションできるようになります。
* **ステータス**: **【config実装済み】** `config.USE_EVESCAPE = True`、`EVESCAPE_CSV` でCSVパス指定。スコアデータファイルの準備と `preprocess.py` 統合は未実装。

---

### 📅 Phase 3: 将来優先度（先端深層学習パラダイムの適用 & 極限の汎化）
基本アーキテクチャの完成後、予測の限界を突破し、未知のパンデミックや別ウイルスに適用するための高度なスケーラビリティ検証フェーズです。

#### [x] 11. 変異段階（Mutation Step）＆共起数に基づくカリキュラム学習
* **カテゴリ**: `【学習・訓練ロジック】`
* **内容**: 初期フェーズでは単純な変異（変異段階が短く共起が1）から学習し、徐々に難易度（多重共起や長い変異段階）を上げていく段階的学習機構。
* **効果**: 基本的な変異しやすい位置（位置バイアス）を定着させてから高度な共起関係を学ばせるため、収束を劇的に助け、不必要な過学習を効果的に防ぎます。
* **ステータス**: **【完了】** (v260625) `config.CURRICULUM_WARMUP_EPOCHS` でウォームアップ期間を指定。`max_length` フィルタで短パスから段階的に開放する方式で実装。

#### [x] 12. 教師あり対照学習（SupCon）による構造化表現学習
* **カテゴリ**: `【学習・訓練ロジック】`
* **内容**: 同一系統や同一流行度のシーケンスを引き合わせ、異なる特性のシーケンスを引き離す対照損失（SupCon Loss）による表現学習。
* **効果**: モデルがウイルスの系統樹トポロジーや進化マップを高次元空間に美しく構造化して記憶できるようになり、未知の系統や出現初期の変異株に対する汎化性能を飛躍的に高めます。
* **ステータス**: **【実装済み】** `config.USE_SUPCON = True` で有効化。温度パラメータ `SUPCON_TEMPERATURE`、損失重み `SUPCON_WEIGHT`、射影次元 `SUPCON_PROJECTION_DIM` で調整可能。

#### [x] 13. 時系列＆アーキテクチャのマルチビュー・アンサンブル
* **カテゴリ**: `【学習・訓練ロジック】` / `【評価・検証機構】`
* **内容**: 異なる基準日（時間窓）で学習させたモデルや、局所文脈抽出の有無（Local/Global）が異なる異種モデルをブレンディング（重み付き平均）して予測するアンサンブル機構。
* **効果**: 将来の進化トレンドシフトに対する極めて頑健な予測と、予測確率の最終安定化を達成します。
* **ステータス**: **【実装済み】** `config.USE_ENSEMBLE = True` かつ `ENSEMBLE_CHECKPOINT_PATHS` にチェックポイントパスのリストを設定することで、`main.py` 実行後に自動的にアンサンブル評価が実行される。

#### 14. 他ウイルスへの汎化検証（先行研究 PETra を踏まえたマルチバイラル展開）
* **カテゴリ**: `【評価・検証機構】` (ドメイン汎化検証)
* **内容**: COVID-19で確立した本構成を、インフルエンザ（Influenza）などデータ豊富な他ウイルスのゲノムデータセットに適用し、先行研究 **PETra** が掲げた予測モデルのマルチバイラルな汎用性を検証します。

---

※ 上記のすべての追加・高度化機能は、`config.py` 上のフラグによりワンタッチで有効/無効の切り替えが可能な設計とします。

---

## 📝 License

[MIT License / Research Use Only]

---

**Author:** Takeru Aiba


---

## 🗒️ 開発メモ（未整理タスク）

### 評価・出力の改善
- **感染規模別の精度ファイル分割**: 現状は一括出力になっているため、感染規模カテゴリ（Low/Med/High）ごとにファイルを分けて保存する
- **感染規模区分の細分化**: 大中小の3区分に加え、対数スケール（〜1,000 / 〜1万 / 〜10万 / 〜100万）でも精度を集計・保存（`{prefix}_strength_fine.csv`）。各系統がどの規模区分に該当するかもセットで保存する
- **亜系統別の精度出力**: timestep 軸とは独立した切り口で、系統（clade）単位に集計した精度を `{prefix}_lineage_metrics.csv` として常時出力する（現状は株単位の `strain_metrics.csv` のみ）
- **亜系統別の予測難易度の可視化**: ターゲット変異分布の偏り（エントロピー等）を定量化し、亜系統ごとの「予測しやすさ」を評価結果に付加する
- **run_summary.json の追加**: 主要指標（HitRate@1/3/5・Macro Recall・Val/Test Loss）だけをまとめた軽量 JSON を毎回出力し、実験間のスクリプト比較を容易にする
- **R-Precision 結果の独立保存**: `USE_R_PRECISION=True` 時の結果を `topk_precision.csv` とは別ファイルに保存する

### 出力ファイルの整理（削減・統合）
- **削除候補**: `{prefix}_random_baseline.csv`（データ統計から決まる固定値に近く毎回生成は冗長）、`{prefix}_prediction_distribution.csv`（`confusion_matrix.csv` と情報が重複）、`{prefix}_recall_summary.csv`（`metrics_by_timestep.csv` / `region_metrics.csv` と内容が被りやすい）
- **フラグ制御化候補**: `{prefix}_strain_metrics.csv`（1万株×複数指標で巨大、詳細分析時のみ生成）、`synonymous_distribution_csv` / `strain_info_csv`（`USE_DB=False` 時のみ意味があり現状ほぼ死にコード）

### 特徴量の拡充
- **シノニマス/ノンシノニマス変異の蓄積数**: 入力パスにおけるシノニマス変異・ノンシノニマス変異それぞれの累積カウントを数値特徴量として追加する
- **コドン頻度の生値（差分）**: 現状は対数比（`Codon_log_ratio`）のみ。RSCU と同じ構造で `HUMAN_CODON_FREQ_DIFF`（ヒト）・`SCV2_CODON_FREQ_DIFF`（SCV2）の変異前後の差分を追加する。さらに絶対レベルの情報として `HUMAN_CODON_FREQ_BEFORE` / `SCV2_CODON_FREQ_BEFORE`（変異前の生値）も候補だが、特徴量増加とのトレードオフをアブレーションで確認してから判断する
- **前後塩基の窓幅拡張**: 現状は前後 ±3 塩基。±5 〜 ±7 程度まで拡張して文脈情報を増やす
