# --- config.py ---
#
# ============================================================
#  目次（セクション見出しで検索できる。例: "# --- 損失関数設定 ---"）
# ============================================================
#  基本         : DEVICE / SEED / マルチシード実験ハーネス
#  Ablation     : ABLATION_MASKS（特徴量0マスク）
#  データ        : データロード / 流行度 / 感染規模フィルタ / パス / DuckDB /
#                 キャッシュ / 保存・ログ / データセット / サンプリング / ボキャブラリー
#  モデル        : モデルアーキテクチャ / Conv1D / Origin Attention /
#                 Temporal Pooling / Shared Trunk / Co-occurrence Attention
#  訓練         : 訓練設定 / 高度な学習 / Soft-Hard-Hybrid Target / 損失関数 /
#                 Warmup / 学習安定化 / Early Stopping
#  評価・可視化   : Top-K / プロット出力の個別トグル(PLOT_OUTPUTS) / strength損失 / ECE
#  精度向上実験   : Curriculum / ALiBi(RPE) / Bio-Informed Loss / TTA
#  ロジック選択   : Optimizer / モデルアーキ / 流行度前処理
#  時系列分割     : データ分割モード(SPLIT_MODE) / walk_forward
#  新規拡張(10)  : Pretraining(MLM/CLM) / AR Decoder / Substitution Head /
#                 R-Precision / ESM-2【未実装】/ 構造特徴【未実装】/ EVEscape【未実装】/
#                 SupCon / Ensemble
#  ラベル多義性(11): 提案6〜12（頻度重み / 枝分解 / Mixture Head 等、既定は全 False）
# ============================================================
import torch

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# --- マルチシード実験ハーネス ---
# True にすると MULTI_SEED_LIST の各シードで学習＋評価を回し、
# 主要指標を mean±std に集計する（single run のノイズと真の改善を区別するため）。
# walk_forward モードとは排他（walk_forward が優先）。デフォルトは無効。
MULTI_SEED = False
MULTI_SEED_LIST = [42, 1, 7]

# --- Optuna ハイパラ探索（scripts/eval/optuna_search.py が使用）---
# ローカル SQLite に保存するためオフライン・再開可能（長時間学習で接続が切れる問題と無縁）。
# 1 trial = 1 学習。目的関数は run_summary.json の val 指標。pruning で悪い trial を早期打ち切り。
OPTUNA_N_TRIALS        = 30
OPTUNA_STUDY_NAME      = 'gmp_hpo'
OPTUNA_STORAGE         = None            # None → outputs/.../scripts/optuna/<study>.db (SQLite)
OPTUNA_DIRECTION       = 'maximize'      # 'maximize'(hit-rate) | 'minimize'(loss)
OPTUNA_OBJECTIVE_SPLIT = 'val'           # 目的指標を読む split。'test' はリークになるので 'val' 推奨
OPTUNA_OBJECTIVE_KEY   = 'top1_position_hit_rate_pct'  # run_summary.json[split] のキー
OPTUNA_SEED            = 42              # 探索中の学習シード（固定して構成の効果を分離）
# pruning を効かせるには EARLY_STOPPING_METRIC を hit-rate、EARLY_STOPPING_MODE='max'、
# OPTUNA_DIRECTION='maximize' に揃えること（方向が一致するときのみ中間報告する）。

# --- Ablation実験用マスク設定 ---
# Trueにすると、その数値特徴量を強制的に0にして学習・推論を行う
ABLATION_MASKS = {
    # --- 既存の物理化学的特徴量 ---
    'FREQ': False,       # 変異頻度を無効化
    'HYDRO': False,      # 疎水性差 (Eisenberg-Weiss) を無効化
    'CHARGE': False,     # 電荷差を無効化
    'SIZE': False,       # サイズ差を無効化
    'BLSM': False,       # BLOSUM62差を無効化
    'PAM250': False,     # PAM250スコアを無効化
    # --- ホスト適応特徴量 (SCV2_host_adaptation_features.csv) ---
    'HOST_LOG_RATIO_DIFF': False,        # host_distance_log_ratio_diff
    'HOST_LOG_RATIO_BEFORE': False,      # host_distance_log_ratio_before
    'HUMAN_RSCU_DIFF': False,            # human_RSCU_diff
    'SCV2_RSCU_DIFF': False,             # SCV2_RSCU_diff
    'OPTIMAL_TO_OPTIMAL': False,         # optimal_to_optimal (2値)
    'NON_OPTIMAL_TO_OPTIMAL': False,     # non_optimal_to_optimal (2値)
    'OPTIMAL_TO_NON_OPTIMAL': False,     # optimal_to_non_optimal (2値)
    'IS_TRANSITION': False,              # is_transition (2値)
    'TRANSITION_HUMAN_RSCU_DIFF': False, # transition_human_RSCU_diff
    'TRANSITION_SCV2_RSCU_DIFF': False,  # transition_SCV2_RSCU_diff
    'CPG_DIFF': False,                   # CpG_diff
    'UPA_DIFF': False,                   # UpA_diff
    'HUMAN_RSCU_BEFORE': False,          # human_RSCU_before
    'SCV2_RSCU_BEFORE': False,           # SCV2_RSCU_before
    'HUMAN_FREQ_BEFORE': False,          # human_freq_before
    'HUMAN_FREQ_DIFF': False,            # human_freq_diff
    'SCV2_FREQ_BEFORE': False,           # SCV2_freq_before
    'SCV2_FREQ_DIFF': False,             # SCV2_freq_diff
    'HUMAN_CAI_BEFORE': False,           # human_CAI_before
    'HUMAN_CAI_DIFF': False,             # human_CAI_diff
    'SCV2_CAI_BEFORE': False,            # SCV2_CAI_before
    'SCV2_CAI_DIFF': False,              # SCV2_CAI_diff
    'HOST_RSCU_RATIO_BEFORE': False,     # host_distance_RSCU_ratio_before
    'HOST_RSCU_RATIO_DIFF': False,       # host_distance_RSCU_ratio_diff
    # --- パス内累積変異カウント ---
    'CUM_SYN': False,                    # 現タイムステップまでのシノニマス変異累積数
    'CUM_NONSYN': False,                 # 現タイムステップまでのノンシノニマス変異累積数
    # --- 亜系統の流行ダイナミクス特徴（USE_LINEAGE_GROWTH_FEATURES=True 時のみ有効）---
    'LINEAGE_LOG_COUNT_RECENT': False,   # 直近K週の件数(規模の因果版)
    'LINEAGE_GROWTH_RATE': False,        # 直近K週の log件数 の傾き（勢い）
    'LINEAGE_REL_GROWTH_ADV': False,     # 直近K週の logitシェア の傾き（相対成長優位）
    'LINEAGE_GROWTH_ACCEL': False,       # growth_rate の加速度
    # --- 前後塩基コンテキスト（CONTEXT_WINDOW 分の各位置を個別制御）---
    'MUTATION_CONTEXT': False,     # 全コンテキスト塩基を一括無効化
        'MUTATION_CONTEXT_L1': False,  # 1文字前の塩基を無効化
        'MUTATION_CONTEXT_L2': False,  # 2文字前の塩基を無効化
        'MUTATION_CONTEXT_L3': False,  # 3文字前の塩基を無効化
        'MUTATION_CONTEXT_L4': False,  # 4文字前の塩基を無効化
        'MUTATION_CONTEXT_L5': False,  # 5文字前の塩基を無効化
        'MUTATION_CONTEXT_L6': False,  # 6文字前の塩基を無効化
        'MUTATION_CONTEXT_L7': False,  # 7文字前の塩基を無効化
        'MUTATION_CONTEXT_R1': False,  # 1文字後の塩基を無効化
        'MUTATION_CONTEXT_R2': False,  # 2文字後の塩基を無効化
        'MUTATION_CONTEXT_R3': False,  # 3文字後の塩基を無効化
        'MUTATION_CONTEXT_R4': False,  # 4文字後の塩基を無効化
        'MUTATION_CONTEXT_R5': False,  # 5文字後の塩基を無効化
        'MUTATION_CONTEXT_R6': False,  # 6文字後の塩基を無効化
        'MUTATION_CONTEXT_R7': False,  # 7文字後の塩基を無効化
}

# --- データロード設定 ---
USE_UNIQUE_FILTER = True        # 学習時にパス重複(raw_path)を除外するか（ユニーク化）

# --- 流行度設定 (株別サンプル数ベース) ---
USE_EVAL_STRENGTH_FILTER = False     # 評価時に流行度フィルタを適用 (Falseでカテゴリ分析を使用)
EVAL_STRENGTH_MIN = 0.0              # 評価時フィルタのしきい値 (0.0=フィルタなし)
# 流行度は log(1 + sample_count) で計算される
STRENGTH_SCORE_FROM_CSV = True


# 流行度カテゴリの閾値設定 (log(1+x)スケール)
# DYNAMIC_STRENGTH_CATEGORY = True にすると、DB全体の分布に基づいて動的に閾値を計算します。
# DYNAMIC_STRENGTH_CATEGORY = False の場合は、下の固定値（STRENGTH_CATEGORY_LOW_MAX / MED_MAX）が使用されます。
DYNAMIC_STRENGTH_CATEGORY = True    # 動的閾値を使用するかどうか
STRENGTH_CATEGORY_LOW_MAX = 6.0      # 小: 0 ~ 6.0 (サンプル数 ~400)
STRENGTH_CATEGORY_MED_MAX = 10.0     # 中: 6.0 ~ 10.0 (サンプル数 ~20,000)
                                     # 大: 10.0~ (サンプル数 20,000+)

# --- 学習時 感染規模フィルタ設定 ---
# 感染規模が小さい株を学習から除外することで、より流行した変異に特化した学習が可能
# ※ 評価 (valid/test) には適用されない。学習データ (split_type=0) のみフィルタされる
USE_TRAIN_STRENGTH_FILTER = False    # 学習時に感染規模フィルタを適用するか
TRAIN_STRENGTH_MIN = 6.0             # 学習に使用する最小感染規模 (log(1+n)スケール)
# 小を除く  : STRENGTH_CATEGORY_LOW_MAX と同じ値 (6.0) を設定
# 小中を除く: STRENGTH_CATEGORY_MED_MAX と同じ値 (10.0) を設定


# --- パス設定 ---
DATA_BASE_DIR = '../usher_output/'
# DATA_BASE_DIR に加えて取り込む追加のUsher出力ディレクトリ（同一の <lineage>/<N>/ 構造が前提）。
# usher/README.md の新規サンプル配置パイプラインで生成した usher_output2 を追加。
EXTRA_DATA_BASE_DIRS = ['../usher_output2/']
CODON_CSV         = "reference/codon/codon_mutation4.csv"
FREQ_CSV          = "reference/table_set.csv"
DISSIMILARITY_CSV = "reference/aa_properties/dissimilarity_metrics.csv"
PAM250_CSV        = "reference/aa_properties/PAM250.csv"
SEQUENCES_CSV     = "reference/sequences-241017.csv"  # 株別サンプル数CSV
# SEQUENCES_CSV に加えて取り込む追加のサンプル一覧CSV（同一スキーマが前提）。
# usher_output2 に対応する新規サンプル一覧。
EXTRA_SEQUENCES_CSV = ['reference/sequences-241017_2.csv']
HOST_ADAPTATION_CSV  = "reference/codon/SCV2_host_adaptation_features.csv"

# --- Point-in-time 変異頻度（リーク修正）---
# FREQ_CSV(reference/table_set.csv)はデータセット全期間から静的に1回だけ計算されており、
# walk-forward等の時系列評価ではtest窓より未来の変異頻度情報が特徴量(codon_freq, x_num[...,0])
# に混入する（時系列データリーク）。True にすると、db/dataset.py の _process_sample が
# config.TEMPORAL_SPLIT_DATE（walk-forwardフォールドのsplit_date、assign_fold_test_window等で
# 設定済み）に対応する reference/point_in_time_freq/{TEMPORAL_SPLIT_DATE}.csv
# （scripts/preprocess/generate_point_in_time_freq.py で事前生成）を使って codon_freq を
# 動的に上書きする（DB再構築は不要。ABLATION_MASKSと同じデータロード時オーバーライド方式）。
# 該当ファイルが無い場合は警告を出しFREQ_CSV由来の値のまま（フォールバック、既定動作と同一）。
USE_POINT_IN_TIME_FREQ = False
POINT_IN_TIME_FREQ_DIR = "reference/point_in_time_freq"

OUTPUT_DIR = 'outputs/transformer_260817/'
MODEL_SAVE_DIR = OUTPUT_DIR + 'models'
RESULT_SAVE_DIR = OUTPUT_DIR + 'results'
# 実験タイプのサブディレクトリ名（例: 'feature', 'date', 'sampling'）
# 空文字の場合は results/<timestamp>/ に直接保存
EXPERIMENT_NAME = ''
INCREMENTAL_CACHE_DIR = 'cache/incremental_features'

# --- DuckDB データベース設定 ---
DB_DIR = 'db'
DB_FILE = None  # データベースファイルを直接指定する場合はパスを設定 (例: 'db/features.duckdb')。Noneの場合は自動生成
USE_DB = True  # Trueの場合、pickleキャッシュの代わりにDuckDBを使用
MIN_SEQ_LEN = 10  # 最小シーケンス長

# --- キャッシュ・効率化設定 ---
CACHE_DIR = 'cache/dataset'
BATCH_SIZE_FEATURE_GEN = 5000
FORCE_REPROCESS = False
ENABLE_LRU_CACHE = True
ENABLE_PARALLEL_PROCESSING = True
CACHE_MAX_SIZE = 10000
NUM_PREPROCESS_WORKERS = 4    # 前処理時の並列ワーカー数上限（strain単位でチャンク処理するためワーカーあたりメモリは有界、28コア環境で4に設定済み）
PREPROCESS_CHUNK_SIZE = 1000    # 前処理ワーカー内でのチャンクバッファサイズ (小さくして省メモリ化)
DB_WRITE_BATCH_SIZE = 10000     # DB書き込み時のバッチサイズ
WEIGHT_DECAY = 0.01

# --- 保存・ログ設定 ---
SAVE_PREDICTIONS = False        # 予測結果の詳細をファイルに出力するか（数100MB の CSV 書き込みのため通常は False）
SAVE_STRAIN_INFO = True         # 使用した株の情報を保存するか（USE_DB=False 時のみ有効）
SAVE_STRAIN_METRICS = False     # 株単位の精度CSV（1万株×複数指標で巨大、詳細分析時のみ）

# --- データセット設定 ---
MAX_SEQ_LEN = 39
TARGET_LEN = 1
TRAIN_MAX = 40 # TS:1-40を学習に利用(TRAIN_MAX > MAX_SEQ_LEN + TARGET_LEN)
VALID_NUM = 3
MAX_CO_OCCURRENCE = 20  # 最大共起数上限（Omicron等に対応するため5から20へ緩和）
EVAL_MAX_Y_CO_OCCURRENCE = 5  # 評価時に過剰な情報漏洩を防ぐためのターゲット共起上限
# MAX_CO_OCCURRENCEが「1サンプル内で同時発生した変異数」の上限なのに対し、こちらは
# 「同一input_path_str（分岐親履歴）を共有し1グループとしてany-of-set統合されるサンプル数」
# の上限。Omicron期は1つの祖先枝に大量のサンプルがぶら下がる巨大グループが生じ
# （fold_3 test最大14,684、fold_4 train最大13,871人実測、2026-07-30/31）、
# db/dataset.py:_build_group_label_cache がこの全メンバー分のラベルをキャッシュするため、
# 学習用DataLoaderのpersistent workerで長時間かけてCopy-on-Write共有が崩れ実質的に
# worker数分重複し、OOMの一因となった。これを超えるグループはメンバーを決定的に
# （代表サンプルIDをseedに）ランダムサブサンプリングしてキャッシュ・Soft Target分配・
# any-of-set評価の対象から間引く（代表サンプル自身は常に残す）。Noneで無効化。
MAX_GROUP_MEMBERS_FOR_CACHE = 4000
RAW_PATH_TRUNCATE_LEN = None  # samples.raw_path 保存時の文字数上限。None=切り詰めなし（完全な履歴文字列を保存、正確性優先）
VALID_RATIO = 0.2

# --- サンプリングモード設定 ---
# (両モード共通) 頻出上位 MAX_STRAIN_NUM 株のデータのみを使用対象とする
MAX_STRAIN_NUM = 10000      # 共通: 使用する上位株数

# 'proportional': 対象株のデータ比率を維持しつつ、合計 MAX_NUM 件になるように抽出
# 'fixed_per_strain': 対象の各株からそれぞれ最大 MAX_NUM_PER_STRAIN 件ずつ抽出
SAMPLING_MODE = 'proportional'

# モードA: 比率サンプリング用 (SAMPLING_MODE = 'proportional')
MAX_NUM = 10000000  # 全体での抽出合計サンプル数の上限

# モードB: 各株の上限数制限用 (SAMPLING_MODE = 'fixed_per_strain')
MAX_NUM_PER_STRAIN = 1000000   # 各株ごとの最大抽出サンプル数
# 下限値MIN_NUM_PER_STRAIN追加？

# --- ボキャブラリー設定 ---
BASE_VOCABS = {'A':1, 'T':2, 'C':3, 'G':4, 'N':5, 'n':6, 'PAD':0}
AA_VOCABS = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8,
             'H':9, 'I':10, 'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16,
             'T':17, 'W':18, 'Y':19, 'V':20, '*':21, 'n':22, 'PAD':0}
PROTEIN_VOCABS = {
    'non_coding1':1, 'nsp1':2, 'nsp2':3, 'nsp3':4, 'nsp4':5,
    'nsp5':6, 'nsp6':7, 'nsp7':8, 'nsp8':9, 'nsp9':10,
    'nsp10':11, 'nsp12':12, 'nsp13':13, 'nsp14':14,
    'nsp15':15, 'nsp16':16, 'non_coding2':17, 'S':18, 'non_coding3':19,
    'ORF3a':20, 'non_coding4':21, 'E':22, 'non_coding5':23, 'M':24,
    'non_coding6':25, 'ORF6':26, 'non_coding7':27, 'ORF7a':28, 'ORF7b':29,
    'non_coding8':30, 'ORF8':31, 'non_coding9':32, 'N':33, 'non_coding10':34,
    'ORF10':35, 'non_coding11':36, 'PAD':0
}

# --- モデルアーキテクチャ設定 ---
CONTEXT_WINDOW = 5                         # 前後コンテキスト窓幅（各方向）。3/5 で切り替え可(7はDBの再構築必要)
NUM_FEATURE_STRING = 9 + 2 * CONTEXT_WINDOW  # 固定9 + 両側コンテキスト

# --- 亜系統の流行ダイナミクス特徴（因果的な成長率＝適応度代理）---
# 各サンプル(系統×収集日)に、収集日以前のデータだけで計算した4値を数値特徴に追加する。
#   lineage_log_count_recent / lineage_growth_rate / lineage_rel_growth_adv / lineage_growth_accel
# epidemic 件数は系列メタCSV(Pangolin × Collection_Date)から週次で構築。詳細は db/growth_features.py。
# True にすると NUM_CHEM_FEATURES が +4 され設定ハッシュが変わる → DB 再構築が必要。
USE_LINEAGE_GROWTH_FEATURES = True
GROWTH_WINDOW_WEEKS         = 4        # 成長率の傾きを取る直近ウィンドウ幅（週）
GROWTH_LINEAGE_CSV_COLUMN   = 'Pangolin'         # SEQUENCES_CSV の系統列
GROWTH_DATE_CSV_COLUMN      = 'Collection_Date'  # SEQUENCES_CSV の収集日列

NUM_CHEM_FEATURES = 32 + (4 if USE_LINEAGE_GROWTH_FEATURES else 0)

VOCAB_SIZE_POSITION = 30006
VOCAB_SIZE_BASE = 7
VOCAB_SIZE_AA = 23
VOCAB_SIZE_CODON_POS = 6
VOCAB_SIZE_AA_POS = 10001
VOCAB_SIZE_SYNONYMOUS = 2      # 0: ノンシノニマス, 1: シノニマス
NUM_REGIONS = 37

EMBED_DIM_POS = 128
EMBED_DIM_BASE = 32
EMBED_DIM_AA = 64
EMBED_DIM_REGION = 128
EMBED_DIM_CODON_POS = 16
EMBED_DIM_AA_POS = 128
EMBED_DIM_SYNONYMOUS = 8  # シノニマス用Embedding (2クラス: 0=非同義, 1=同義)

FEATURE_DIM = 256
HIDDEN_DIM = FEATURE_DIM
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1

# --- Conv1D局所特徴抽出の有無 ---
# 実験結果: 精度向上なし。CONTEXT_WINDOW による前後塩基埋め込みで局所文脈は十分カバー済み。
# Trueの場合: Conv1D層を使用して局所的な文脈情報を抽出
USE_LOCAL_CONV1D = False
LOCAL_CONTEXT_KERNEL_SIZE = 3

# --- Origin Attentionの有無 ---
# 実験結果: 精度向上なし。武漢株との差分は変異特徴量として既に入力に含まれている。
# Trueの場合: 原点（Wuhan株）を常に参照するCross-Attentionを使用
USE_ORIGIN_ATTENTION = False
ORIGIN_ATTENTION_HEADS = 4  # Origin Attentionのヘッド数

# --- Temporal Pooling モード（Ablation Study用） ---
# Transformer Encoderの全タイムステップ出力 [B, T, F] から
# どの位置を予測ヘッドへの入力として使うかを制御する
#
#   'last': x[:, -1, :]  末尾固定（デフォルト）
#           ※ PADトークンが末尾に来ている場合は無効な表現を読む可能性あり
#
#   'mean': Global Average Pooling（PADを除外した重み付き平均）
#           全タイムステップの情報を均等にブレンドする
#           src_key_padding_mask が None の場合は単純平均にフォールバック
#
#   'cls' : [CLS]トークン（BERT方式）
#           シーケンス先頭に学習可能な専用トークンを追加し、
#           Self-Attentionを通じてシーケンス全体の情報が集約された
#           [CLS]の出力表現 x[:, 0, :] を予測ヘッドに使う。
#           padding_mask は CLS トークン分(+1)自動的にずらされる。
TEMPORAL_POOLING = "last"  # 'last' | 'mean' | 'cls'

# --- Shared Trunk（予測ヘッドの共通中間層）設定（Ablation Study用） ---
# Trueの場合: latest_context → Shared Trunk → 各ヘッド (特徴の再利用で汎化係数減少)
# Falseの場合: latest_context → 各ヘッドが並列に直接受け取る（現状）
USE_SHARED_TRUNK = False

# --- Co-occurrence Attention 設定 ---
# ① CO_ATTN_DIM: 集約の内部次元。FEATURE_DIM と同値なら in_proj/out_proj は不要（デフォルト推奨）
#    CO_ATTN_DIM は CO_ATTN_N_HEADS で割り切れる必要あり
# ② CO_ATTN_N_LAYERS:
#    1 = Cross-Attention のみ（学習クエリによる重み付き平均）
#    2 = Self-Attention（変異間相互作用）→ Cross-Attention（集約）← 現在値・推奨
#    3 以上 = Self-Attention を N-1 層重ねてから Cross-Attention 集約
CO_ATTN_N_HEADS  = N_HEADS        # 共起 Attention のヘッド数（デフォルト = N_HEADS）
CO_ATTN_N_LAYERS = 1             # 共起 Attention の層数（2: Self-Attn + Cross-Attn）
CO_ATTN_DIM      = FEATURE_DIM   # 共起 Attention の内部次元（デフォルト = FEATURE_DIM）

# ③ USE_FLAT_COATTN: 共起集約をスキップし、全変異を独立トークンとして Transformer に渡す
#    [B, T, C, F] → [B, T*C, F]
#    T*C = 39*20 = 780 トークン/サンプルになるため BATCH_SIZE を大幅に下げる必要あり（例: 32〜64）
USE_FLAT_COATTN = False

# FLAT_COATTN_1D_PE: USE_FLAT_COATTN=True のときの位置エンコーディング方式
#   False: 2D PE（タイムステップ次元 + 共起内インデックス次元）← モデルが共起グループを識別できる
#   True : 1D PE（正弦波・通し番号）← 共起情報を持たない純粋な時系列扱い = PETra相当
FLAT_COATTN_1D_PE = False

# --- Broadcast-back Cross-Attention（USE_FLAT_COATTN=False時のみ意味を持つ） ---
# Co-occurrence Attentionでタイムステップ内の共起変異集合を1本のベクトルへ集約した後、
# 時系列Transformer Encoderは代表ベクトル同士でしかAttentionを取らないため、「ある共起
# 変異群の1つの変異」と「他タイムステップの変異」間の個別粒度でのAttentionが構造的に
# 取れない（集約後は代表ベクトルしか残らないため）。この対策として、時系列Transformer
# Encoder適用後の各タイムステップ代表ベクトル r'_t を Key/Value、集約前の個々の変異
# embeddingをQueryとするCross-Attentionを追加し、その出力をCoOccurrenceAttentionで
# 再集約した r''_t を、既存の latest_context への残差として学習可能ゲート付きで加算する。
# USE_FLAT_COATTN=True時は全変異が最初から相互Attentionを取れるため無関係（no-op）。
# 既定Offのため無効時の挙動は完全に同一。
# 2026-08-04: walk_forward全7fold本実行で効果検証のためTrueに設定（USE_COATTN_FREQUENCY_
# PENALTY=Trueのwalk_forward 20260802_082112とはこのフラグのみが差分の比較用run）。
# 2026-08-06: 上記A/B比較（20260802_082112 vs 20260805_034330）の結果、R-Precision全体で
# position -0.015pt（実質無変化）、region -0.367pt/codon_pos -0.165pt/synonymous -0.544pt
# とむしろ悪化。fold_3（Omicron early、本来この機構で改善したかった巨大共起グループの
# 期間）でもposition 4.76%→4.30%と悪化しており、狙った効果は確認できなかったため
# 既定をFalseに戻す。ゲート（0初期化）が15epoch×foldの学習量では有効な信号を
# 獲得しきれなかった可能性がある。
USE_BROADCAST_BACK_ATTENTION = False
BROADCAST_BACK_ATTENTION_HEADS = None  # Noneの場合 N_HEADS を使用

# --- 訓練設定 ---
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
EPOCHS = 15              # Soft Target収束のため余裕を持たせる
TOP_K_EVAL = 1 # Top-5でのRecallなども見たい場合はここを変更

# --- DataLoader 並列化（学習高速化）---
# 学習は GPU 計算ではなく DuckDB ストリーミング＋per-sample の Python 処理が律速なため、
# ワーカーを増やしてデータ前処理を GPU 計算とオーバーラップさせると時短できる。
# メモリ増は「ワーカー数 × (sample_ids のスライス + prefetch バッファ + DuckDB 接続)」程度で、
# データ総量には比例しない。0 にすると従来のメインプロセス同期読み込みに戻る。
NUM_DATALOADER_WORKERS        = 8      # 0=同期（従来）。実測で 8 が ~2.0x（4 は ~1.3x, 28コア環境）。CPU/メモリと要相談
# val/test（評価用）ローダーは学習用ほどスループットが律速要因にならない一方、
# run_final_evaluation で val_loader・test_loader が同時に生存し続けるため、
# worker数はそのままメモリ負荷に直結する。walk_forward fold_3 OOM調査（2026-07-28）を
# 受け、評価用ローダーは学習用の半分のworker数に絞る。
EVAL_NUM_DATALOADER_WORKERS   = max(1, NUM_DATALOADER_WORKERS // 2)
# 学習用ローダーは DATALOADER_PERSISTENT_WORKERS=True で同一worker群が全epoch（15epoch×
# 数百〜千バッチ）にわたり生存し続ける。_group_label_cache はfork前にメインプロセスで
# 1回だけ構築しCopy-on-Writeで全workerに共有される設計だが、CPythonは「読むだけ」でも
# オブジェクトの参照カウント書き込みが発生するため、バッチを処理するたびに少しずつ
# ページがコピーされ、長時間かけて実質的に全worker分重複してしまう。巨大共起グループ
# （fold_4のtrainで最大13,871人実測）を含むsplitでは、1 workerあたり約6GBまで肥大化し
# 8 worker合計・評価用待機workerとの合算でOOMした（2026-07-31実測）。MAX_GROUP_MEMBERS_
# FOR_CACHEでキャッシュ自体を縮小しつつ、worker数も絞って複製の総量を抑える。
TRAIN_NUM_DATALOADER_WORKERS  = max(1, NUM_DATALOADER_WORKERS // 2)
DATALOADER_PREFETCH_FACTOR    = 2      # 各ワーカーが先読みするバッチ数（num_workers>0 時のみ有効）
DATALOADER_PERSISTENT_WORKERS = True   # エポック間でワーカーを再利用（num_workers>0 時のみ有効）
DATALOADER_PIN_MEMORY         = True   # host→GPU 転送を高速化（CUDA 時のみ効果）
# 各ワーカーの DuckDB 接続のスレッド数上限。None=ワーカー時は自動で 2、メイン(0)時は DuckDB 既定。
# ワーカー×全コアのスレッド張りすぎ（オーバーサブスクライブ）を防ぐため小さめ推奨。
DATALOADER_DUCKDB_THREADS     = None

LOSS_WEIGHT_REGION = 0.14         # 基因領域の損失重み
LOSS_WEIGHT_POSITION = 0.7        # 塩基位置の損失重み
LOSS_WEIGHT_AA_POS = 0.10         # アミノ酸配列位置の損失重み
LOSS_WEIGHT_CODON_POS = 0.04      # コドン位置の損失重み (1,2,3の3クラス)
LOSS_WEIGHT_SYNONYMOUS = 0.02    # シノニマス/ノンシノニマスの損失重み (2クラス)
LOSS_WEIGHT_STRENGTH = 0.02       # 流行度予測の損失重み (回帰)

# --- 高度な学習設定 (追加) ---
# Label Smoothing: Soft Target + FocalLoss モードでは Hard Targetのみ有効なため False
# Hard Target モード (HYBRID_ALPHA=0.0) に切り替える場合は True に戻すこと
USE_LABEL_SMOOTHING = False
LABEL_SMOOTHING_FACTOR = 0.1

# --- Soft/Hard/Hybrid Target 学習設定 (260417 新規) ---
# HYBRID_ALPHA で学習モードを一本管理する（USE_SOFT_TARGET は廃止）
#
#   HYBRID_ALPHA = 1.0 : 純粋 Soft Target
#                        loss = CE(soft_labels, pred)
#                        グループ化・確率分配あり / Hard CE なし
#
#   HYBRID_ALPHA = 0.0 : 純粋 Hard Target
#                        loss = CE(hard_labels, pred)
#                        グループ化なし（collate_fn がシンプルになる）
#
#   0 < HYBRID_ALPHA < 1 : ハイブリッド (Knowledge Distillation 的アプローチ)
#                        loss = α × CE(soft_labels, pred)
#                             + (1-α) × CE(hard_labels, pred)
#                        グループ化あり・両 CE を同時計算
#
HYBRID_ALPHA = 1.0

# Soft Target の Temperature パラメータ (HYBRID_ALPHA > 0 の場合のみ有効)
# 生成した確率分配ベクトルに指数変換を施すことで分布の鋭さを調整する
#   T = 1.0 : 変化なし（デフォルト、1/K × 1/M の均等分配そのまま）
#   T < 1.0 : よりシャープ（頻出変異への集中を強化 → 分類シグナルを強める）
#   T > 1.0 : より均一（ラベルスムージング的効果 → 正則化が強まる）
SOFT_TARGET_TEMPERATURE = 1.0

# --- 損失関数設定 ---
# 実験結果: CE が最も精度が高く安定。cbce・focal は精度改善なし。
#   'ce'      : 通常のCrossEntropyLoss（推奨）
#   'cbce'    : Class-Balanced CrossEntropyLoss
#   'cb_focal': Class-Balanced + FocalLoss（大規模データで再試する場合）
# 選択肢: 'ce', 'wce', 'cbce', 'focal', 'cb_focal'
LOSS_FUNCTION_TYPE = 'ce'
FOCAL_LOSS_GAMMA = 1.0           # focal/cb_focal用
FOCAL_LOSS_ALPHA = None          # focal専用 (cb_focalではCB重みをalphaとして自動使用)
CLASS_BALANCED_BETA = 0.9999     # Class-Balanced用パラメータ (通常0.99〜0.9999)
NORMALIZE_LOSS_WEIGHTS = False    # 頻度が少ないクラスでの損失爆発を防ぐため重みを正規化する

# Scheduler: 学習率の自動調整 (Trueで適用)
USE_SCHEDULER = True
SCHEDULER_ETA_MIN = 1e-6  # 最小学習率

# --- Warmup スケジューラ (LinearWarmup + CosineAnnealingLR) 設定 ---
# True の場合: 学習強度を WARMUP_EPOCHS エポックかけて左影 0 → LEARNING_RATE に抜た後、
#              CosineAnnealing で SCHEDULER_ETA_MIN へ㠻閏す
# False の場合: 従来通り CosineAnnealingLR のみ
USE_WARMUP = True   # True | False
WARMUP_EPOCHS = 2   # ウォームアップのエポック数

# Weights & Biases: 実験可視化 (Trueで適用)
USE_WANDB = False
WANDB_OFFLINE = True  # Trueにするとオフラインモード (長時間処理時の接続切れ対策)
WANDB_PROJECT_NAME = "viral-mutation-transformer"
WANDB_RUN_NAME = None # Noneなら自動生成 (例: "run_20251121_...")

# MultiTaskLoss 設定
# Trueの場合、config.LOSS_WEIGHT_REGION/POSITION は無視され、自動調整される
USE_MULTITASK_LOSS = True

# --- 学習安定化・Optimizer設定 (新規追加) ---
MAX_GRAD_NORM = 1.0              # 勾配クリッピングの値 (Noneで無効化)
OPTIMIZER_BETA1 = 0.9            # AdamW の Beta1
OPTIMIZER_BETA2 = 0.999          # AdamW の Beta2

# --- Early Stopping 設定 ---
# 【学習損失 vs モデル選択基準の分離】
# 学習損失は CE (Soft Target) のまま維持する。R-Precision は離散指標のため微分不可能であり、
# 勾配を流すには ListNet 等のサロゲート損失が必要になるが複雑化に見合わない。
# Soft Target + CE はすでに R-Precision と方向が一致している：
#   共起変異全体に確率質量を均等分配 → 全正解変異を高くランク付け → R-Precision が上がる。
#
# 一方、モデル選択（Early Stopping）の基準を "val_r_precision" にすると、
# 学習損失を変えずに「実際に評価したい指標が最も高いエポック」のモデルを残せる。
# ただし R-Precision はエポック間でノイジーになりやすいため、
# PATIENCE を val_loss 時より大きめに設定することを推奨する（例: 10）。
#
#   val_loss : 滑らか・安定。収束の検出に優れるが、hit-rate と乖離することがある
#              （例: MLM は hit-rate を上げるが loss は悪化）。評価指標で選びたいなら hit-rate を使う。
#
# hit-rate 等を対象にする場合（エポックごとに evaluate() が val_metrics に計算する値。MODE='max' 必須）:
#   タスク {region, position, aa_pos, codon_pos, synonymous} × 指標
#     {hit_rate, macro_recall, weighted_recall, f1, precision, recall} を 'val_<task>_<指標>' で指定可。
#   例: 'val_position_hit_rate'（塩基位置 Top-1、最弱タスク・伸びしろ大）
#       'val_position_macro_recall'（クラス平均リコール。稀な位置も平等に評価）
#       'val_region_hit_rate' / 'val_aa_pos_hit_rate' 等
#   ※ 'val_' 接頭辞は自動で外して val_metrics のキーに突き合わせる（num_samples 重み付き平均）。
#   ※ top-k / r_precision はエポックごとには未計算のため早期終了には使えない（最終評価のみ）。
#   ※ 複合指標（region+position の重み付き和）は今後追加予定。
#   ※ 存在しないキーを指定すると WARNING を出して val_loss にフォールバックする（事故防止）。
EARLY_STOPPING_METRIC = "val_loss"  # 上記の 'val_<task>_<指標>' も指定可
EARLY_STOPPING_MODE = "min"         # 'min' (loss等) | 'max' (hit-rate 等、大きいほど良い)
EARLY_STOPPING_PATIENCE = 5   # [260417] エポック増加に合わせて余裕を持たせる

# --- 評価・可視化のハードコード解除 (新規追加) ---
EVAL_TOP_KS = (1, 3, 5)             # Top-K 評価で計算する K のリスト

# --- 高確信サブセット評価（C-4: 多義サンプルに引きずられない実力を見る）---
# Top-1 予測確率でサンプルを降順ソートし、確信度上位の各カバレッジ割合における
# Top-1 hit-rate（Recall@1 相当）と平均確信度を出力する。1.0=全体（従来指標と一致）。
# 選択的予測（abstention）で「自信のある予測だけ出す」運用時の精度見積りに使う。
SAVE_CONFIDENT_SUBSET     = True
CONFIDENT_COVERAGE_LEVELS = [1.0, 0.75, 0.5, 0.25, 0.1]
PLOT_TOP_N_POSITIONS = 40           # 塩基位置の Recall でプロットする上位 N 件
DIVERSITY_PLOT_MIN_SAMPLES = 10     # 多様度 vs hit-rate 散布図/CSV で系統を採用する最小サンプル数
SAVE_ATTENTION_HEATMAP = False      # Attentionヒートマップを保存するかどうか（可視化のみ・学習時間に影響）

# --- プロット出力の個別トグル ---
# 各プロットを出力するかを個別に制御する。キーが無い/True なら出力（デフォルト＝全出力で従来動作）。
# 特定のプロットだけ止めたいとき False にする。重い可視化（tsne / network 系）を切ると高速化できる。
# 注: 'attention_heatmap' は SAVE_ATTENTION_HEATMAP と AND 条件（どちらも True のときのみ出力）。
PLOT_OUTPUTS = {
    # 学習曲線・評価メトリクス系
    'training_curve': True,
    'metrics_by_timestep': True,
    'category_metrics': True,
    'strength_calibration': True,
    'per_position_recall': True,
    'strength_fine': True,
    'lineage_metrics': True,
    'entropy_vs_accuracy': True,
    'labeldiversity_vs_accuracy': True,   # エントロピー代替: ユニーク率(ユニーク位置/全位置)で hit-rate を見る図
    'simpson_vs_accuracy': True,          # エントロピー代替: Simpson 多様度(生態学標準)で hit-rate を見る図
    'entropy_bin': True,
    'region_metrics': True,
    'calibration': True,
    'combined_val_test_metrics': True,
    'combined_category_comparison': True,
    'metrics_by_date': True,
    'topk_precision': True,
    'r_precision': True,
    # 可視化系（重い）
    'mutation_dist': True,
    'attention_heatmap': True,
    'base_embedding_network': True,
    'position_threshold_network': True,
    'position_tsne': True,
    'position_cluster_network': True,
    'major_mutation_network': True,
}

# --- strength ヘッドの損失関数 ---
# 'mse'  : 均乗誤差（現状・外れ値に弱い）
# 'mae'  : 平均絶対誤差（外れ値に強健）
# 'huber': Huber Loss（MAEとMSEの中間、delta内はMSE、delta外はMAE）
STRENGTH_LOSS_TYPE = 'mae'  # 'mse' | 'mae' | 'huber'
STRENGTH_HUBER_DELTA = 1.0  # Huber Lossの間屠パラメータ (huber選択時のみ有効)

# --- ECE (期待キャリブレーション誤差) 設定 ---
# True の場合: evaluate() 内で ECE を計算し、final_metrics_by_ts に追加する
# 安定性のため n_bins 分割した単純隣値分割方式 (Equal-Width Binning)
SAVE_ECE = False       # True | False（学習ループ内で全サンプル計算するため重い。分析時のみ True に）
ECE_N_BINS = 10        # ビン数 (10 が標準定義)

# ============================================================
# --- 精度向上実験設定 ---
# ============================================================

# --- Curriculum Learning ---
# 学習初期は短いパスのみを使い、徐々に長いパスを追加する。
# 初期エポックは正解シグナルが明確な短いパスで基礎を学習し、
# CURRICULUM_WARMUP_EPOCHS エポック後に全長のパスを解禁する。
CURRICULUM_LEARNING = False
CURRICULUM_MAX_LEN_INIT = 5     # 最初のエポックで使う最大パス長（短いパスのみ）
CURRICULUM_WARMUP_EPOCHS = 5    # このエポック数をかけて上限を TRAIN_MAX まで線形拡大

# --- Relative Position Encoding (ALiBi) ---
# 正弦波の絶対位置エンコーディングを ALiBi スタイルの相対位置バイアスに置き換える。
# ALiBi: bias[h,i,j] = -m_h × |i-j|  (学習パラメータなし、外挿性能が高い)
# True の場合: PositionalEncoding をスキップし、ALiBi バイアスを attn_mask に注入
# False の場合: 従来の正弦波 PositionalEncoding を使用
USE_RPE = False

# --- Biologically Informed Loss（位置近傍スムージング）---
# 塩基位置・AA位置タスクで正解クラスの周辺にガウス分布で確率を分散させた
# soft target を使う。例: 正解位置 21846 → 21844〜21848 も小さく正解扱い。
# Soft Target モード (HYBRID_ALPHA > 0) でのみ有効。
# Hard Target モードでは label_smoothing で代替すること。
USE_BIO_INFORMED_LOSS = False
BIO_INFORMED_SIGMA = 2.0              # ガウスの標準偏差（クラス単位）
BIO_INFORMED_RADIUS = 10              # 計算範囲（±radius クラスのみ処理・高速化）
BIO_INFORMED_TASKS = ['position', 'aa_pos']  # 適用タスク

# --- Test-Time Augmentation（TTA / MC Dropout）---
# 推論時も Dropout を有効にして TTA_N_PASSES 回推論し softmax を平均する。
# 単一推論より分散が減り、確信度が高いときだけ Top-K 予測が変わる。
# True の場合: evaluate() / evaluate_topk() で TTA を適用
# False の場合: 通常の eval モード推論（従来動作）
USE_TTA = False
TTA_N_PASSES = 5    # 推論回数（多いほど安定するが推論時間が N 倍になる）

# ============================================================
# --- ロジック・アーキテクチャ選択設定 ---
# ============================================================

# --- Optimizer 選択 ---
# 'adamw' : AdamW 最適化 (デフォルト)
# 'adam'  : Adam 最適化
# 'sgd'   : SGD 最適化
OPTIMIZER_TYPE = 'adamw'
OPTIMIZER_SGD_MOMENTUM = 0.9      # SGD選択時のみ有効

# --- モデル・アーキテクチャ ---
# FFN層の次元倍率 (デフォルト: 4)
FFN_RATIO = 4

# 活性化関数の種類
# 'gelu' : GeLU (デフォルト)
# 'relu' : ReLU
# 'silu' : SiLU
# 'elu'  : ELU
ACTIVATION = 'gelu'

# Pre-Norm / Post-Norm の切り替え
# True  : Pre-Norm (層の入力でレイヤー正規化、学習が安定しやすい、デフォルト)
# False : Post-Norm (層の出力でレイヤー正規化)
NORM_FIRST = True

# パラメータ初期化時のスケール (デフォルト: 0.02)
INITIALIZATION_SCALE = 0.02

# --- 流行度前処理ロジック ---
# 流行度CSVでグループ化に使うカラム名 (デフォルト: 'Pango lineage')
STRENGTH_CSV_COLUMN = 'Pangolin'

# 流行度（株別サンプル数）の計算ロジック
# 'log1p' : log(count + 1) (デフォルト)
# 'sqrt'  : sqrt(count)
# 'raw'   : count (生の件数)
# 'log10' : log10(count + 1)
STRENGTH_CALC_METHOD = 'log1p'

# 流行度と系統名の参照ソース基準
# 'ncbi'  : NCBIメタデータ (sequences-241017.csv) に基づく系統名・流行度を使用
# 'usher' : UShERでの系統樹配置結果 (clades.txt) に基づく系統名・流行度を使用
STRENGTH_SOURCE = 'usher'


# ============================================================
# --- Feature 7: 基準日ベース時系列分割 & 時間軸評価 ---
# ============================================================

# --- データ分割モード ---
# 'timestep'     : 現状通り（path_length ベースの分割）← デフォルト
# 'date'         : TEMPORAL_SPLIT_DATE を基準に train/valid/test を分割（単一カットオフ）
# 'walk_forward' : 半年次フォールドを順番に学習＋評価（scripts/eval/walk_forward.py の FOLDS を参照）
SPLIT_MODE = 'walk_forward'          # 'timestep' | 'date' | 'walk_forward'

# 基準日（ISO 形式: YYYY-MM-DD）
# SPLIT_MODE='date'のときのみ有効。この日より前 → train/valid、この日以降 → test
TEMPORAL_SPLIT_DATE = '2024-01-01'

# テスト期間の終端日（ISO 形式: YYYY-MM-DD）。None の場合は上限なし
# TEMPORAL_SPLIT_DATE <= collection_date < TEMPORAL_SPLIT_TEST_END → test
# collection_date >= TEMPORAL_SPLIT_TEST_END → 除外（split_type_date = -1）
# walk_forward モードでは各フォールド実行時に自動設定される。
TEMPORAL_SPLIT_TEST_END = None

# date / walk_forward モードでの Validation 比率（基準日前のデータからランダムに抽出）
DATE_VALID_RATIO = 0.1

# True のとき main.py 起動時に split を再計算する
# （TEMPORAL_SPLIT_DATE を変更した場合などに使用）
# False のとき preprocess.py 実行時の分割をそのまま使用
FORCE_DATE_REASSIGN = False

# walk_forward モードで実行するフォールド番号のリスト（None = 全フォールド、例: [2, 6]）
WALK_FORWARD_FOLDS = None

# walk_forward: 各フォールドの訓練ウィンドウ開始日（None = 上限なし = Fold 1 のみ）
# run_walk_forward() が自動的に各フォールドに設定する。手動変更不要。
WALK_FORWARD_TRAIN_START = None

# walk_forward: 直前フォールドの best_model.pth パス（run_walk_forward() が自動設定）
# Fold 1 = None（事前学習 or ランダム初期化）、Fold N = Fold N-1 の重み
WF_PREV_FOLD_CHECKPOINT = None

# walk_forward: --folds で部分/非連続実行する際、直前フォールドのcheckpointを
# 自動探索(_wf_find_prev_fold_checkpoint)せず明示的に使うパス（--prev_checkpoint CLI引数）。
# None なら自動探索にフォールバックする。
WF_PREV_CHECKPOINT_OVERRIDE = None


# ============================================================
# 10. 新規拡張実験設定
# ============================================================

# --- Item 3: MLM / CLM Pretraining ---
# 事前学習モードを有効にする (scripts/pretrain.py で使用)
USE_PRETRAINING        = True
PRETRAINING_MODE       = 'mlm'   # 'mlm' | 'clm'
PRETRAINING_MASK_RATIO = 0.15    # MLM マスク率
# C-2: MLM マスク方式。'random'=各タイムステップ独立にマスク（従来）、
#      'span'=連続する PRETRAINING_SPAN_LEN タイムステップをまとめてマスク（SpanBERT 風）。
#      系統は連続変異の塊で進むため、単発マスクより「近傍から次を復元する」難しい課題になり、
#      position タスクの表現学習を強化する狙い。random と同等の総マスク率を保つ。
PRETRAINING_MASK_MODE  = 'span'  # 'random' | 'span'
PRETRAINING_SPAN_LEN   = 3       # span 1本あたりのタイムステップ数
PRETRAINING_EPOCHS        = 5       # 事前学習エポック数
PRETRAINING_LR            = 1e-4    # 事前学習の学習率（ピーク値）
PRETRAINING_WARMUP_EPOCHS = 1       # LinearWarmup エポック数（0 で無効）
PRETRAINING_ETA_MIN       = 1e-6    # CosineAnnealing の最小学習率

# --- Item 4: Autoregressive Decoder ---
# 各タスク用クエリ埋め込みを TransformerDecoder に通し、タスク別の特徴を生成する
USE_AUTOREGRESSIVE_DECODER = False
AR_DECODER_LAYERS          = 1   # Decoder 層数
AR_DECODER_HEADS           = 4   # Decoder ヘッド数

# --- Item 5: Substitution Prediction Head ---
# base_after (塩基変化先) と aa_after (アミノ酸変化先) を予測する追加ヘッド
# True の場合: forward() が 8-tuple を返し、labels も 7 要素になる
USE_SUBSTITUTION_HEAD  = True
LOSS_WEIGHT_BASE_AFTER = 0.05   # base_after 損失の重み
LOSS_WEIGHT_AA_AFTER   = 0.05   # aa_after 損失の重み

# --- Item 6: R-Precision ---
# evaluate_topk() で K=len(target_set) の動的 K を使った R-Precision を計算する
# True の場合: 固定 K 結果と並列に 'r_precision' キーで結果が追加される
USE_R_PRECISION = True

# R-Precisionの動的Kはバッチ内の最大target_set長（max_r_k）を全サンプル共通のneed_kとして
# 使うため、バッチ中にたった1サンプルでも巨大な共起グループ（any-of-set評価で全メンバーの
# ターゲットを合算するため、グループサイズがそのままtarget_set長になる）が混入すると、
# そのバッチの全サンプルに対してtorch.topk(..., k=need_k)とCPU化(.tolist())が走り、
# k×batch_size相当のテンソル/Pythonリストを瞬間的に確保してOOMする
# （walk_forward fold_3、Omicron早期BA.1/BA.2で共起グループ最大14,684人を実測、2026-07-30）。
# fold別のグループ内ユニークラベル数（position, group_label_count_histogram.py実測、
# 2026-07-30）: fold_3のみ突出（最大12,751）、他fold（1,2,4,5,6,7）は最大でも3,510
# （fold_2）。3,510は旧cap無し実装でも実際にOOMせず動作した実測値のため、4000に設定すれば
# fold_3以外は打ち切りなしで厳密なR-Precisionを維持しつつ、fold_3の外れ値のみ打ち切れる。
MAX_R_PRECISION_K = 4000

# --- Item 8: ESM-2 外部タンパク質言語モデル特徴量 ---【未実装スタブ】
# 【未実装】config フラグと model.py の esm2_projection スタブのみ存在。
# preprocess.py での埋め込み抽出・InputEmbedding への結線が未実装のため True にしても効果なし。
USE_ESM2        = False
ESM2_MODEL_NAME = 'esm2_t6_8M_UR50D'
ESM2_EMBED_DIM  = 320            # ESM-2 埋め込み次元 (モデルによって異なる)

# --- Item 9: 構造特徴量 (SASA / B-factor) ---【未実装スタブ】
# 【未実装】config フラグのみ。preprocess.py での特徴量付与・num_norm 拡張が未実装。
USE_STRUCTURE_FEATURES = False
STRUCTURE_CSV          = 'reference/structure/sars_cov2_structure.csv'

# --- Item 10: EVEscape スコア ---【未実装スタブ】
# 【未実装】config フラグのみ。preprocess.py での特徴量付与・num_norm 拡張が未実装。
USE_EVESCAPE = False
EVESCAPE_CSV = 'reference/evescape/sars_cov2_evescape.csv'

# --- Item 12: Supervised Contrastive Learning (SupCon) ---
# strain ラベルをポジティブペアとして同一株変異を引き付け、異株を弾く対照学習
USE_SUPCON            = False
SUPCON_TEMPERATURE    = 0.07   # SupCon の温度パラメータ
SUPCON_WEIGHT         = 0.1    # SupCon 損失の重み
SUPCON_PROJECTION_DIM = 128    # 射影ヘッドの出力次元

# --- Item 13: Ensemble ---
# 複数のチェックポイントをロードして予測を平均するアンサンブル評価
# ENSEMBLE_CHECKPOINT_PATHS が空リストの場合はアンサンブルを実行しない
USE_ENSEMBLE              = False
ENSEMBLE_N_MODELS         = 3
ENSEMBLE_CHECKPOINT_PATHS = []   # ロードするチェックポイントパスのリスト (絶対パス)


# ============================================================
# 11. ラベル多義性への発展的アプローチ（README 改善の方向性 6〜12）
#     すべてデフォルト False。OFF 時は現行挙動と完全一致（無回帰）。
#     各案は独立に有効化できるが、6 と 7 は同じラベル構築層を扱うため排他
#     （どちらか一方のみを有効にする）。
# ============================================================

# --- 提案6: 子孫頻度で重み付けした確率的ターゲット ---
# 内部ノードの union ターゲットを、それを保有する子孫リーフ数で重み付けして
# ソフトターゲット化する。w_t ∝ n_t^FREQ_WEIGHT_GAMMA。
# 【要 preprocess 再実行】labels テーブルに weights 列を保存する必要がある。
#   weights 列が無い DB では自動的に一様重み（現行と同一）にフォールバックする。
USE_FREQ_WEIGHTED_TARGETS = False
FREQ_WEIGHT_GAMMA         = 1.0    # 重みの温度。<1 で平滑化（稀な逃避変異の過小評価を緩和）

# --- 提案7: 内部ノードの脱・合算（子枝ごとにサンプル分割）---
# 内部ノードの union を作らず (親→各子枝) を個別サンプルとして emit する。
# 6 とは排他。【要 preprocess 再実行】系統樹走査をエッジ単位に変更する必要がある。
USE_BRANCH_DISAGGREGATION = False
BRANCH_SELECTOR_EMBED     = False  # 子系統セレクタ埋め込みを入力条件に加えるか

# --- 提案8: 多峰対応の Position 出力ヘッド（Mixture / Gated Experts）---
# 単一 softmax の代わりに K 個のエキスパートsoftmaxをゲートで混合し多峰分布を表現。
# Position ヘッドのみ差し替え。preprocess 不要・モデル側のみで有効化可能。
USE_MIXTURE_POSITION_HEAD   = False
POSITION_MIXTURE_K          = 4     # エキスパート数
MIXTURE_LOAD_BALANCE_WEIGHT = 0.01  # ゲート崩壊防止の負荷分散正則化の重み

# --- 提案9: 選択的予測・棄却（Abstention）ヘッド ---
# サンプルのターゲット多義性（log(1+ターゲット数)）を回帰する不確実性ヘッドを追加し、
# 推論時に閾値超過サンプルを棄却する。preprocess 不要。
USE_ABSTENTION_HEAD    = False
LOSS_WEIGHT_ABSTENTION = 0.05
ABSTENTION_THRESHOLD   = 2.0    # 推論時に棄却する log(1+ターゲット数) 予測値の閾値

# --- 提案10: ホモプラシー（収斂進化）事前分布 ---
# 各ゲノム位置の系統樹上の再発回数を Position ロジットへ log(1+r) バイアスとして注入。
# CSV (position_id,recurrence_count) を事前生成しておく。preprocess 不要（学習は CSV を読むだけ）。
#   CSV が存在しない場合はバイアス無効（現行と同一）にフォールバックする。
USE_HOMOPLASY_PRIOR   = False
HOMOPLASY_CSV         = 'reference/homoplasy/site_recurrence.csv'
HOMOPLASY_PRIOR_SCALE = 1.0     # 学習可能スケール係数の初期値

# --- 学習時 Region 条件付き Position 予測 ---
# evaluate.py の USE_HIERARCHICAL_PREDICTION（推論時の事後マスキング。予測Region上位K個に
# 属さないposition候補のロジットを除外する）は、fold_3のcheckpointで再学習なしに
# position_hit_rate +0.309pt・全fold平均+0.105ptの改善を実測済み（2026-08-08、
# hierarchical_prediction_check.py）。一方モデル自体（region_head・position_head）は
# 完全に独立に学習されており「regionが分かればpositionが絞れる」構造を学習時には
# 教わっていない。position_headの出力に、region_headの予測確率をposition→region写像
# （db/queries.py:get_position_region_map、事前分布としてUSE_HOMOPLASY_PRIORと同型）で
# 引き当てた log確率を学習可能スケール付きで加算し、この構造を学習時から組み込む。
USE_REGION_CONDITIONED_POSITION = True
REGION_HIER_SCALE = 1.0   # 学習可能スケール係数の初期値（HOMOPLASY_PRIOR等と同じ1.0初期化）

# --- 提案: Co-occurrence Attentionへの頻度ペナルティ ---
# 大きな共起グループ(オミクロン系統等)で、ほぼ全系統に共通する創始変異(nsp12:P323L等)に
# Attentionが機械的に収束する現象への対策（verify_coattn_frequency_confound.py で確認、
# embeddingノルム由来ではないことは verify_position_embedding_norm_confound.py で確認済み）。
# HOMOPLASY_CSV（上記と共用）の再発回数が高い変異ほど、Cross-Attentionスコアへ学習可能
# スケール付きの負バイアスを加算し、系統非依存の"定番"変異への注目を抑制する。
# USE_HOMOPLASY_PRIOR（position headへの正バイアス）と対称的な設計。既定Offで挙動不変。
# 2026-07-23: walk_forward全7fold本実行のためTrueに設定（transformer_260707のbaseline
# walk_forward(20260719_001947)とはこのフラグのみが差分、他のconfigは完全一致させている）。
# 2026-08-03: True版walk_forward（20260802_082112、一連のOOM修正・MAX_GROUP_MEMBERS_FOR_CACHE
# 適用済み）とのクリーンなA/B比較のため、同一コード・同一修正状態でFalseに戻して新規学習
# （walk_forward 20260803_115144として完了）。260707 baseline(20260719_001947)はOOM修正等を
# 含まずフラグ以外の条件も揃っていないため比較対象から外す。
# 2026-08-04: 上記A/B比較の結果、R-Precision全体でposition -0.637pt/aa_pos -0.537pt
# （fold_3等Omicron期を含むほぼ全foldでTrueが劣る）、一方region +0.531pt/codon_pos +0.167pt/
# synonymous +0.375ptとTrueが有利な逆のトレードオフを確認。本プロジェクトが重視するposition
# 予測でむしろ悪化し、「巨大共起グループでの創始変異へのAttention収束がposition精度を
# 下げる」という導入時の仮説を裏付ける効果は確認できなかったため、既定をFalseに戻す
# （task別に出し分ける案も検討したが、共有パイプライン全体の二重計算が必要でコストに
# 見合わないと判断し見送り）。
USE_COATTN_FREQUENCY_PENALTY = False
COATTN_FREQUENCY_PENALTY_SCALE = 1.0   # 学習可能スケール係数の初期値

# --- 提案11: 分子時計（枝長）補助タスク ---
# 「次の変異までの枝長」を回帰する補助ヘッドを追加し時系列表現を正則化。
# 【要 preprocess 再実行】labels テーブルに branch_length 列を保存する必要がある。
USE_BRANCH_LENGTH_AUX     = False
LOSS_WEIGHT_BRANCH_LENGTH = 0.02

# --- 提案12: kNN 検索拡張出力（kNN-LM 型）---
# 学習後にエンコーダ表現→実ターゲットのデータストアを構築し、推論時に近傍検索で
# softmax を補間する。p_final = λ·p_kNN + (1-λ)·p_model。preprocess・学習不要（後付け）。
USE_KNN_OUTPUT     = False
KNN_K              = 16
KNN_LAMBDA         = 0.25
KNN_DATASTORE_PATH = 'cache/knn_datastore'


# ============================================================
# 12. 改善の方向性 2〜5（難サンプル回避・出力空間圧縮・分布シフト対策）
#     README「### 改善の方向性」の 2〜5 に対応。すべてデフォルト False
#     （＝現行挙動と完全一致・無回帰）。5（エントロピー別層別評価）は
#     常時出力のためフラグを持たない（main.py の _save_results が常に実行）。
# ============================================================

# --- 提案2: ターゲットエントロピーによる学習サンプルフィルタリング ---
# 学習時（Train split のみ）に、系統別の target_position_entropy_norm が
# TRAIN_ENTROPY_MAX を超える系統のサンプルを除外し、低エントロピー（学習可能）
# サンプルに集中する。valid/test は除外しない（層別評価のため）。
# preprocess 再実行は不要: 系統別エントロピーは labels テーブルから動的に算出する。
USE_TRAIN_ENTROPY_FILTER = False
TRAIN_ENTROPY_MAX        = 0.7    # 正規化エントロピー上限（0〜1、要チューニング）

# --- 提案3: 階層的予測（Region → Position 分解, Option A: 推論時マスキング）---
# 評価時に、予測 Region（上位 HIERARCHICAL_TOPK_REGIONS 個）に属さない位置の
# Position ロジットを -inf でマスクしてから softmax/topk を行う。学習は不変。
# Option A のみ実装（学習時間増加ゼロ）。Option B/C（条件付きヘッド・2段階）は未実装。
USE_HIERARCHICAL_PREDICTION = False
HIERARCHICAL_TOPK_REGIONS   = 3    # マスクに使う予測 Region の上位個数（>=1）

# --- 提案4: 主要系統クレード埋め込み（分布シフト対策）---
# strain 名（Pango 系統）を主要クレード（Wuhan/Alpha/Delta/BA.1/BA.2/BA.4-5/
# XBB/JN 等）へ写像し、その Embedding を pooled 表現(latest_context)へ加算する。
# 未知系統は clade_id=0（padding, ゼロベクトル）にフォールバック。
USE_CLADE_EMBEDDING = False


