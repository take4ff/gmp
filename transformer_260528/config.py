# --- config.py ---
import torch

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# --- Ablation実験用マスク設定 ---
# Trueにすると、その数値特徴量を強制的に0にして学習・推論を行う
ABLATION_MASKS = {
    'FREQ': False,       # 変異頻度を無効化
    'HYDRO': False,      # 疎水性差 (Eisenberg-Weiss) を無効化
    'CHARGE': False,     # 電荷差を無効化
    'SIZE': False,       # サイズ差を無効化
    'BLSM': False,       # BLOSUM62差を無効化
    'PAM250': False,     # PAM250スコアを無効化
    'CODON_LOG_RATIO_DIFF': True,
    'CODON_LOG_RATIO_BEFORE': True,
    'HUMAN_CODON_RSCU_DIFF': True,
    'SCV2_CODON_RSCU_DIFF': True,
    'MUTATION_CONTEXT': True,     # 前後3文字（6塩基）をすべて無効化
        'MUTATION_CONTEXT_L3': False,  # 3文字前の塩基を無効化
        'MUTATION_CONTEXT_L2': False,  # 2文字前の塩基を無効化
        'MUTATION_CONTEXT_L1': False,  # 1文字前の塩基を無効化
        'MUTATION_CONTEXT_R1': False,  # 1文字後の塩基を無効化
        'MUTATION_CONTEXT_R2': False,  # 2文字後の塩基を無効化
        'MUTATION_CONTEXT_R3': False,  # 3文字後の塩基を無効化
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
CODON_CSV         = "meta_data/codon_mutation4.csv"
FREQ_CSV          = "meta_data/table_set.csv"
DISSIMILARITY_CSV = "meta_data/aa_properties/dissimilarity_metrics.csv"
PAM250_CSV        = "meta_data/aa_properties/PAM250.csv"
SEQUENCES_CSV     = "meta_data/sequences-241017.csv"  # 株別サンプル数CSV
CODON_LOG_RATIO_CSV  = "meta_data/codon_usage/Codon_log_ratio.csv"
HUMAN_CODON_RSCU_CSV = "meta_data/codon_usage/human_Codon_RSCU.csv"
SCV2_CODON_RSCU_CSV  = "meta_data/codon_usage/SCV2_Codon_RSCU.csv"

OUTPUT_DIR = 'outputs/transformer_260528/'
MODEL_SAVE_DIR = OUTPUT_DIR + 'models'
RESULT_SAVE_DIR = OUTPUT_DIR + 'results'
INCREMENTAL_CACHE_DIR = 'cache/incremental_features'

# --- DuckDB データベース設定 ---
DB_DIR = 'db'
USE_DB = True  # Trueの場合、pickleキャッシュの代わりにDuckDBを使用
MIN_SEQ_LEN = 10  # 最小シーケンス長

# --- キャッシュ・効率化設定 ---
CACHE_DIR = 'cache/dataset'
BATCH_SIZE_FEATURE_GEN = 5000
FORCE_REPROCESS = False
ENABLE_LRU_CACHE = True
ENABLE_PARALLEL_PROCESSING = True
CACHE_MAX_SIZE = 10000
NUM_PREPROCESS_WORKERS = 1    # 前処理時の並列ワーカー数上限 (OOM安全策として1を推奨)
PREPROCESS_CHUNK_SIZE = 1000    # 前処理ワーカー内でのチャンクバッファサイズ (小さくして省メモリ化)
DB_WRITE_BATCH_SIZE = 10000     # DB書き込み時のバッチサイズ
EARLY_STOPPING_PATIENCE = 7   # [260417] エポック増加に合わせて余裕を持たせる
WEIGHT_DECAY = 0.01

# --- 保存・ログ設定 ---
SAVE_PREDICTIONS = True         # 予測結果の詳細をファイルに出力するか
SAVE_STRAIN_INFO = True         # 使用した株の情報を保存するか

# --- データセット設定 ---
MAX_SEQ_LEN = 39
TARGET_LEN = 1
TRAIN_MAX = 40 # TS:1-40を学習に利用(TRAIN_MAX > MAX_SEQ_LEN + TARGET_LEN)
VALID_NUM = 3
MAX_CO_OCCURRENCE = 5
VALID_RATIO = 0.2

# --- サンプリングモード設定 ---
# (両モード共通) 頻出上位 MAX_STRAIN_NUM 株のデータのみを使用対象とする
MAX_STRAIN_NUM = 10000       # 共通: 使用する上位株数

# 'proportional': 対象株のデータ比率を維持しつつ、合計 MAX_NUM 件になるように抽出
# 'fixed_per_strain': 対象の各株からそれぞれ最大 MAX_NUM_PER_STRAIN 件ずつ抽出
SAMPLING_MODE = 'proportional'

# モードA: 比率サンプリング用 (SAMPLING_MODE = 'proportional')
MAX_NUM = 10000000  # 全体での抽出合計サンプル数の上限

# モードB: 各株の上限数制限用 (SAMPLING_MODE = 'fixed_per_strain')
MAX_NUM_PER_STRAIN = 20000   # 各株ごとの最大抽出サンプル数

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
NUM_FEATURE_STRING = 15  # 特徴量文字列数 (シノニマス + 前後3塩基)
NUM_CHEM_FEATURES = 10

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

# --- Conv1D局所特徴抽出の有無（Ablation Study用） ---
# Trueの場合: Conv1D層を使用して局所的な文脈情報を抽出
# Falseの場合: Conv1D層をスキップ（ベースラインとの比較用）
USE_LOCAL_CONV1D = False
LOCAL_CONTEXT_KERNEL_SIZE = 3

# --- Origin Attentionの有無（Ablation Study用） ---
# Trueの場合: 原点（Wuhan株）を常に参照するCross-Attentionを使用
# Falseの場合: Origin Attentionをスキップ
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

# --- 訓練設定 ---
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
EPOCHS = 15              # Soft Target収束のため余裕を持たせる
TOP_K_EVAL = 1 # Top-5でのRecallなども見たい場合はここを変更

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
# 実験結果: 中規模データでFLはCEより精度低下、大規模では維持したため cbce を採用
#   'cbce'    : Class-Balanced CrossEntropyLoss（推奨：安定性高、Soft Targetと相性良好）
#   'cb_focal': Class-Balanced + FocalLoss アプローチA（大規模データで再試する場合）
#   'ce'      : 通常のCrossEntropyLoss（デバッグ・ベースライン用）
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
USE_WANDB = True
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

# --- Early Stopping 設定 (新規追加) ---
EARLY_STOPPING_METRIC = "val_loss"  # 早期停止の基準 ('val_loss', 'val_macro_recall' 等)
EARLY_STOPPING_MODE = "min"         # 'min' (loss等, 小さいほど良い) または 'max' (accuracy等, 大きいほど良い)

# --- 評価・可視化のハードコード解除 (新規追加) ---
EVAL_TOP_KS = (1, 3, 5)             # Top-K 評価で計算する K のリスト
PLOT_TOP_N_POSITIONS = 40           # 塩基位置の Recall でプロットする上位 N 件
SAVE_ATTENTION_HEATMAP = True       # Attentionヒートマップを保存するかどうか

# --- データ前処理 (新規追加) ---
NORMALIZE_NUM_FEATURES = False      # 数値特徴量(FREQ, HYDRO等)を正規化するかどうか

# --- strength ヘッドの損失関数 ---
# 'mse'  : 均乗誤差（現状・外れ値に弱い）
# 'mae'  : 平均絶対誤差（外れ値に強健）
# 'huber': Huber Loss（MAEとMSEの中間、delta内はMSE、delta外はMAE）
STRENGTH_LOSS_TYPE = 'mse'  # 'mse' | 'mae' | 'huber'
STRENGTH_HUBER_DELTA = 1.0  # Huber Lossの間屠パラメータ (huber選択時のみ有効)

# --- ECE (期待キャリブレーション誤差) 設定 ---
# True の場合: evaluate() 内で ECE を計算し、final_metrics_by_ts に追加する
# 安定性のため n_bins 分割した単純隣値分割方式 (Equal-Width Binning)
SAVE_ECE = True        # True | False
ECE_N_BINS = 10        # ビン数 (10 が標準定義)

# ============================================================
# --- 精度向上実験設定 ---
# ============================================================

# --- Curriculum Learning ---
# 学習初期は短いパスのみを使い、徐々に長いパスを追加する。
# 初期エポックは正解シグナルが明確な短いパスで基礎を学習し、
# CURRICULUM_WARMUP_EPOCHS エポック後に全長のパスを解禁する。
CURRICULUM_LEARNING = False
CURRICULUM_MIN_LEN_INIT = 5     # 最初のエポックで使う最短パス長
CURRICULUM_WARMUP_EPOCHS = 5    # このエポック数をかけて制限を線形緩和

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
STRENGTH_SOURCE = 'ncbi'


# ============================================================
# --- Feature 7: 基準日ベース時系列分割 & 時間軸評価 ---
# ============================================================

# --- データ分割モード ---
# 'timestep' : 現状通り（path_length ベースの分割）← デフォルト
# 'date'     : TEMPORAL_SPLIT_DATE を基準に train/valid/test を分割
SPLIT_MODE = 'timestep'          # 'timestep' | 'date'

# 基準日（ISO 形式: YYYY-MM-DD）
# SPLIT_MODE='date' のときのみ有効。この日より前 → train/valid、以降 → test
TEMPORAL_SPLIT_DATE = '2023-12-31'

# date モードでの Validation 比率（基準日前のデータからランダムに抽出）
DATE_VALID_RATIO = 0.1

# True のとき main.py 起動時に split を再計算する
# （TEMPORAL_SPLIT_DATE を変更した場合などに使用）
# False のとき preprocess.py 実行時の分割をそのまま使用
FORCE_DATE_REASSIGN = False

# --- 評価 X 軸モード ---
# 'timestep' : X 軸 = path_length ← デフォルト
# 'date'     : X 軸 = collection_date の年月 (YYYY-MM)、valid と test を凡例で区別
EVAL_X_AXIS = 'timestep'         # 'timestep' | 'date'


