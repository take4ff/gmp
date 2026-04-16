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
}

# --- データロード設定 ---
USE_UNIQUE_FILTER = True        # 学習時にパス重複(raw_path)を除外するか（ユニーク化）

# --- 流行度設定 (株別サンプル数ベース) ---
USE_STRENGTH_FILTER = False          # 評価時に流行度フィルタを適用 (Falseでカテゴリ分析を使用)
STRENGTH_THRESHOLD = 0.0             # 流行度のしきい値 (0.0=フィルタなし)
# 流行度は log(1 + sample_count) で計算される

# 流行度カテゴリの閾値 (log(1+x)スケール)、自動設定機能がない場合は手動で設定
# 例: log(1+10)=2.4, log(1+100)=4.6, log(1+1000)=6.9
STRENGTH_CATEGORY_LOW_MAX = 3.0      # 小: 0 ~ 3.0 (サンプル数 ~20)
STRENGTH_CATEGORY_MED_MAX = 5.0      # 中: 3.0 ~ 5.0 (サンプル数 ~150)
                                     # 大: 5.0~ (サンプル数 150+)

# --- パス設定 ---
DATA_BASE_DIR = '../usher_output/'
CODON_CSV         = "meta_data/codon_mutation4.csv"
FREQ_CSV          = "outputs/table_heatmap/251031/table_set/table_set.csv"
DISSIMILARITY_CSV = "meta_data/aa_properties/dissimilarity_metrics.csv"
PAM250_CSV        = "meta_data/aa_properties/PAM250.csv"

OUTPUT_DIR = 'outputs/transformer_260416/'
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
EARLY_STOPPING_PATIENCE = 5
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
MAX_NUM = 10000  # 全体での抽出合計サンプル数の上限

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
NUM_FEATURE_STRING = 9  # 8から9に変更（シノニマス特徴量追加）
NUM_CHEM_FEATURES = 6

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
USE_LOCAL_CONV1D = True
LOCAL_CONTEXT_KERNEL_SIZE = 3

# --- Origin Attentionの有無（Ablation Study用） ---
# Trueの場合: 原点（Wuhan株）を常に参照するCross-Attentionを使用
# Falseの場合: Origin Attentionをスキップ
USE_ORIGIN_ATTENTION = True
ORIGIN_ATTENTION_HEADS = 4  # Origin Attentionのヘッド数

# --- 訓練設定 ---
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
EPOCHS = 15
TOP_K_EVAL = 1 # Top-5でのRecallなども見たい場合はここを変更

LOSS_WEIGHT_REGION = 0.14         # 基因領域の損失重み
LOSS_WEIGHT_POSITION = 0.7        # 塩基位置の損失重み
LOSS_WEIGHT_AA_POS = 0.10         # アミノ酸配列位置の損失重み
LOSS_WEIGHT_CODON_POS = 0.04      # コドン位置の損失重み (1,2,3の3クラス)
LOSS_WEIGHT_SYNONYMOUS = 0.02    # シノニマス/ノンシノニマスの損失重み (2クラス)
LOSS_WEIGHT_STRENGTH = 0.02       # 流行度予測の損失重み (回帰)

# --- 高度な学習設定 (追加) ---
# Label Smoothing: 過学習抑制 (Trueで適用)
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING_FACTOR = 0.1

# --- 損失関数設定 (新規追加・統合) ---
# 選択肢: 'ce', 'focal', 'wce', 'cbce', 'cb_focal'
LOSS_FUNCTION_TYPE = 'focal'
FOCAL_LOSS_GAMMA = 1.0           # focal, cb_focal用 (0で通常のCE相当、大きいほど難サンプル重視)
FOCAL_LOSS_ALPHA = None          # focal用 (floatで指定可能, 例: 0.25, Noneで無効)
CLASS_BALANCED_BETA = 0.9999     # Class-Balanced用のパラメータ (通常0.99～0.9999)
NORMALIZE_LOSS_WEIGHTS = True    # 頻度が少ないクラスでの損失爆発を防ぐため重みを正規化する

# Scheduler: 学習率の自動調整 (Trueで適用)
USE_SCHEDULER = True
SCHEDULER_ETA_MIN = 1e-6  # 最小学習率

# Weights & Biases: 実験可視化 (Trueで適用)
USE_WANDB = True
WANDB_OFFLINE = True  # Trueにするとオフラインモード (長時間処理時の接続切れ対策)
WANDB_PROJECT_NAME = "viral-mutation-transformer"
WANDB_RUN_NAME = None # Noneなら自動生成 (例: "run_20251121_...")

# MultiTaskLoss 設定
# Trueの場合、config.LOSS_WEIGHT_REGION/POSITION は無視され、自動調整される
# 最終的な重みを表示？
USE_MULTITASK_LOSS = False

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
