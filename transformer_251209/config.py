# --- config.py ---
import torch

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# --- Ablation実験用マスク設定 (★追加) ---
# Trueにすると、その特徴量を強制的に0（または無効値）にして学習・推論を行う
# 注: PROTEIN_POSは予測ターゲットのためマスク対象外
ABLATION_MASKS = {
    'CHEM_FEATURES': False,     # 数値特徴量（疎水性など）を無効化
    'CO_OCCURRENCE': False,     # 共起情報を無視（共起数1として扱う、またはAttention無効化）
    'AA_MUTATION': False,       # アミノ酸変異 (Before/After) をマスク
    'CODON_POS': False,         # コドン位置をマスク
}

# --- 強度スコア設定 (株別サンプル数ベース) ---
USE_STRENGTH_FILTER = False          # 評価時に強度フィルタを適用 (Falseでカテゴリ分析を使用)
STRENGTH_THRESHOLD = 0.0             # 強度スコアのしきい値 (0.0=フィルタなし)
# 強度スコアは log(1 + sample_count) で計算される

# 強度カテゴリの閾値 (log(1+x)スケール)
# 例: log(1+10)=2.4, log(1+100)=4.6, log(1+1000)=6.9
STRENGTH_CATEGORY_LOW_MAX = 3.0      # 小: 0 ~ 3.0 (サンプル数 ~20)
STRENGTH_CATEGORY_MED_MAX = 5.0      # 中: 3.0 ~ 5.0 (サンプル数 ~150)
                                     # 大: 5.0~ (サンプル数 150+)

# --- キャッシュ・効率化設定 ---
CACHE_DIR = './cache'
BATCH_SIZE_FEATURE_GEN = 5000
FORCE_REPROCESS = False
ENABLE_LRU_CACHE = True
ENABLE_PARALLEL_PROCESSING = True
CACHE_MAX_SIZE = 10000
EARLY_STOPPING_PATIENCE = 5
WEIGHT_DECAY = 0.01

# --- パス設定 ---
DATA_BASE_DIR = '../usher_output/'
Codon_csv = "meta_data/codon_mutation4.csv"
Freq_csv = "outputs/table_heatmap/251031/table_set/table_set.csv"
Disimilarity_csv = "meta_data/aa_properties/dissimilarity_metrics.csv"
PAM250_csv = "meta_data/aa_properties/PAM250.csv"

OUTPUT_DIR = './outputs/transformer_251209/'
MODEL_SAVE_DIR = OUTPUT_DIR + 'models'
RESULT_SAVE_DIR = OUTPUT_DIR + 'results'
INCREMENTAL_CACHE_DIR = OUTPUT_DIR + 'cache/incremental_features'

# --- 保存・ログ設定 ---
SAVE_PREDICTIONS = True         # 予測結果の詳細をファイルに出力するか
SAVE_STRAIN_INFO = True         # 使用した株の情報を保存するか

# --- データセット設定 ---
MAX_SEQ_LEN = 39
TARGET_LEN = 1
TRAIN_MAX = 40 # TS:1-40を学習に利用(TRAIN_MAX > MAX_SEQ_LEN + TARGET_LEN)
VALID_NUM = 3
MAX_CO_OCCURRENCE = 20
VALID_RATIO = 0.2

# --- サンプリングモード設定 ---
# 'proportional': 比率サンプリング (MAX_NUM件を各株の比率に応じて抽出)
# 'fixed_per_strain': 株数×サンプル数制限 (MAX_STRAIN_NUM株からMAX_NUM_PER_STRAIN件ずつ)
SAMPLING_MODE = 'fixed_per_strain'

# モードA: 比率サンプリング用 (SAMPLING_MODE = 'proportional')
MAX_NUM = 10000  # 合計サンプル数（各株から比率に応じて抽出）

# モードB: 株数×サンプル数制限用 (SAMPLING_MODE = 'fixed_per_strain')
MAX_NUM_PER_STRAIN = 50   # 各株からの最大サンプル数
MAX_STRAIN_NUM = 100      # 使用する株数

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
NUM_FEATURE_STRING = 8
NUM_CHEM_FEATURES = 6

VOCAB_SIZE_POSITION = 30006
VOCAB_SIZE_BASE = 7
VOCAB_SIZE_AA = 23
VOCAB_SIZE_CODON_POS = 6
VOCAB_SIZE_PROTEIN_POS = 10001
NUM_REGIONS = 37

EMBED_DIM_POS = 128
EMBED_DIM_BASE = 32
EMBED_DIM_AA = 64
EMBED_DIM_REGION = 128
EMBED_DIM_CODON_POS = 16
EMBED_DIM_PROTEIN_POS = 128

FEATURE_DIM = 768
HIDDEN_DIM = FEATURE_DIM
N_HEADS = 8
N_LAYERS = 8
DROPOUT = 0.1
LOCAL_CONTEXT_KERNEL_SIZE = 7

# --- 訓練設定 ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 5
TOP_K_EVAL = 1 # Top-5でのRecallなども見たい場合はここを変更

LOSS_WEIGHT_REGION = 0.3
LOSS_WEIGHT_POSITION = 0.3
LOSS_WEIGHT_PROTEIN_POS = 0.2    # タンパク質位置の損失重み
LOSS_WEIGHT_STRENGTH = 0.2       # 強度スコア予測の損失重み (回帰)

# --- 高度な学習設定 (追加) ---
# Label Smoothing: 過学習抑制 (Trueで適用)
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING_FACTOR = 0.1

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
USE_MULTITASK_LOSS = True