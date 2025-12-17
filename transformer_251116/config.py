import torch

# --- 基本設定 ---
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- パス設定 ---
# 補助データのパスを個別に指定
CODON_CSV_PATH = 'meta_data/codon_mutation4.csv'
FREQ_CSV_PATH = 'outputs/table_heatmap/251031/table_set/table_set.csv'
DISSIMILARITY_CSV_PATH = 'meta_data/aa_properties/dissimilarity_metrics.csv'

USHER_DIR = '../usher_output/' # usherからの出力が配置されているディレクトリ
CACHE_DIR = './cache/transformer_251116/' # 前処理済みデータを保存するディレクトリ
OUTPUT_DIR = './outputs/models/transformer_251116/' # 学習済みモデルや結果を保存するディレクトリ

# --- データ前処理設定 ---
MAX_TOTAL_SAMPLES = None # 全体のサンプル数をこの値に制限する (層化サンプリング)。Noneですべて。
MAX_STRAIN_NUM = None # 読み込む株の最大数 (Noneですべて)
MAX_TOP_STRAINS = 10 # 上位N株のみ使用 (Noneですべて)
MAX_NUM_PER_STRAIN = 20000 # 1株あたりの最大サンプル数 (Noneですべて)
MAX_CO_OCCUR = 5 # 1タイムステップあたりの最大共起変異数
TRAIN_MAX_LEN = 40 # 学習に使用するシーケンス長の最大値
VALID_NUM = 3 # 学習データと重複させる評価データのシーケンス長（期間）
VALID_RATIO = 0.2 # TRAIN_MAX_LEN-VALID_NUM+1 ~ TRAIN_MAX_LENの期間のデータのうち、評価データに回す割合

# --- モデル入力長設定 ---
SEQ_LEN = None # モデルに入力するシーケンスの最大長。NoneにするとTRAIN_MAX_LENと同じになる

# --- 効率化設定 ---
ENABLE_EARLY_FEATURE_GENERATION = True # 特徴量生成を分割前に実行するかどうか
BATCH_SIZE_FEATURE_GEN = 10000 # 特徴量生成時のバッチサイズ

# --- モデルアーキテクチャ設定 ---
# カテゴリカル特徴量の埋め込み次元
EMBED_DIM_BASE = 8
EMBED_DIM_POS = 32
EMBED_DIM_CODON_POS = 4
EMBED_DIM_AA = 16
EMBED_DIM_PROTEIN = 16
EMBED_DIM_PROTEIN_POS = 32

# 各特徴量を結合した後の次元
# [bef, pos, aft, codon_pos, aa_b, prot_pos, aa_a, protein]
CAT_FEATURE_DIM = EMBED_DIM_BASE * 2 + EMBED_DIM_POS + EMBED_DIM_CODON_POS + EMBED_DIM_AA * 2 + EMBED_DIM_PROTEIN + EMBED_DIM_PROTEIN_POS
NUM_CAT_FEATURES = 8
NUM_CHEM_FEATURES = 5
FEATURE_DIM = 128 # NUM_HEADSで割り切れる値

# Transformer Encoderの設定
# ★ 共起変異用 (Timestep-wise) Transformer
N_LAYERS_TIMESTEP = 3
N_HEADS_TIMESTEP = 4

# ★ 時系列用 (Sequence) Transformer
N_LAYERS_SEQUENCE = 3
N_HEADS_SEQUENCE = 4

HIDDEN_DIM_MULTIPLIER = 4 # FFNNの中間層の次元を入力次元の何倍にするか
DROPOUT = 0.1

# --- 語彙数設定 (0はパディング用に予約) ---
VOCAB_SIZE_BASE = 5      # 1-4: A,C,G,T
VOCAB_SIZE_POS = 29904   # 1-29903
VOCAB_SIZE_CODON_POS = 4 # 1-3
VOCAB_SIZE_AA = 22       # 1-21
VOCAB_SIZE_PROTEIN = 37  # 1-36
VOCAB_SIZE_PROTEIN_POS = 10001  # 1-10000

# --- 訓練設定 ---
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5 #このエポック数、検証ロスが改善しなければ停止
LOSS_WEIGHT_REGION = 0.5 # 領域予測の損失に対する重み
LOSS_WEIGHT_POSITION = 0.5 # 位置予測の損失に対する重み
PRINT_COUNT = 10000 # 特徴量生成中にこの件数ごとに進捗を表示

# --- デバッグ設定 ---
DEBUG_PRINT_BATCH = True # Trueにすると、訓練開始時に最初のバッチの中身を詳細表示する

# --- 評価設定 ---
TOP_K_EVAL = 1 # Top-K 評価で使用する K の値

#　特徴量生成ログの表示間隔
PRINT_COUNT = 10000

# --- 語彙定義 ---
BASE_VOCABS = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
AA_VOCABS = {
    'F': 1, 'L': 2, 'I': 3, 'M': 4, 'V': 5, 'S': 6, 'P': 7, 'T': 8, 'A': 9, 'Y': 10,
    'H': 11, 'Q': 12, 'N': 13, 'K': 14, 'D': 15, 'E': 16, 'C': 17, 'W': 18, 'R': 19, 'G': 20,
    '*': 21, 'n': 0
}
PROTEIN_VOCABS = {'non_coding1':1, 'nsp1':2, 'nsp2':3, 'nsp3':4, 'nsp4':5,
                    'nsp5':6, 'nsp6':7, 'nsp7':8, 'nsp8':9, 'nsp9':10,
                    'nsp10':11, 'nsp12':12, 'nsp13':13, 'nsp14':14,
                    'nsp15':15, 'nsp16':16, 'non_coding2':17, 'S':18,'non_coding3':19,
                    'ORF3a':20, 'non_coding4':21, 'E':22, 'non_coding5':23,'M':24, 
                    'non_coding6':25,'ORF6':26, 'non_coding7':27, 'ORF7a':28,'ORF7b':29,
                    'non_coding8':30,'ORF8':31, 'non_coding9':32,'N':33, 'non_coding10':34,
                    'ORF10':35,'non_coding11':36, 'PAD':0}
DNA2PROTEIN = {
    'TTT' : 'F', 'TCT' : 'S', 'TAT' : 'Y', 'TGT' : 'C', 'TTC' : 'F', 'TCC' : 'S', 'TAC' : 'Y', 'TGC' : 'C',
    'TTA' : 'L', 'TCA' : 'S', 'TAA' : '*', 'TGA' : '*', 'TTG' : 'L', 'TCG' : 'S', 'TAG' : '*', 'TGG' : 'W',
    'CTT' : 'L', 'CCT' : 'P', 'CAT' : 'H', 'CGT' : 'R', 'CTC' : 'L', 'CCC' : 'P', 'CAC' : 'H', 'CGC' : 'R',
    'CTA' : 'L', 'CCA' : 'P', 'CAA' : 'Q', 'CGA' : 'R', 'CTG' : 'L', 'CCG' : 'P', 'CAG' : 'Q', 'CGG' : 'R',
    'ATT' : 'I', 'ACT' : 'T', 'AAT' : 'N', 'AGT' : 'S', 'ATC' : 'I', 'ACC' : 'T', 'AAC' : 'N', 'AGC' : 'S',
    'ATA' : 'I', 'ACA' : 'T', 'AAA' : 'K', 'AGA' : 'R', 'ATG' : 'M', 'ACG' : 'T', 'AAG' : 'K', 'AGG' : 'R',
    'GTT' : 'V', 'GCT' : 'A', 'GAT' : 'D', 'GGT' : 'G', 'GTC' : 'V', 'GCC' : 'A', 'GAC' : 'D', 'GGC' : 'G',
    'GTA' : 'V', 'GCA' : 'A', 'GAA' : 'E', 'GGA' : 'G', 'GTG' : 'V', 'GCG' : 'A', 'GAG' : 'E', 'GGG' : 'G',
    'nnn':'n'
}