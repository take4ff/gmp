# --- config.py ---
import torch

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42  # 再現性確保のための乱数シード

# キャッシュ設定
CACHE_DIR = './cache' # キャッシュを保存するディレクトリ
FORCE_REPROCESS = False # Trueにするとキャッシュを無視して強制的に再処理

# 外部データパス (dataset.py が参照)
Codon_csv="meta_data/codon_mutation4.csv"
Freq_csv="table_heatmap/251031/table_set/table_set.csv"
Disimilarity_csv="meta_data/aa_properties/dissimilarity_metrics.csv"

# データセット設定
SEQ_LEN = 39         # 入力の長さ
TARGET_LEN = 1             # ターゲットの長さ
TRAIN_MAX = SEQ_LEN+TARGET_LEN       # 訓練・評価に使う最大時系列長 (1~40)、テストはこれ以上
VALID_NUM = 3        # 検証データで扱うタイムステップ数（SEQ_LENが40の場合は38,39,40の一部を利用）
DATA_BASE_DIR = '../usher_output/'  # データのベースディレクトリ
MAX_NUM = None       # 読み込むデータの最大数 (Noneの場合は全件読み込み)
MAX_NUM_PER_STRAIN = 10000  # ユニーク後のstrainベースで読み込むデータの最大数 (Noneの場合は全件読み込み)
MAX_STRAIN_NUM = 10  # 読み込むstrainの最大数 (Noneの場合は全strainを使用)


MAX_CO_OCCUR = 5     # 1タイムステップで同時に起こる変異の最大数（パディング用）
VALID_RATIO = 0.2  # 境界期間内のデータを訓練/評価に分割する割合

# ボキャブラリー設定
# ([bef, base_pos, aft, codon_pos, DNA2Protein[codon], protein_pos, DNA2Protein[new_codon], protein],[freq, hydro, charge, size, blsm])
BASE_VOCABS = {'A':1, 'T':2, 'C':3, 'G':4, 'N':5, 'n':6, 'PAD':0}
AA_VOCABS = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8,
                'H':9, 'I':10, 'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16,
                'T':17, 'W':18, 'Y':19, 'V':20, '*':21, 'n':22, 'PAD':0}
PROTEIN_VOCABS = {'non_coding1':1, 'nsp1':2, 'nsp2':3, 'nsp3':4, 'nsp4':5,
                    'nsp5':6, 'nsp6':7, 'nsp7':8, 'nsp8':9, 'nsp9':10,
                    'nsp10':11, 'nsp12':12, 'nsp13':13, 'nsp14':14,
                    'nsp15':15, 'nsp16':16, 'non_coding2':17, 'S':18,'non_coding3':19,
                    'ORF3a':20, 'non_coding4':21, 'E':22, 'non_coding5':23,'M':24, 
                    'non_coding6':25,'ORF6':26, 'non_coding7':27, 'ORF7a':28,'ORF7b':29,
                    'non_coding8':30,'ORF8':31, 'non_coding9':32,'N':33, 'non_coding10':34,
                    'ORF10':35,'non_coding11':36, 'PAD':0}
VOCAB_TYPES = (["str", # bef
            "int",  # base_pos
            "str",  # aft
            "int",  # codon_pos
            "str",  # DNA2Protein[codon]
            "int",  # protein_pos
            "str",  # DNA2Protein[new_codon]
            "str"   # protein
            ], [
            "float", # freq
            "float", # hydro
            "float", # charge
            "float", # size
            "float"  # blsm
            ])

# モデルのアーキテクチャ設定
NUM_FEATURE_STRING = 8       # number of numerical features
VOCAB_SIZE_POSITION = 30005  # ゲノム位置 (約30,000) + パディング等
VOCAB_SIZE_BASE = 7          # 塩基 (A,T,C,G,N) + パディング
VOCAB_SIZE_AA = 23           # アミノ酸 (20種 + Stop + PAD)
VOCAB_SIZE_CODON_POS = 6   # コドン位置 (例: 0=PAD, 1, 2, 3, 4=None)
VOCAB_SIZE_PROTEIN_POS = 10001 # タンパク質位置 (最大長+1)

NUM_REGIONS = 37             # 予測する領域の総数 (S, N, UTRなど)
NUM_CHEM_FEATURES = 5        # アミノ酸の化学的性質の変化量の数 (例: 疎水性, 電荷, 分子量, BLOSUM62スコア)
                             # 変異頻度（TSごとの重複無しかどうかどっちのcsv使うか）を入れて重複を削除するか、変異頻度入れずに重複も保持するか

EMBED_DIM_POS = 64           # 位置Embeddingの次元
EMBED_DIM_BASE = 16          # 塩基Embeddingの次元
EMBED_DIM_AA = 16            # アミノ酸Embeddingの次元
EMBED_DIM_REGION = 16        # 領域Embeddingの次元
EMBED_DIM_CODON_POS = 8    # コドン位置Embeddingの次元
EMBED_DIM_PROTEIN_POS = 32     # タンパク質位置Embeddingの次元

# 特徴ベクトルの合計次元 (結合後)
FEATURE_DIM = 128

# Transformer設定
HIDDEN_DIM = FEATURE_DIM     # Transformer内部の次元 (入力と合わせる)
N_HEADS = 4                  # アテンション・ヘッドの数
N_LAYERS = 3                 # Transformerエンコーダ層の数
DROPOUT = 0.1

# 訓練設定
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
TOP_K_EVAL = 1               # 評価時にTop-NのNを指定

# 2つのタスクの損失（Loss）に対する重み
LOSS_WEIGHT_REGION = 0.5     # 領域予測タスクの重み
LOSS_WEIGHT_POSITION = 0.5   # 位置予測タスクの重み

#　特徴量生成ログの表示間隔
PRINT_COUNT = 10000

MODEL_SAVE_DIR = './models/transformer_251028/'  # モデルの保存ディレクトリ