# petra/config.py — PETra-style Decoder-only Transformer 設定

# --- DB ---
DB_FILE = None          # None = transformer_260625 の get_db_path() を使用
SPLIT_MODE = 'timestep' # 'timestep' | 'date'

# --- モデルアーキテクチャ ---
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
FFN_DIM = 1024
DROPOUT = 0.1
MAX_SEQ_LEN = 512       # 1サンプルの最大トークン長（超過は末尾切り捨て）

# --- 学習 ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MAX_EPOCHS = 30
PATIENCE = 5            # val_loss が改善しないエポック数でearly stopping
WARMUP_STEPS = 1000

# --- 評価 ---
TOP_K_LIST = [1, 5, 10, 50, 100]

# --- その他 ---
DEVICE = 'cuda'
SEED = 42
NUM_WORKERS = 4
CHUNK_SIZE = 2000       # DBストリーミング読み込み単位

# --- パス ---
OUTPUT_DIR = 'outputs/petra'
VOCAB_CACHE = 'cache/petra_vocab.pkl'
