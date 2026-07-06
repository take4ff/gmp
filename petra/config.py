# petra/config.py — PETra-style Decoder-only Transformer 設定
#
# PETRA との厳密比較用（本体 transformer_260702 と scale-matched・同一データ・同一時系列分割）。
# 詳細は memory: petra_comparison.md を参照。

# --- DB ---
# None = transformer_260702.db.connection.get_db_path() を使用（config.py の現設定でハッシュ決定）。
# 現物DBが transformer_260702.config の現設定と異なるハッシュの場合は接続失敗するため、
# 既存DBをそのまま使いたい場合はここに直接パスを指定する（例: 'db/features_15d433c4.duckdb'）。
DB_FILE = None
# 'date' = split_type_date 列を参照。これは本体 transformer_260702 が SPLIT_MODE='date' で
# 実行した際に TEMPORAL_SPLIT_DATE を境に確定した列を**そのまま共有**するため、
# 「同一データ・同一時系列カットオフ（リークなし）」が自動的に保証される（再計算不要）。
# 'timestep' は擬似ランダム分割で時系列の公平性が無いため、PETRA比較では 'date' を既定とする。
SPLIT_MODE = 'date'    # 'timestep' | 'date'

# --- モデルアーキテクチャ（scale-matched: 本体 HierarchicalTransformer ≈19.5M params に近づける）---
# 本体は FEATURE_DIM=256/N_LAYERS=4/N_HEADS=4/FFN_RATIO=4 で ~19.46M params
# （その大半は VOCAB_SIZE_POSITION=30006 の位置埋め込み・出力ヘッド由来）。
# petra 側は変異トークン語彙（DB からの動的構築、位置より小さい想定）で埋め込みを共有するため、
# 同じ d_model でも総パラメータ数は語彙サイズ次第で変わる。学習実行時に main.py が
# 'Parameters: N' を出力するので、そこで実測して必要なら D_MODEL/N_LAYERS/FFN_DIM を調整すること。
D_MODEL = 320
N_HEADS = 8
N_LAYERS = 6
FFN_DIM = 1280
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
