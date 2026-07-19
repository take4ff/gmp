# petra/config.py — PETra-style Decoder-only Transformer 設定
#
# PETRA との厳密比較用（本体 transformer_260707 と scale-matched・同一データ・同一時系列分割）。
# 詳細は memory: petra_comparison.md を参照。

# --- DB ---
# None = transformer_260707.db.connection.get_db_path() を使用（config.py の現設定でハッシュ決定）。
# 現物DBが transformer_260707.config の現設定と異なるハッシュの場合は接続失敗するため、
# 既存DBをそのまま使いたい場合はここに直接パスを指定する（例: 'db/features_15d433c4.duckdb'）。
DB_FILE = None
# 'date' = split_type_date 列を参照。これは本体 transformer_260707 が SPLIT_MODE='date' で
# 実行した際に TEMPORAL_SPLIT_DATE を境に確定した列を**そのまま共有**するため、
# 「同一データ・同一時系列カットオフ（リークなし）」が自動的に保証される（再計算不要）。
# 'timestep' は擬似ランダム分割で時系列の公平性が無いため、PETRA比較では 'date' を既定とする。
SPLIT_MODE = 'date'    # 'timestep' | 'date'

# --- モデルアーキテクチャ（scale-matched: 本体 HierarchicalTransformer ≈19.5M params に近づける）---
# 本体は FEATURE_DIM=256/N_LAYERS=4/N_HEADS=4/FFN_RATIO=4 で ~19.46M params
# （その大半は VOCAB_SIZE_POSITION=30006 の位置埋め込み・出力ヘッド由来）。
# PetraDecoderは埋め込み/出力ヘッドを重み共有(vocab_size×d_model)しており、これがパラメータの
# 大半(現vocab=92,332なら約80%)を占める。本体と同じN_LAYERS=4/N_HEADS=4/FFN_RATIO=4の形状のまま
# d_modelだけをvocab_sizeに応じて調整し、総パラメータ数を本体に合わせる
# （2026-07-19実測: D_MODEL=320では36.9Mと本体の約1.9倍だったため、192に縮小して19.51Mに調整済み。
# vocab_sizeが変わった場合は再実測して調整すること）。
D_MODEL = 192
N_HEADS = 4
N_LAYERS = 4
FFN_DIM = 768
DROPOUT = 0.1
MAX_SEQ_LEN = 512       # 1サンプルの最大トークン長（超過は末尾切り捨て）
# 本体(transformer_260707.config.MAX_SEQ_LEN=39)と同じ「直近何タイムステップ(">"区切り)を
# 入力に使うか」の上限。Noneなら制限なし(=系統樹の根からの全履歴、従来動作)。
# 本体は raw_path.split('>')[-MAX_HISTORY_STEPS:] で直近だけを残すため、ここも同じ切り出し方に揃える。
MAX_HISTORY_STEPS = 39

# --- 学習 ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MAX_EPOCHS = 30
PATIENCE = 5            # val_loss が改善しないエポック数でearly stopping
WARMUP_STEPS = 1000
USE_AMP = True          # bfloat16自動混合精度。v2は語彙拡大+系列長増でVRAM圧迫のため既定で有効

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

# --- v2（次バージョン）: region予測を含む9フィールド簡略版・RoPE化・spike限定評価 ---
# 現在進行中の walk_forward 実行（v1、VOCAB_CACHEを使用中）とは別のキャッシュファイルを
# 使うこと。同じファイルを上書きすると、実行中プロセスが次フォールドでキャッシュを再読込した際に
# 語彙サイズが変わり、フォールド間のチェックポイント引き継ぎが壊れる。
USE_REGION_FIELDS = False       # True で6フィールド化（mut/pos/nuc_mut/region/region_pos/region_mut）
VOCAB_CACHE_V2 = 'cache/petra_vocab_v2.pkl'
CODON_CSV = 'reference/codon/codon_mutation4.csv'

# --- 原論文 mutation_encoding(version=1) 準拠オプション ---
# 原論文の実際のheadlineモデル(116M/458Mパラメータ)は、[mut]トークンを元の塩基(ref)を
# 無視して(position, 変異後塩基)のみで決定する（A1TとG1Tは同一トークン）。
# 現状のUSE_REGION_FIELDS=Falseの実装は生の変異文字列(例"A23403G", ref込み)をそのまま
# トークン化しており、この点だけ原論文と異なる。Trueにすると、USE_REGION_FIELDSの値に
# 関わらず[mut]トークンの決定方法だけをversion=1準拠に切り替える（直交フラグ）。
# 既存のv1実行(walk_forward済み)とキャッシュ・語彙サイズが変わるため別キャッシュを使う。
DROP_REF_BASE = False
VOCAB_CACHE_NOREF = 'cache/petra_vocab_noref.pkl'
VOCAB_CACHE_V2_NOREF = 'cache/petra_vocab_v2_noref.pkl'
