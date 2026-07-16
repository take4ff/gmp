# 🔬 Ablation Study（アブレーション実験）— 全フラグ一覧

[← README に戻る](../README.md)

`config.py` 内の以下のフラグで各特徴量やニューラルネットワーク層の有効/無効を切り替え、各コンポーネントが予測性能に与える影響度を精密に測定できます。README には構造系の主要フラグのみ抜粋しているため、`ABLATION_MASKS` の全項目はこちらを参照してください。

```python
USE_LOCAL_CONV1D = False         # Conv1d局所特徴抽出の切り替え
USE_ORIGIN_ATTENTION = False     # Origin Attention（武漢株参照）の切り替え

# Co-occurrence Attention 拡張（共起集約の表現力比較）
CO_ATTN_N_HEADS  = 4            # 共起 Attention のヘッド数
CO_ATTN_N_LAYERS = 1            # 2以上で変異間 Self-Attention を多段化
CO_ATTN_DIM      = 256          # 内部次元
USE_FLAT_COATTN  = False        # True: 全変異を独立トークンとして Transformer に渡す

# コンテキスト窓幅（再前処理が必要）
CONTEXT_WINDOW = 5              # 前後各方向の塩基数。3 / 5 / 7 で切り替え

# 数値・カテゴリ特徴量の個別アブレーションマスク（再前処理不要・動的適用）
ABLATION_MASKS = {
    # --- 物理化学的特徴量 (num[0..5]) ---
    'FREQ': False,    # 変異発生頻度
    'HYDRO': False,   # 疎水性差
    'CHARGE': False,  # 電荷差
    'SIZE': False,    # サイズ差
    'BLSM': False,    # BLOSUM62スコア差
    'PAM250': False,  # PAM250スコア

    # --- ホスト適応特徴量 (num[6..29]) ---
    'HOST_LOG_RATIO_DIFF': False,         # 宿主距離対数比差分
    'HOST_LOG_RATIO_BEFORE': False,       # 変異前宿主距離対数比
    'HUMAN_RSCU_DIFF': False,             # ヒトRSCU差
    'SCV2_RSCU_DIFF': False,              # SCV2 RSCU差
    'OPTIMAL_TO_OPTIMAL': False,          # 最適→最適コドン変化フラグ
    'NON_OPTIMAL_TO_OPTIMAL': False,      # 非最適→最適コドン変化フラグ
    'OPTIMAL_TO_NON_OPTIMAL': False,      # 最適→非最適コドン変化フラグ
    'IS_TRANSITION': False,               # トランジション/トランスバージョン
    'TRANSITION_HUMAN_RSCU_DIFF': False,  # トランジション×ヒトRSCU差
    'TRANSITION_SCV2_RSCU_DIFF': False,   # トランジション×SCV2 RSCU差
    'CPG_DIFF': False,                    # CpGジヌクレオチド変化量
    'UPA_DIFF': False,                    # UpAジヌクレオチド変化量
    'HUMAN_RSCU_BEFORE': False,           # 変異前ヒトRSCU
    'SCV2_RSCU_BEFORE': False,            # 変異前SCV2 RSCU
    'HUMAN_FREQ_BEFORE': False,           # 変異前ヒトコドン頻度
    'HUMAN_FREQ_DIFF': False,             # ヒトコドン頻度差分
    'SCV2_FREQ_BEFORE': False,            # 変異前SCV2コドン頻度
    'SCV2_FREQ_DIFF': False,              # SCV2コドン頻度差分
    'HUMAN_CAI_BEFORE': False,            # 変異前ヒトCAI
    'HUMAN_CAI_DIFF': False,              # ヒトCAI差分
    'SCV2_CAI_BEFORE': False,             # 変異前SCV2 CAI
    'SCV2_CAI_DIFF': False,               # SCV2 CAI差分
    'HOST_RSCU_RATIO_BEFORE': False,      # 変異前宿主RSCU比
    'HOST_RSCU_RATIO_DIFF': False,        # 宿主RSCU比差分

    # --- 累積変異カウント (num[30..31]) ---
    'CUM_SYN': False,    # パス内シノニマス変異累積数
    'CUM_NONSYN': False, # パス内ノンシノニマス変異累積数

    # --- 亜系統の流行ダイナミクス (num[32..35]、USE_LINEAGE_GROWTH_FEATURES=True 時) ---
    'LINEAGE_LOG_COUNT_RECENT': False,  # 直近K週件数（規模）
    'LINEAGE_GROWTH_RATE': False,       # log件数の傾き（勢い）
    'LINEAGE_REL_GROWTH_ADV': False,    # logitシェアの傾き（相対成長優位）
    'LINEAGE_GROWTH_ACCEL': False,      # growth_rate の加速度

    # --- 前後塩基コンテキスト（CONTEXT_WINDOW 分まで個別制御可）---
    'MUTATION_CONTEXT': False,     # 全コンテキスト塩基を一括無効化
    'MUTATION_CONTEXT_L1': False,  # 1文字前 (P-1)
    'MUTATION_CONTEXT_L2': False,  # 2文字前 (P-2)
    'MUTATION_CONTEXT_L3': False,  # 3文字前 (P-3)
    'MUTATION_CONTEXT_L4': False,  # 4文字前 (P-4)
    'MUTATION_CONTEXT_L5': False,  # 5文字前 (P-5)
    'MUTATION_CONTEXT_R1': False,  # 1文字後 (P+1)
    'MUTATION_CONTEXT_R2': False,  # 2文字後 (P+2)
    'MUTATION_CONTEXT_R3': False,  # 3文字後 (P+3)
    'MUTATION_CONTEXT_R4': False,  # 4文字後 (P+4)
    'MUTATION_CONTEXT_R5': False,  # 5文字後 (P+5)
}
```
