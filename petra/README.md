# petra/ — PETRA比較用ベースライン実装

本体モデル（`transformer_260707`）との厳密比較のために構築した、PETRA
（arXiv:2511.03976 / `petra/PETra/`）型のdecoder-only Transformerベースライン。

## 位置づけ

「PETRA論文の数値を再現する再現実験」ではなく、「共起集約Attentionを持たない
decoder-onlyベースラインを、本体モデルと同一データ・同一評価プロトコル・
同等パラメータ規模で学習し、共起集約Attention導入の効果を切り出す」ための
controlled ablationベースラインという位置づけ。詳細は
`PLAN_abstract_strengthening.md`のPhase Eセクションを参照。

## 構成

- `tokenizer.py` — 変異パスのトークン化
- `dataset.py` — DuckDBからのストリーム読み込み（`PetraDataset`）
- `model.py` — Decoder-only Transformer本体（`PetraDecoder`）
- `evaluate.py` — PETRA準拠のRecall@K評価
- `main.py` — 学習エントリポイント（`run_training()`）
- `walk_forward.py` — 本体と同じFOLDS構成でのwalk-forward学習
- `eval_tail_by_daterange.py` — 変異パス末尾のみの評価（`split_type_wf`列を使わず
  日付範囲で直接クエリするため、他フォールドの学習と並行実行可能）
- `plot_monthly_hitrate.py` — walk-forward結果の月別Hit Rateプロット

## 原論文との準拠状況

| 項目 | 状況 |
|---|---|
| 評価指標（Recall@1/10/100、per-sequence macro平均、representativeness重み） | 準拠済み |
| 採取国（representativeness重み付け） | 準拠済み（`samples.country`が100%整備済み、Average≠Weightedで重み付けが機能していることを確認済み） |
| Megatron不使用 | 問題なし（Megatronは並列化基盤であり、単一GPUでの同一層構造の再現で数学的に等価） |
| モデル規模・語彙サイズ | 意図的に本体モデル（~19-37Mパラメータ）にscale-matchさせ、原論文公式規模（512次元/12層〜1024次元/24層）には合わせない方針を継続。パラメータ数を揃えて共起集約Attentionの効果だけを切り出すため |

### v2実装済み（2026-07-08。v1のwalk-forward実行中に並行実装、次回学習から使用）

現在実行中のv1 walk-forward（`cache/petra_vocab.pkl`使用）とは完全に分離した形で実装済み。
v1のキャッシュ・学習には一切影響しない（`petra_config.USE_REGION_FIELDS`で切替、
v2は`cache/petra_vocab_v2.pkl`を使用）。

- **spike限定評価**: 本体の`petra_recall.py`と同じく、核酸全体とスパイク領域(21563-25384)を
  `petra/evaluate.py`で別々に報告するよう追加済み（`build_spike_token_ids`、
  `evaluate_recall_topk`/`evaluate_recall_topk_tail_only`の`spike`キー）。
- **RoPE化**: `petra/model.py`を全面書き換え。学習可能な絶対位置埋め込みを廃し、
  `CausalSelfAttentionRoPE`（`F.scaled_dot_product_attention`+RoPE回転）による
  decoder blockに置換。因果性（未来トークンが過去の出力に影響しないこと）を
  検証済み。
- **9フィールド化（6フィールド版、region予測を含む）**: 原論文
  （`petra/PETra/data_preprocess/utils.py`の`mutation_encoding()`, version=0）の
  `[mut, pos, nuc_mut, anno, anno_pos, anno_mut, overlap_anno, overlap_anno_pos,
  overlap_anno_mut]`のうち、overlap系3フィールド（本体に対応データ無し）を省いた
  `[mut, pos, nuc_mut, region, region_pos, region_mut]`を実装。
  `reference/codon/codon_mutation4.csv`から位置の全域を列挙して語彙構築
  （DBスキャン不要）。実データで検証済み: A23403G(D614G)→
  `[REGION_S][REGIONPOS_S_614][REGIONMUT_DG]`、C14408T(P323L)→
  `[REGION_nsp12][REGIONPOS_nsp12_323][REGIONMUT_PL]`と正しくデコードされることを
  確認。region語彙数=37で本体の`NUM_REGIONS=37`と一致。
  語彙サイズは132,335まで増加（原論文公式規模184,531〜192,710に近づいた）。
- **region単位のRecall@K**: spike限定評価と同じパターンで`evaluate.py`に追加
  （`build_region_token_ids`、`region`キー）。実データで`n_sequences>0`となり
  正しくregionトークン位置を検出することを確認済み。abstractの「遺伝子領域精度」と
  直接比較できるようになった。
- **AMP（bfloat16自動混合精度）**: 語彙拡大(132,335)+変異1件が6トークンに展開される
  ことによる実効系列長増加（実測: 62変異のサンプルで67トークン→377トークン、
  約5.6倍）で、logitsテンソル(B,T,V)の活性化メモリがv1比で大幅増（推定:
  batch=64,T=512上限でv1=12.1GB→v2=17.35GB、+5.24GB）。v1学習は既にVRAM
  20.2GB/24.5GB使用中で余裕が乏しかったため、`petra/train.py`にAMPを追加し
  概ね半減させた。bfloat16はfp32と同じ指数範囲を持つためGradScaler不要。
  `petra_config.USE_AMP`（既定True）で制御。

### 今後の展望（未着手・優先度低）

- **Bloom推定器ベースライン（生物物理学的フィットネス推定量）との比較**:
  PETRA原論文自体も`evaluate_bloom.py`でBloomら（[SARS2-mut-fitness](
  https://github.com/jbloomlab/SARS2-mut-fitness)）のDMS（deep mutational
  scanning）由来フィットネス推定量と比較している。このDMSデータは手元に無く、
  外部調達が必要なため未着手。機械学習を使わない生物物理学的手法との比較という
  観点で価値はあるが、既に一様ランダム・多数決・Fanoエントロピー上限・PETRA
  アーキテクチャ比較の4種類の基準があるため、優先度は低い。データ入手ができた
  際の追加比較候補として記録しておく。

- **学習時の重複排除(dedup)とcountry特徴量の相互作用**（2026-07-08検討・対応不要と判断）:
  本体と同じ`raw_path`完全一致による重複排除（`USE_UNIQUE_FILTER`相当）は、DB全体で見ると
  **サンプル数の50.0%が複数国にまたがる重複グループに属しており**、dedup後は1グループにつき
  1カ国分のサンプルしか残らない（例: 最大の重複グループは240万件・70カ国にまたがるが、
  dedup後は1件・1カ国のみ残る）。PETRAは`[COUNTRY_xxx]`トークンを入力特徴量として使うため、
  理論上はこの多様性喪失が学習信号を痩せさせる懸念があった。しかし原論文
  （`petra_article.pdf` Figure 5, 4.4.2節）の「PETRA」vs「PETRA-No-Location」アブレーション
  実測で、location特徴量の寄与はNuc Recall@1で0.0857→0.0945（+0.88pt、1%未満）と非常に
  小さいことが示されている。この結果から、dedupによる多様性喪失が最終的な精度に与える影響も
  同程度に小さいと判断し、**学習側のdedupは現状維持**とした（本体モデルとの重複排除方式の
  一貫性も保たれる）。評価側のWeighted Recall（representativeness重み付け、Table 2で
  Average比▲2〜3pt/相対13〜17%と別途確認済み）は、location特徴量の話とは別の集計方式の
  論点であり、今回のabstractでは厳密な精度比較を求められていないため、あわせて今後の課題
  として先送りする。

### 実行して初めてわかる項目（2026-07-08時点・保留）

以下は実際に動かさないと判断できないため、v1 walk-forward完走後まで保留:

1. **v2（RoPE+9フィールド+AMP）の実学習が未実施**: 小さいダミーモデルでのsmoke testのみ
   検証済み。長時間の実学習で初めて顕在化する問題（VRAM超過、勾配安定性、収束）は未検証。
2. **`plot_monthly_hitrate.py`が一度も実行されていない**: 実装・import確認のみ。
3. **v2はMAX_SEQ_LEN=512の切り詰めがv1より頻繁に起きる**: 変異1件が6トークンに展開されるため
   長いパスほど末尾が切れやすい。上限を増やすとVRAMがさらに圧迫されるトレードオフが未解決。

対応方針: 上記3点は実際にv2を動かして問題が出た時点で対処する。**VRAM 32GBのサーバも
利用可能**なため、24GBで不足する場合はそちらで検証する選択肢がある。
