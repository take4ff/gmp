# 特徴量重要度・生物学的 XAI 分析

[← README に戻る](../README.md)

学習済みチェックポイントに対する後付け分析。すべて単独 CLI（`--checkpoint` 指定、`--force_cpu` 対応）で、共通基盤 `scripts/analysis/_xai_common.py`（config_snapshot 反映・モデル/ローダ・出力保存を集約）の上に載る。

```bash
# 特徴量重要度（数値特徴32/36次元）
python -m transformer_260707.scripts.analysis.feature_importance --checkpoint <best.pth>        # Gradient×Input（感度）＋埋め込みノルム＋Co-Attn重み
python -m transformer_260707.scripts.analysis.permutation_importance --checkpoint <best.pth> \
    --metric position_hit_rate --n_batches 50 --n_repeats 2                                      # 精度寄与（再学習不要）

# 生物学的 XAI（位置→遺伝子 annotation・外部データと突き合わせ）
python -m transformer_260707.scripts.analysis.genome_track --checkpoint <best.pth>               # ①位置/遺伝子別の予測質量 → ゲノム地図
python -m transformer_260707.scripts.analysis.homoplasy_alignment --checkpoint <best.pth>        # ②重要度 vs homoplasy 相関＋problematic 負コントロール
python -m transformer_260707.scripts.analysis.cooccurrence_constellation --checkpoint <best.pth> # ③共予測ペア vs 実共起（変異株コンステレーション復元）
python -m transformer_260707.scripts.analysis.local_explain --checkpoint <best.pth>              # ④Integrated Gradients による1予測の局所説明
python -m transformer_260707.scripts.analysis.probe_representation --checkpoint <best.pth>       # ⑤内部表現の線形プロービング（gene/Spike/同義 を復元できるか）

# ↑の新規5本を一括実行（出力を xai_all/<ts>/<analysis>/ に集約、失敗継続）
python -m transformer_260707.scripts.analysis.run_all_xai --checkpoint <best.pth> --split test
#   --only / --skip で対象選択、--n_batches で共通上書き、--force_cpu、--dry_run（コマンド確認のみ）
```

| 分析 | 何を見るか | 主な外部データ |
|---|---|---|
| `genome_track` | モデルが Spike/RdRp 等の意味ある領域に注目しているか | `codon_mutation4.csv`（遺伝子annotation） |
| `homoplasy_alignment` | 重要部位が**正の選択下の収斂部位**と一致するか／アーチファクト部位を回避しているか | `homoplasy/site_recurrence.csv`, `problematic_sites…vcf` |
| `cooccurrence_constellation` | 既知ハプロタイプ（連鎖・エピスタシス）を再発見しているか | — |
| `local_explain` | 個々の予測を「どの過去変異・特徴が押し上げたか」に帰属 | — |
| `probe_representation` | 内部表現に生物概念（遺伝子/Spike/同義）が符号化されているか | — |
