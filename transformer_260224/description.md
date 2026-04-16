# transformer_260224

`transformer_260112` からの主要な変更点をまとめます。

## 概要

DuckDBを使用したデータベースアーキテクチャにより、大規模データ（900万サンプル）を効率的に処理。

---

## v260224 での変更点

### 1. Ablationマスクの改修
数値特徴量6つ（FREQ, HYDRO, CHARGE, SIZE, BLSM, PAM250）を個別にマスク可能に変更。
共起・アミノ酸変異・コドン位置のマスクは削除。

### 2. DuckDB データベースアーキテクチャ (v260112から継続)

メモリ効率を大幅に改善するため、pickleキャッシュからDuckDBに移行。

#### 2フェーズ実行

```
Phase 1: python -m transformer_260224.preprocess
  → 特徴量生成 → features.duckdb 保存

Phase 2: python -m transformer_260224.main
  → DB読み込み → 学習 → 評価
```

#### 3フェーズ実行
```
Step 1: 特徴量生成（高速・シリアル）
python -m transformer_260224.preprocess_cache
nohup python -m transformer_260224.preprocess_cache > nohup_cache.out &

Step 2: DB変換（ストリーミング）
python -m transformer_260224.preprocess_db
nohup python -m transformer_260224.preprocess_db > nohup_db.out &

Step 3: 学習
python -m transformer_260224.main
nohup python -m transformer_260224.main > nohup_main.out &
```

#### テーブル構成

| テーブル | 説明 |
|---------|------|
| `strains` | 株マスタ（流行度含む） |
| `samples` | サンプル（パス長、共起数） |
| `features` | 特徴量（タイムステップ別） |
| `labels` | ラベル（予測対象） |
| `schema_info` | スキーマバージョン |

### 3. 新規ファイル (v260112で追加)

| ファイル | 説明 |
|---------|------|
| `db_utils.py` | DB操作ユーティリティ |
| `preprocess.py` | 特徴量生成 → DB保存 |

### 4. メリット

- 全データをRAMに保持しない
- 前処理は1回、学習は何度でも再実行可能
- SQLでフィルタリング（共起数、パス長など）

---

## 実行方法

### 初回（DB構築）

```bash
python -m transformer_260224.preprocess
```

### 学習

```bash
python -m transformer_260224.main
```

---

## 設定パラメータ

```python
# config.py
DB_DIR = 'cache/db'
USE_DB = True  # Trueの場合、DuckDBを使用
```

