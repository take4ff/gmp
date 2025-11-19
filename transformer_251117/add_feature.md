はい、承知いたしました。
特徴量を追加する場合、それが「数値データ」か「文字列（カテゴリカル）データ」かで、修正が必要なファイルが異なります。

-----

## 📈 数値データ（化学的性質など）を追加する場合

（例：「折り畳みエネルギーの変化量」など、`NUM_CHEM_FEATURES` を3個から4個に増やす場合）

**結論： 修正は `config.py` だけで完了します。** モデルは自動的に対応します。

  * **`config.py` （要修正）**

      * `NUM_CHEM_FEATURES` の値を、新しい特徴量の総数（例: `4`）に変更します。

    <!-- end list -->

    ```python
    # 変更前
    NUM_CHEM_FEATURES = 3
    # 変更後
    NUM_CHEM_FEATURES = 4
    ```

  * **`dataset.py` （要修正：データ読み込み部分）**

      * `ViralDataset` クラスの構造（`padded_num` テンソルの作成）は、`config.py` を参照するため**修正不要**です。
      * ただし、`get_mock_data` を置き換える**ご自身のデータ読み込み関数**で、新しい4番目の数値を読み込み、`num_features` リストに正しく追加するよう修正が**必要**です。

  * **`model.py` （修正不要）**

      * `InputEmbedding` クラスは、`config.py` から渡された `num_chem`（`4`）を自動的に認識します。
      * `self.num_norm = nn.LayerNorm(num_chem)` や `self.projection` 層 の入力次元は、すべて自動で `4` に対応します。

-----

## 🔡 文字列データ（カテゴリカル）を追加する場合

（例：「タンパク質ドメインID」など、新しいID情報を入力に追加する場合）

**結論： `config.py`, `dataset.py`, `model.py` の3ファイルに手動で修正が必要です。**

  * **`config.py` （要修正）**

      * 新しい特徴量の「語彙数（Vocabulary Size）」と「埋め込み次元（Embedding Dimension）」を**新しく定義**します。

    <!-- end list -->

    ```python
    # (...既存の設定...)
    VOCAB_SIZE_AA = 22
    EMBED_DIM_AA = 16

    # --- ▼追加▼ ---
    VOCAB_SIZE_DOMAIN = 50   # 例: ドメインIDが50種類ある場合
    EMBED_DIM_DOMAIN = 10    # 例: ドメインIDを10次元ベクトルにする
    # --- ▲追加▲ ---

    EMBED_DIM_REGION = 16
    ```

  * **`dataset.py` （要修正）**

    1.  **データ読み込み関数（`get_mock_data` を置き換える関数）**:
          * 新しい「ドメインID」を読み込み、`cat_features` リストに追加します。`cat_features` は 6要素から7要素のリストになります。
    2.  **`ViralDataset` クラス**:
          * `__getitem__` 内の `padded_cat` を作成する行の次元を、`6` から `7` に変更します。

    <!-- end list -->

    ```python
    # 変更前
    padded_cat = np.zeros(
        (config.MAX_TRAIN_MAX, config.MAX_CO_OCCURRENCE, 6), dtype=np.int64
    )
    # 変更後
    padded_cat = np.zeros(
        (config.MAX_TRAIN_MAX, config.MAX_CO_OCCURRENCE, 7), dtype=np.int64
    )
    ```

  * **`model.py` （要修正）**

    1.  **`InputEmbedding` クラスの `__init__`**:
          * `__init__` の引数に `vocab_domain, embed_domain` を追加します。
          * 新しい `nn.Embedding` 層（辞書）を定義します。
          * `total_embed_dim` の計算に `embed_domain` を追加します。

    <!-- end list -->

    ```python
    # __init__ 内
    self.aa_embed = nn.Embedding(vocab_aa, embed_aa, padding_idx=0)
    self.region_embed = nn.Embedding(vocab_region, embed_region, padding_idx=0)
    # --- ▼追加▼ ---
    self.domain_embed = nn.Embedding(vocab_domain, embed_domain, padding_idx=0)
    # --- ▲追加▲ ---

    # ...
    total_embed_dim = embed_pos + (embed_base * 2) + (embed_aa * 2) + embed_region + embed_domain + num_chem
    self.projection = nn.Linear(total_embed_dim, feature_dim)
    ```

    2.  **`InputEmbedding` クラスの `forward`**:
          * `x_cat` から新しいID（`...[6]`）を取り出し、ベクトル化します。
          * ベクトル化した `domain` を `torch.cat` のリストに追加します。

    <!-- end list -->

    ```python
    # forward 内
    region = self.region_embed(x_cat[..., 5])
    # --- ▼追加▼ ---
    domain = self.domain_embed(x_cat[..., 6])
    # --- ▲追加▲ ---

    num = self.num_norm(x_num)

    combined = torch.cat([
        pos, base_before, base_after, 
        aa_before, aa_after, 
        region, 
        domain, # ★追加
        num
    ], dim=-1)
    ```

    3.  **`HierarchicalTransformer` クラスの `__init__`**:
          * `InputEmbedding` を呼び出す際に、`config.py` から新しい設定値を渡します。

    <!-- end list -->

    ```python
    # HierarchicalTransformer.__init__ 内
    self.input_embed = InputEmbedding(
        config.VOCAB_SIZE_POSITION, config.VOCAB_SIZE_BASE, 
        config.VOCAB_SIZE_AA,
        config.NUM_REGIONS,
        config.VOCAB_SIZE_DOMAIN,    # ★追加
        config.NUM_CHEM_FEATURES, 
        config.EMBED_DIM_POS, config.EMBED_DIM_BASE,
        config.EMBED_DIM_AA,
        config.EMBED_DIM_REGION,
        config.EMBED_DIM_DOMAIN,     # ★追加
        config.FEATURE_DIM
    )
    ```