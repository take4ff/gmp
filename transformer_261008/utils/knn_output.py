# --- utils/knn_output.py ---
# 提案12: kNN 検索拡張出力（kNN-LM 型）
#
# 学習後にエンコーダ表現(latest_context)→ターゲット位置分布のデータストアを構築し、
# 推論時にクエリ表現の近傍 k 件から経験的な位置分布 p_kNN を作り、
# モデルの softmax 出力と補間する:  p_final = λ·p_kNN + (1-λ)·p_model
#
# preprocess・モデル学習には一切触れない後付け機構。USE_KNN_OUTPUT=True かつ
# データストア構築済みのときのみ evaluate 側から利用する。
#
# 改訂 2026-10-06: 旧実装は build_datastore が y_batch のタプルリストだけを読んでおり、
# HYBRID_ALPHA=1.0（Soft Target、y が dict）では常に空のデータストアになっていた。
# ・ラベルは「1キー(=1共起グループの代表表現)に対する位置分布」を疎(CSR)形式で保持する
#   （Soft Target の position ベクトルの非ゼロ要素＝学習ターゲットそのもの）。
#   旧方式のようにターゲット数ぶんキーを複製しないので、巨大共起グループでもメモリが膨らまない。
# ・Hard Target(タプルリスト)の場合は一意な位置に均等重みを割り当てて同じ形式に変換する。
#
# 使い方:
#   1. scripts/eval/build_knn_datastore.py で (keys, ラベル) を構築・保存
#   2. 評価時に KNNOutput.load() → blend(query_repr, position_scores) で補間（evaluate.py に配線済み）
#
# FAISS があれば高速検索、無ければ torch のチャンク分割ブルートフォースにフォールバックする。

import os
import numpy as np
import torch

from .. import config
from . import logging as _log

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


def labels_to_sparse(y_item, position_vocab_size):
    """1サンプルのラベルを (positions[int64], weights[float32]) の疎表現にする。

    y_item: Soft Target の場合は dict（'position': FloatTensor [V]）、
            Hard Target の場合は ターゲットタプルのリスト [(region, pos, aa_pos, codon, syn, ...), ...]。
    重みは合計 1 に正規化する（空なら空配列）。範囲外の位置は捨てる。
    """
    if isinstance(y_item, dict):
        vec = y_item['position']
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()
        pos = np.nonzero(vec > 0)[0].astype(np.int64)
        w = vec[pos].astype(np.float32)
    else:
        counts = {}
        for t in (y_item or []):
            p = int(t[1])
            if 0 <= p < position_vocab_size:
                counts[p] = counts.get(p, 0) + 1
        pos = np.array(sorted(counts), dtype=np.int64)
        w = np.array([counts[p] for p in pos], dtype=np.float32)
    s = float(w.sum())
    if s > 0:
        w = w / s
    return pos, w


class KNNOutput:
    """kNN データストアと補間ロジック。

    keys    : [N, D] float32（代表サンプルのエンコーダ表現）
    indptr  : [N+1] int64（CSR）  positions/weights の key i 分は indptr[i]:indptr[i+1]
    positions: [nnz] int64、weights: [nnz] float32（各keyの位置分布、key毎に合計1）
    """

    def __init__(self, keys, indptr, positions, weights, vocab_size):
        self.keys = keys
        self.indptr = indptr
        self.positions = positions
        self.weights = weights
        self.vocab_size = vocab_size
        self.index = None
        if _HAS_FAISS and keys is not None and len(keys) > 0:
            self.index = faiss.IndexFlatL2(keys.shape[1])
            self.index.add(np.ascontiguousarray(keys.astype(np.float32)))

    def __len__(self):
        return 0 if self.keys is None else len(self.keys)

    # ---- 構築・保存・ロード ---------------------------------------------------
    @staticmethod
    def datastore_paths():
        base = getattr(config, 'KNN_DATASTORE_PATH', 'cache/knn_datastore')
        return [base + s for s in ('_keys.npy', '_indptr.npy', '_positions.npy', '_weights.npy')]

    def save(self):
        paths = self.datastore_paths()
        os.makedirs(os.path.dirname(paths[0]) or '.', exist_ok=True)
        for path, arr in zip(paths, (self.keys, self.indptr, self.positions, self.weights)):
            np.save(path, arr)
        _log.force_print(f"[INFO] kNN datastore saved: {len(self):,} keys, "
                         f"{len(self.positions):,} label entries → {paths[0]}")

    @classmethod
    def load(cls, vocab_size):
        paths = cls.datastore_paths()
        if not all(os.path.exists(p) for p in paths):
            _log.force_print(f"[WARN] kNN datastore not found ({paths[0]}). kNN output disabled.")
            return None
        keys, indptr, positions, weights = (np.load(p) for p in paths)
        if len(keys) == 0:
            _log.force_print("[WARN] kNN datastore is empty. kNN output disabled.")
            return None
        return cls(keys, indptr, positions, weights, vocab_size)

    # ---- 検索・補間 ----------------------------------------------------------
    def _search(self, query):
        """query: np.ndarray [B, D] → neighbor_idx [B, k]（faissは不足分を -1 で埋める）"""
        k = min(getattr(config, 'KNN_K', 16), len(self))
        q = np.ascontiguousarray(query.astype(np.float32))
        if self.index is not None:
            _, idx = self.index.search(q, k)
            return idx
        qt = torch.from_numpy(q)
        keys = torch.from_numpy(self.keys).float()
        out = []
        for s in range(0, len(qt), 64):                      # メモリ節約のためクエリをチャンク分割
            d = torch.cdist(qt[s:s + 64], keys)
            out.append(torch.topk(d, k, largest=False).indices.numpy())
        return np.concatenate(out, axis=0)

    def knn_distribution(self, query):
        """query: [B, D] → p_kNN: FloatTensor [B, vocab_size]（近傍の位置分布の平均）"""
        idx = self._search(query)
        B = idx.shape[0]
        p = np.zeros((B, self.vocab_size), dtype=np.float32)
        for b in range(B):
            valid = [j for j in idx[b] if j >= 0]
            for j in valid:
                s, e = self.indptr[j], self.indptr[j + 1]
                np.add.at(p[b], self.positions[s:e], self.weights[s:e])
            if valid:
                p[b] /= len(valid)
        return torch.from_numpy(p)

    def blend(self, query_repr, scores, allowed_mask=None):
        """モデルの位置スコア(ロジット)と kNN 分布を補間した確率 p_final [B, V] を返す。

        scores       : [B, V] ロジット（階層マスク適用後ならマスク外は極小値）
        allowed_mask : [B, V] bool または None。与えた場合は p_kNN も許可位置に限定する
                       （階層的予測のマスクを kNN 側で破らないため）。
        """
        lam = getattr(config, 'KNN_LAMBDA', 0.25)
        if isinstance(query_repr, torch.Tensor):
            query_repr = query_repr.detach().cpu().numpy()
        p_model = torch.softmax(scores.detach().float(), dim=-1)
        p_knn = self.knn_distribution(query_repr).to(p_model.device)
        if allowed_mask is not None:
            p_knn = p_knn * allowed_mask.to(p_knn.dtype)
        return lam * p_knn + (1.0 - lam) * p_model

    def interpolate(self, query_repr, model_logits):
        """後方互換: マスク無しの補間。"""
        return self.blend(query_repr, model_logits).cpu()


def build_datastore(model, dataloader, device, max_groups=None, log_every=200):
    """学習済みモデルで (代表表現, 位置分布) のデータストアを構築する。

    dataloader の各item（1共起グループの代表サンプル）が1キー。ラベルは Soft Target の
    position 分布（dict）または Hard Target のタプルリストから疎表現に変換する。
    max_groups を超えたら打ち切る（パイロット用。呼び出し側で sample_ids を無作為抽出しておくこと）。
    戻り値: KNNOutput（呼び出し側で .save() する）。
    """
    model.eval()
    keys_list, pos_list, w_list, lens = [], [], [], []
    n = 0
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            (x_cat, x_num, mask), y_batch, *_rest = batch
            model(x_cat.to(device), x_num.to(device), src_key_padding_mask=mask.to(device))
            repr_vec = getattr(model, '_last_context', None)
            if repr_vec is None:
                raise RuntimeError(
                    "build_datastore は model._last_context を必要とします。"
                    "forward で latest_context を self._last_context に公開してください。")
            repr_np = repr_vec.detach().cpu().numpy().astype(np.float32)
            for i, yi in enumerate(y_batch):
                pos, w = labels_to_sparse(yi, config.VOCAB_SIZE_POSITION)
                if len(pos) == 0:
                    continue                      # ラベルが無いキーは検索結果を汚すだけなので捨てる
                keys_list.append(repr_np[i])
                pos_list.append(pos)
                w_list.append(w)
                lens.append(len(pos))
                n += 1
                if max_groups is not None and n >= max_groups:
                    break
            if bi % log_every == 0:
                _log.force_print(f"  build_datastore: batch {bi}, keys={n:,}")
            if max_groups is not None and n >= max_groups:
                break
    if n == 0:
        keys = np.zeros((0, config.FEATURE_DIM), dtype=np.float32)
        return KNNOutput(keys, np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64),
                         np.zeros(0, dtype=np.float32), config.VOCAB_SIZE_POSITION)
    indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return KNNOutput(np.stack(keys_list), indptr, np.concatenate(pos_list),
                     np.concatenate(w_list), config.VOCAB_SIZE_POSITION)
