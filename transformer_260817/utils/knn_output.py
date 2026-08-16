# --- utils/knn_output.py ---
# 提案12: kNN 検索拡張出力（kNN-LM 型）
#
# 学習後にエンコーダ表現(latest_context)→実ターゲット位置のデータストアを構築し、
# 推論時にテスト表現の近傍 k 件から経験的ターゲット分布 p_kNN を作り、
# モデルの softmax 出力と補間する:  p_final = λ·p_kNN + (1-λ)·p_model
#
# preprocess・学習には一切触れない後付け機構。USE_KNN_OUTPUT=True かつ
# データストア構築済みのときのみ evaluate 側から利用する。
#
# 使い方:
#   1. 学習済みモデルで build_datastore() を呼び、(keys, values) を保存
#   2. 評価時に KNNOutput.load() → interpolate(query_repr, model_logits) で補間
#
# FAISS があれば高速検索を使い、無ければ torch のブルートフォースにフォールバックする。

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


class KNNOutput:
    """kNN データストアと補間ロジック。"""

    def __init__(self, keys, values, vocab_size):
        # keys: [N, D] float32, values: [N] long (position id)
        self.keys = keys
        self.values = values
        self.vocab_size = vocab_size
        self.index = None
        if _HAS_FAISS and keys is not None and len(keys) > 0:
            d = keys.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(np.ascontiguousarray(keys.astype(np.float32)))

    # ---- 構築・保存・ロード ---------------------------------------------------
    @staticmethod
    def datastore_paths():
        base = getattr(config, 'KNN_DATASTORE_PATH', 'cache/knn_datastore')
        return base + '_keys.npy', base + '_values.npy'

    def save(self):
        kpath, vpath = self.datastore_paths()
        os.makedirs(os.path.dirname(kpath) or '.', exist_ok=True)
        np.save(kpath, self.keys)
        np.save(vpath, self.values)
        _log.force_print(f"[INFO] kNN datastore saved: {len(self.values):,} entries → {kpath}")

    @classmethod
    def load(cls, vocab_size):
        kpath, vpath = cls.datastore_paths()
        if not (os.path.exists(kpath) and os.path.exists(vpath)):
            _log.force_print(f"[WARN] kNN datastore not found ({kpath}). kNN output disabled.")
            return None
        keys = np.load(kpath)
        values = np.load(vpath)
        return cls(keys, values, vocab_size)

    # ---- 検索・補間 ----------------------------------------------------------
    def _search(self, query):
        """query: np.ndarray [B, D] → (neighbor_idx [B, k])"""
        k = getattr(config, 'KNN_K', 16)
        k = min(k, len(self.values))
        if self.index is not None:
            _, idx = self.index.search(np.ascontiguousarray(query.astype(np.float32)), k)
            return idx
        # フォールバック: torch ブルートフォース L2
        q = torch.from_numpy(query).float()
        keys = torch.from_numpy(self.keys).float()
        # [B, N] 距離（メモリに注意：大規模データストアでは FAISS 推奨）
        d = torch.cdist(q, keys)
        return torch.topk(d, k, largest=False).indices.numpy()

    def knn_distribution(self, query):
        """query: [B, D] → p_kNN: torch.FloatTensor [B, vocab_size]（近傍の経験分布）"""
        idx = self._search(query)                       # [B, k]
        B = idx.shape[0]
        p = torch.zeros(B, self.vocab_size, dtype=torch.float)
        vals = self.values
        for b in range(B):
            for j in idx[b]:
                pos = int(vals[j])
                if 0 <= pos < self.vocab_size:
                    p[b, pos] += 1.0
        s = p.sum(dim=1, keepdim=True).clamp_min(1e-9)
        return p / s

    def interpolate(self, query_repr, model_logits):
        """モデルの位置ロジットと kNN 経験分布を補間して確率を返す。

        Args:
            query_repr:  [B, D]  エンコーダ表現（latest_context）
            model_logits:[B, V]  Position ヘッドのロジット（Mixture 時は疑似ロジット）
        Returns:
            p_final: [B, V]  補間後の確率分布
        """
        lam = getattr(config, 'KNN_LAMBDA', 0.25)
        if isinstance(query_repr, torch.Tensor):
            query_repr = query_repr.detach().cpu().numpy()
        p_model = torch.softmax(model_logits.detach().cpu(), dim=-1)
        p_knn = self.knn_distribution(query_repr)
        return lam * p_knn + (1.0 - lam) * p_model


def build_datastore(model, dataloader, device):
    """学習済みモデルで (エンコーダ表現, 正解位置) のデータストアを構築する。

    正解が複数（共起）の場合は各正解位置に同じキーを複製して登録する。
    戻り値: KNNOutput インスタンス（呼び出し側で .save() する）。
    """
    model.eval()
    keys_list, vals_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            (x_cat, x_num, mask), y_batch, *_rest = batch
            _ = model(x_cat.to(device), x_num.to(device), src_key_padding_mask=mask.to(device))
            repr_vec = getattr(model, '_last_context', None)
            if repr_vec is None:
                raise RuntimeError(
                    "build_datastore は model._last_context を必要とします。"
                    "forward で latest_context を self._last_context に公開してください。"
                )
            repr_np = repr_vec.detach().cpu().numpy()
            for i, yi in enumerate(y_batch):
                tuples = yi if not isinstance(yi, dict) else []
                for t in tuples:
                    keys_list.append(repr_np[i])
                    vals_list.append(int(t[1]))  # position id
    keys = np.stack(keys_list) if keys_list else np.zeros((0, config.FEATURE_DIM), dtype=np.float32)
    vals = np.array(vals_list, dtype=np.int64)
    return KNNOutput(keys, vals, config.VOCAB_SIZE_POSITION)
