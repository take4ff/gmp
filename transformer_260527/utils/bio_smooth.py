# --- utils/bio_smooth.py --- (transformer_260527)
# Biologically Informed Loss 用のガウススムージング関数
# 塩基位置・AA位置タスクの soft target に位置近傍の確率を分散させる

import torch


def gaussian_smooth_target(soft_vec: torch.Tensor, sigma: float, radius: int = 10) -> torch.Tensor:
    """soft target ベクトルにガウス分布スムージングを施す。

    正解クラスの周辺 ±radius クラスのみ処理するスパース実装のため、
    vocab_size が大きい（30006 など）タスクでも高速に動作する。

    Args:
        soft_vec: FloatTensor [V] - 元の soft target 確率ベクトル (sum ≈ 1.0)
        sigma:    float - ガウスの標準偏差（クラス番号の差の単位）
        radius:   int   - スムージング半径（±radius クラスのみ処理）

    Returns:
        FloatTensor [V] - スムージング後の確率ベクトル（再正規化済み）
        sigma <= 0 の場合は入力をそのまま返す
    """
    if sigma <= 0.0 or soft_vec.sum() == 0:
        return soft_vec

    V = soft_vec.shape[0]
    result = torch.zeros_like(soft_vec)

    # 非ゼロ位置のみ処理（スパース処理で高速化）
    nonzero_indices = soft_vec.nonzero(as_tuple=False).squeeze(1)

    for idx_tensor in nonzero_indices:
        idx = idx_tensor.item()
        weight = soft_vec[idx].item()
        if weight == 0.0:
            continue

        # ±radius の範囲でガウスカーネルを計算
        lo = max(0, idx - radius)
        hi = min(V, idx + radius + 1)
        offsets = torch.arange(lo, hi, dtype=torch.float32, device=soft_vec.device) - idx
        gauss = torch.exp(-0.5 * (offsets / sigma) ** 2)
        gauss = gauss / gauss.sum()  # カーネル内で正規化

        result[lo:hi] += weight * gauss

    # 全体を再正規化（sum=1.0 を保証）
    total = result.sum()
    if total > 0:
        result = result / total

    return result
