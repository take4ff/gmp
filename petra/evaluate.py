# petra/evaluate.py — Teacher-forced Recall@K 評価

from collections import defaultdict

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_recall_topk(model, loader: DataLoader, tokenizer,
                          device: torch.device, top_k_list: list[int]) -> dict:
    """Teacher-forced Recall@K 評価。

    各位置で正解トークンが top-K 予測に含まれる割合を計算する。
    コンテキストトークン（BOS/EOS/PAD/年・月・クレード）は評価対象外。

    Returns:
        dict: {'recall@1': 0.xx, 'recall@5': ..., 'total_mut_tokens': N}
    """
    model.eval()
    max_k = max(top_k_list)

    # 評価対象外のトークン ID を事前計算
    context_ids = set(
        i for i in range(tokenizer.vocab_size) if tokenizer.is_context_token(i)
    )
    context_tensor = torch.tensor(sorted(context_ids), dtype=torch.long, device=device)

    hits = defaultdict(int)
    total = 0

    for input_ids, target_ids, _ in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)                               # B, T, V
        topk_preds = logits.topk(max_k, dim=-1).indices        # B, T, max_k

        # 変異トークン位置のマスク（コンテキスト・padding は除外）
        is_context = torch.isin(target_ids, context_tensor)
        is_valid = (~is_context) & (target_ids != -100)

        if is_valid.sum() == 0:
            continue

        for k in top_k_list:
            correct = (topk_preds[..., :k] == target_ids.unsqueeze(-1)).any(-1)
            hits[k] += (correct & is_valid).sum().item()

        total += is_valid.sum().item()

    metrics = {f'recall@{k}': hits[k] / total if total > 0 else 0.0
               for k in top_k_list}
    metrics['total_mut_tokens'] = total
    return metrics
