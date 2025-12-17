import torch
from tqdm import tqdm
from . import config
import pandas as pd

@torch.no_grad()
def evaluate(model, dataloader, loss_fn):
    """モデルの評価を行う"""
    model.eval()
    total_loss = 0.0
    
    all_preds_protein = []
    all_preds_pos = []
    all_targets_protein = []
    all_targets_pos = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        # データをデバイスに転送
        x_cat = batch['x_cat'].to(config.DEVICE)
        x_num = batch['x_num'].to(config.DEVICE)
        x_cat_mask = batch['x_cat_mask'].to(config.DEVICE)
        seq_mask = batch['seq_mask'].to(config.DEVICE)
        y_protein = batch['y_protein'].to(config.DEVICE)
        y_pos = batch['y_pos'].to(config.DEVICE)

        # 順伝播
        protein_logits, pos_logits = model(x_cat, x_num, x_cat_mask, seq_mask)
        
        # 損失計算
        loss_protein = loss_fn(protein_logits, y_protein)
        loss_pos = loss_fn(pos_logits, y_pos)
        total_loss += ((config.LOSS_WEIGHT_REGION * loss_protein) + \
                       (config.LOSS_WEIGHT_POSITION * loss_pos)).item()

        # Top-K予測を取得
        topk_preds_protein = torch.topk(protein_logits, config.TOP_K_EVAL, dim=1).indices
        topk_preds_pos = torch.topk(pos_logits, config.TOP_K_EVAL, dim=1).indices
        
        # 結果を保存
        all_preds_protein.append(topk_preds_protein.cpu())
        all_preds_pos.append(topk_preds_pos.cpu())
        all_targets_protein.append(y_protein.cpu())
        all_targets_pos.append(y_pos.cpu())

    # 全バッチの結果を結合
    all_preds_protein = torch.cat(all_preds_protein)
    all_preds_pos = torch.cat(all_preds_pos)
    all_targets_protein = torch.cat(all_targets_protein).unsqueeze(1) # 比較のために次元を追加
    all_targets_pos = torch.cat(all_targets_pos).unsqueeze(1)

    # Top-K ヒット率の計算 (ベクトル化)
    # (B, K) と (B, 1) を比較
    hits_protein = (all_preds_protein == all_targets_protein).any(dim=1).sum().item()
    hits_pos = (all_preds_pos == all_targets_pos).any(dim=1).sum().item()
    
    total_samples = len(all_targets_protein)

    avg_loss = total_loss / len(dataloader)
    protein_accuracy = (hits_protein / total_samples) * 100
    position_accuracy = (hits_pos / total_samples) * 100

    metrics = {
        'loss': avg_loss,
        f'protein_top{config.TOP_K_EVAL}_acc': protein_accuracy,
        f'position_top{config.TOP_K_EVAL}_acc': position_accuracy
    }
    
    return metrics

def evaluate_by_timestep(model, dataloader):
    """
    タイムステップ（original_len）ごとの予測結果を収集するための評価関数。
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []

    for batch in dataloader:
        # データをデバイスに移動
        x_cat = batch['x_cat'].to(device)
        x_num = batch['x_num'].to(device)
        x_cat_mask = batch['x_cat_mask'].to(device)
        seq_mask = batch['seq_mask'].to(device)
        
        # 予測
        protein_logits, pos_logits = model(x_cat, x_num, x_cat_mask, seq_mask)
        
        # Top-K 予測を取得
        _, protein_preds_topk = torch.topk(protein_logits, k=config.TOP_K_EVAL, dim=1)
        _, pos_preds_topk = torch.topk(pos_logits, k=config.TOP_K_EVAL, dim=1)
        
        # 結果をCPUに移動してリストに追加
        for i in range(len(batch['original_len'])):
            all_preds.append({
                'original_len': batch['original_len'][i].item(),
                'true_protein': batch['y_protein'][i].item(),
                'pred_protein_topk': protein_preds_topk[i].cpu().numpy(),
                'true_pos': batch['y_pos'][i].item(),
                'pred_pos_topk': pos_preds_topk[i].cpu().numpy(),
            })
            
    return pd.DataFrame(all_preds)