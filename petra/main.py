"""PETra-style Decoder-only Transformer のエントリポイント。

Usage:
    python -m petra.main
    python -m petra.main --rebuild-vocab   # 語彙を強制再構築
"""

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import PetraDataset
from .evaluate import evaluate_recall_topk, PetraRepresentativenessWeights
from .model import PetraDecoder
from .tokenizer import MutationTokenizer
from .train import evaluate_loss, train_epoch


# ------------------------------------------------------------------ #
# DB パス解決
# ------------------------------------------------------------------ #

def get_db_path() -> str:
    if config.DB_FILE:
        return config.DB_FILE
    try:
        # 正典版 transformer_260707 と同一 DB を参照する（PETRA比較で train/test を完全一致させるため）
        from transformer_260707.db.connection import get_db_path as _get
        return _get()
    except Exception:
        import glob
        dbs = sorted(glob.glob('db/*.duckdb'))
        if not dbs:
            raise FileNotFoundError('No DuckDB files found in db/')
        return dbs[-1]


def get_split_col() -> str:
    if config.SPLIT_MODE == 'date':
        return 'split_type_date'
    return 'split_type'


# ------------------------------------------------------------------ #
# 語彙構築
# ------------------------------------------------------------------ #

def build_tokenizer(db_path: str, force: bool = False) -> MutationTokenizer:
    """USE_REGION_FIELDS に応じて v1(cache/petra_vocab.pkl) / v2(cache/petra_vocab_v2.pkl)
    のキャッシュを使い分ける。v1実行中プロセスとキャッシュファイルが衝突しないようにするため。
    """
    use_region = getattr(config, 'USE_REGION_FIELDS', False)
    cache = config.VOCAB_CACHE_V2 if use_region else config.VOCAB_CACHE
    if not force and os.path.exists(cache):
        print(f'Loading tokenizer from {cache}')
        tok = MutationTokenizer.load(cache)
        print(f'Vocab size: {tok.vocab_size}')
        return tok

    print(f'Building tokenizer from DB (use_region_fields={use_region})...')
    tok = MutationTokenizer(use_region_fields=use_region)
    codon_csv = config.CODON_CSV if use_region else None
    tok.build_from_db(db_path, chunk_size=config.CHUNK_SIZE, codon_csv=codon_csv)
    tok.save(cache)
    return tok


# ------------------------------------------------------------------ #
# エントリポイント
# ------------------------------------------------------------------ #

def run_training(split_col=None, init_checkpoint=None, out_dir=None,
                  rebuild_vocab=False, log_prefix=''):
    """1回分の学習・評価を実行する（単発実行・walk_forwardの各フォールドの両方から呼ばれる）。

    Args:
        split_col: 参照する split カラム名（None なら get_split_col() に従う）
        init_checkpoint: 初期重みとして読み込むチェックポイント（None ならランダム初期化）
        out_dir: 出力先ディレクトリ（None ならタイムスタンプで自動生成）
        rebuild_vocab: 語彙キャッシュを強制再構築するか
        log_prefix: ログ行の先頭に付与する文字列（walk_forwardのフォールド識別用）

    Returns:
        dict: results（'best_model_path' に今回保存したチェックポイントのパスを含む）
    """
    torch.manual_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'{log_prefix}Device: {device}')

    db_path = get_db_path()
    print(f'{log_prefix}DB: {db_path}')

    split_col = split_col or get_split_col()
    tokenizer = build_tokenizer(db_path, force=rebuild_vocab)

    # ---- Datasets ----
    def make_ds(split_type, shuffle):
        return PetraDataset(db_path, tokenizer, split_type=split_type,
                            max_seq_len=config.MAX_SEQ_LEN, shuffle=shuffle,
                            chunk_size=config.CHUNK_SIZE, split_col=split_col,
                            max_history_steps=config.MAX_HISTORY_STEPS)

    train_ds = make_ds(0, shuffle=True)
    val_ds   = make_ds(1, shuffle=False)
    test_ds  = make_ds(2, shuffle=False)
    print(f'{log_prefix}Train={len(train_ds):,}  Val={len(val_ds):,}  Test={len(test_ds):,}')

    loader_kwargs = dict(
        collate_fn=PetraDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
    )
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE,
                              **loader_kwargs)

    # ---- Model ----
    model = PetraDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        ffn_dim=config.FFN_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT,
    ).to(device)

    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'{log_prefix}Initialized weights from: {init_checkpoint}')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'{log_prefix}Parameters: {n_params:,}')

    # ---- Output dir ----
    if out_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(config.OUTPUT_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, 'best_model.pth')

    # representativeness 重み（country 未整備 DB では全て 1.0 にフォールバックする）
    weights = PetraRepresentativenessWeights(db_path, split_type=2)

    # ---- Optimizer / Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)

    total_steps = config.MAX_EPOCHS * len(train_loader)
    warmup = min(config.WARMUP_STEPS, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training loop ----
    best_val_loss = float('inf')
    patience_count = 0

    print(f'\n{log_prefix}--- Training ---')
    for epoch in range(1, config.MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, config.GRAD_CLIP,
                                 use_amp=getattr(config, 'USE_AMP', True))
        val_loss = evaluate_loss(model, val_loader, device,
                                 use_amp=getattr(config, 'USE_AMP', True))
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f'{log_prefix}Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} '
              f'| lr={lr_now:.2e} | {elapsed:.0f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'vocab_size': tokenizer.vocab_size,
            }, best_model_path)
            print(f'{log_prefix}  -> Saved (val_loss={val_loss:.4f})')
        else:
            patience_count += 1
            if patience_count >= config.PATIENCE:
                print(f'{log_prefix}Early stopping at epoch {epoch}')
                break

    # ---- Evaluate best model ----
    print(f'\n{log_prefix}--- Evaluating best model (epoch saved) ---')
    ckpt = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    results = _run_evaluation(model, test_loader, tokenizer, device, out_dir, weights,
                              log_prefix=log_prefix)
    results['best_val_loss'] = best_val_loss
    results['best_model_path'] = best_model_path
    results['n_params'] = n_params

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'{log_prefix}Results saved to {out_dir}')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild-vocab', action='store_true',
                        help='語彙キャッシュを強制再構築する')
    parser.add_argument('--eval-only', type=str, default=None, metavar='CKPT',
                        help='学習済みチェックポイントで評価のみ実行')
    args = parser.parse_args()

    if args.eval_only:
        torch.manual_seed(config.SEED)
        device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        db_path = get_db_path()
        split_col = get_split_col()
        tokenizer = build_tokenizer(db_path, force=args.rebuild_vocab)
        test_ds = PetraDataset(db_path, tokenizer, split_type=2,
                               max_seq_len=config.MAX_SEQ_LEN, shuffle=False,
                               chunk_size=config.CHUNK_SIZE, split_col=split_col,
                               max_history_steps=config.MAX_HISTORY_STEPS)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                                 collate_fn=PetraDataset.collate_fn,
                                 num_workers=config.NUM_WORKERS,
                                 pin_memory=(device.type == 'cuda'))
        model = PetraDecoder(
            vocab_size=tokenizer.vocab_size, d_model=config.D_MODEL,
            n_heads=config.N_HEADS, n_layers=config.N_LAYERS,
            ffn_dim=config.FFN_DIM, max_seq_len=config.MAX_SEQ_LEN,
            dropout=config.DROPOUT,
        ).to(device)
        print(f'Loading checkpoint: {args.eval_only}')
        ckpt = torch.load(args.eval_only, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        weights = PetraRepresentativenessWeights(db_path, split_type=2)
        out_dir = os.path.dirname(args.eval_only)
        _run_evaluation(model, test_loader, tokenizer, device, out_dir, weights)
        return

    return run_training(rebuild_vocab=args.rebuild_vocab)


def _run_evaluation(model, test_loader, tokenizer, device, out_dir, weights,
                    log_prefix='') -> dict:
    from .evaluate import build_spike_token_ids, build_region_token_ids
    spike_ids = build_spike_token_ids(tokenizer)
    region_ids = build_region_token_ids(tokenizer)

    test_loss = evaluate_loss(model, test_loader, device)
    recall = evaluate_recall_topk(model, test_loader, tokenizer, device,
                                  config.TOP_K_LIST, weights=weights,
                                  spike_token_ids=spike_ids, region_token_ids=region_ids)

    print(f'{log_prefix}Test loss (perplexity): {test_loss:.4f} (ppl={math.exp(test_loss):.1f})')
    print(f'{log_prefix}Test Recall@K [核酸全体] (per-sequence macro-average, n={recall["n_sequences"]:,} sequences):')
    for k in config.TOP_K_LIST:
        print(f'{log_prefix}  @{k:3d}: Average={recall["average"][f"recall@{k}"]:.4f}  '
              f'Weighted={recall["weighted"][f"recall@{k}"]:.4f}')
    if 'spike' in recall:
        print(f'{log_prefix}Test Recall@K [スパイク限定] (n={recall["spike"]["n_sequences"]:,} sequences):')
        for k in config.TOP_K_LIST:
            print(f'{log_prefix}  @{k:3d}: Average={recall["spike"]["average"][f"recall@{k}"]:.4f}  '
                  f'Weighted={recall["spike"]["weighted"][f"recall@{k}"]:.4f}')
    if 'region' in recall:
        print(f'{log_prefix}Test Recall@K [region限定] (n={recall["region"]["n_sequences"]:,} sequences, '
              f'本体の region_hit_rate と対応):')
        for k in config.TOP_K_LIST:
            print(f'{log_prefix}  @{k:3d}: Average={recall["region"]["average"][f"recall@{k}"]:.4f}  '
                  f'Weighted={recall["region"]["weighted"][f"recall@{k}"]:.4f}')

    results = {'test_loss': test_loss, 'recall': recall}
    return results


if __name__ == '__main__':
    main()
