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
from .evaluate import evaluate_recall_topk
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
        from transformer_260625.db.connection import get_db_path as _get
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
    cache = config.VOCAB_CACHE
    if not force and os.path.exists(cache):
        print(f'Loading tokenizer from {cache}')
        tok = MutationTokenizer.load(cache)
        print(f'Vocab size: {tok.vocab_size}')
        return tok

    print('Building tokenizer from DB...')
    tok = MutationTokenizer()
    tok.build_from_db(db_path, chunk_size=config.CHUNK_SIZE)
    tok.save(cache)
    return tok


# ------------------------------------------------------------------ #
# エントリポイント
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild-vocab', action='store_true',
                        help='語彙キャッシュを強制再構築する')
    parser.add_argument('--eval-only', type=str, default=None, metavar='CKPT',
                        help='学習済みチェックポイントで評価のみ実行')
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    db_path = get_db_path()
    print(f'DB: {db_path}')

    split_col = get_split_col()
    tokenizer = build_tokenizer(db_path, force=args.rebuild_vocab)

    # ---- Datasets ----
    def make_ds(split_type, shuffle):
        return PetraDataset(db_path, tokenizer, split_type=split_type,
                            max_seq_len=config.MAX_SEQ_LEN, shuffle=shuffle,
                            chunk_size=config.CHUNK_SIZE, split_col=split_col)

    train_ds = make_ds(0, shuffle=True)
    val_ds   = make_ds(1, shuffle=False)
    test_ds  = make_ds(2, shuffle=False)
    print(f'Train={len(train_ds):,}  Val={len(val_ds):,}  Test={len(test_ds):,}')

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

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params:,}')

    # ---- Output dir ----
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(config.OUTPUT_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, 'best_model.pth')

    # ---- Eval-only モード ----
    if args.eval_only:
        print(f'Loading checkpoint: {args.eval_only}')
        ckpt = torch.load(args.eval_only, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        _run_evaluation(model, test_loader, tokenizer, device, out_dir)
        return

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

    print('\n--- Training ---')
    for epoch in range(1, config.MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, config.GRAD_CLIP)
        val_loss = evaluate_loss(model, val_loader, device)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} '
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
            print(f'  -> Saved (val_loss={val_loss:.4f})')
        else:
            patience_count += 1
            if patience_count >= config.PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break

    # ---- Evaluate best model ----
    print(f'\n--- Evaluating best model (epoch saved) ---')
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    results = _run_evaluation(model, test_loader, tokenizer, device, out_dir)
    results['best_val_loss'] = best_val_loss

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nResults saved to {out_dir}')
    return results


def _run_evaluation(model, test_loader, tokenizer, device, out_dir) -> dict:
    test_loss = evaluate_loss(model, test_loader, device)
    recall = evaluate_recall_topk(model, test_loader, tokenizer, device,
                                  config.TOP_K_LIST)

    print(f'Test loss (perplexity): {test_loss:.4f} (ppl={math.exp(test_loss):.1f})')
    print('Test Recall@K (mutation tokens only):')
    for k in config.TOP_K_LIST:
        print(f'  @{k:3d}: {recall[f"recall@{k}"]:.4f}')
    print(f'  evaluated on {recall["total_mut_tokens"]:,} mutation tokens')

    results = {'test_loss': test_loss, **recall}
    return results


if __name__ == '__main__':
    main()
