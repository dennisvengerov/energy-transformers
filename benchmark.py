"""Measure training throughput on this GPU with the real recipe (same model, batch, compile and
optimizer step as train.py, on random tokens) and estimate wall-clock time of a full run per depth.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
"""

import argparse
import json
import time

import common
import torch
from model import ModelConfig, Transformer
from train import get_args, make_optimizer, train_step

# Training tokens if data/meta.json doesn't exist yet: 11.85M documents x ~152.5 tokens
# (measured on a 48k-row sample from all 17 shards), minus the ~0.1% held out.
ESTIMATED_TRAIN_TOKENS = 1.805e9


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depths", type=int, nargs="+", default=[12, 6, 4])
    p.add_argument("--warmup", type=int, default=5, help="untimed steps (includes compilation)")
    p.add_argument("--steps", type=int, default=20, help="timed optimizer steps")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    recipe = get_args([])  # train.py defaults = the experiment recipe

    device = common.get_device(args.device)
    if device.type != "cuda":
        print("WARNING: no CUDA device; numbers below are not representative.")
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, bf16 {torch.cuda.is_bf16_supported()}")

    meta_path = common.DATA_DIR / "meta.json"
    if meta_path.exists():
        n_train_tokens = json.loads(meta_path.read_text())["n_train_tokens"]
        source = "data/meta.json"
    else:
        n_train_tokens, source = ESTIMATED_TRAIN_TOKENS, "estimate (data not prepared yet)"
    samples_per_step = recipe.batch_size * recipe.grad_accum
    tokens_per_step = samples_per_step * recipe.seq_len
    n_windows = int(n_train_tokens - 1) // recipe.seq_len
    max_steps = int(recipe.epochs * n_windows) // samples_per_step
    print(f"recipe: {recipe.epochs} epochs x {n_train_tokens / 1e9:.2f}B tokens ({source}) = {max_steps:,} steps "
          f"x {tokens_per_step:,} tokens; micro-batch {recipe.batch_size} x {recipe.seq_len}, grad accum {recipe.grad_accum}")

    rows = []
    for n_layer in args.depths:
        torch.manual_seed(0)
        cfg = ModelConfig(n_layer=n_layer, d_model=recipe.d_model, n_head=recipe.n_head, d_ff=recipe.d_ff, max_seq_len=recipe.seq_len)
        model = Transformer(cfg).to(device)
        optimizer = make_optimizer(model, recipe.lr, recipe.weight_decay, device)
        fwd = torch.compile(model) if device.type == "cuda" else model
        data = torch.randint(0, cfg.vocab_size, (recipe.grad_accum, recipe.batch_size, recipe.seq_len + 1), device=device)
        micro = [(d[:, :-1], d[:, 1:]) for d in data]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for i in range(args.warmup + args.steps):
            if i == args.warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
            loss, _ = train_step(fwd, model, optimizer, micro, recipe.grad_clip, device)
        loss.item()  # wait for the GPU
        dt = (time.time() - t0) / args.steps
        tok_s = tokens_per_step / dt
        mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
        rows.append((n_layer, model.num_params() / 1e6, tok_s, dt, mem, max_steps * dt / 3600))
        print(f"n_layer={n_layer}: {tok_s / 1e3:.0f}k tok/s, {dt:.2f} s/step, peak mem {mem:.1f} GB", flush=True)
        del model, optimizer, fwd, data, micro
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'n_layer':>7} {'params':>8} {'tok/s':>9} {'s/step':>7} {'peak GB':>8} {'full run':>9}")
    for n_layer, params, tok_s, dt, mem, hours in rows:
        print(f"{n_layer:>7} {params:>7.1f}M {tok_s / 1e3:>8.0f}k {dt:>7.2f} {mem:>8.1f} {hours:>8.1f}h")
    print(f"{'total':>7} {'':>8} {'':>9} {'':>7} {'':>8} {sum(r[5] for r in rows):>8.1f}h")
    print("(training only; evaluation is not included)")


if __name__ == "__main__":
    main()
