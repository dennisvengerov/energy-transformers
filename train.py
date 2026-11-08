"""Train the Llama-style baseline from scratch on TinyGSM. Depth is the only flag that differs
between the variants; every other default is the shared recipe.

    python train.py --n-layer 12      # writes runs/L12/{ckpt.pt, config.json, log.jsonl}

Re-running the same command resumes from runs/L{n}/ckpt.pt. The batch at each step is a fixed
function of (seed, step), so all depths see identical data in identical order, and resuming is exact.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import common
import numpy as np
import torch
from model import ModelConfig, Transformer

# Arguments that define the experiment; a resumed run must match them exactly.
RECIPE_KEYS = [
    "n_layer", "d_model", "n_head", "d_ff", "seq_len", "batch_size", "grad_accum", "epochs", "max_steps",
    "lr", "min_lr_ratio", "warmup_steps", "weight_decay", "grad_clip", "seed",
]


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-layer", type=int, default=12)
    # Width: keep the defaults for the experiments (smaller values are for local smoke tests).
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=1024)
    # Optimization: 32 x 8 x 1024 = 262,144 tokens per step; 2 epochs ~ 3.6B tokens ~ 13.8k steps.
    p.add_argument("--batch-size", type=int, default=32, help="sequences per micro-batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=None, help="overrides the step count implied by --epochs")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine decays to lr * this")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    # Bookkeeping.
    p.add_argument("--data-dir", default=str(common.DATA_DIR))
    p.add_argument("--run-dir", default=None, help="default: runs/L{n_layer}")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--device", default=None)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None, help="default: on for CUDA")
    return p.parse_args(argv)


class TokenWindows:
    """(seq_len + 1)-token windows over a uint16 token file, visited in a seeded random order per epoch."""

    def __init__(self, path, seq_len, seed):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len, self.seed = seq_len, seed
        self.n_windows = (len(self.tokens) - 1) // seq_len
        self._epoch, self._order = None, None

    def window(self, i):
        """The i-th sample overall (epoch = i // n_windows)."""
        epoch, j = divmod(i, self.n_windows)
        if epoch != self._epoch:
            self._epoch, self._order = epoch, np.random.default_rng([self.seed, epoch]).permutation(self.n_windows)
        start = int(self._order[j]) * self.seq_len
        return self.tokens[start:start + self.seq_len + 1]

    def batch(self, first, size, device):
        return to_xy(np.stack([self.window(i) for i in range(first, first + size)]), device)


def to_xy(windows, device):
    t = torch.from_numpy(windows.astype(np.int64))
    if device.type == "cuda":
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        t = t.to(device)
    return t[:, :-1], t[:, 1:]


def make_optimizer(model, lr, weight_decay, device):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")


def lr_at(step, max_steps, args):
    """Linear warmup, then cosine decay from lr to lr * min_lr_ratio."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def train_step(fwd, model, optimizer, micro_batches, grad_clip, device):
    """One optimizer step over a list of (x, y) micro-batches. Returns (mean loss, grad norm) tensors."""
    loss_sum = 0.0
    for x, y in micro_batches:
        with common.autocast(device):
            loss = fwd(x, y)
        (loss / len(micro_batches)).backward()
        loss_sum += loss.detach()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss_sum / len(micro_batches), grad_norm


@torch.no_grad()
def val_loss(fwd, path, seq_len, batch_size, n_batches, device):
    """Mean loss over the first n_batches * batch_size consecutive windows of val.bin."""
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    n = min(n_batches * batch_size, (len(tokens) - 1) // seq_len)
    losses = []
    for first in range(0, n, batch_size):
        rows = range(first, min(first + batch_size, n))
        x, y = to_xy(np.stack([tokens[i * seq_len:(i + 1) * seq_len + 1] for i in rows]), device)
        with common.autocast(device):
            losses.append(fwd(x, y).item() * len(rows))
    return sum(losses) / n


def save_checkpoint(path, model, optimizer, step, args):
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
         "model_config": asdict(model.cfg), "args": vars(args)},
        tmp,
    )
    os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the previous checkpoint


def main(argv=None):
    args = get_args(argv)
    run_dir = Path(args.run_dir or common.RUNS_DIR / f"L{args.n_layer}")
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    if not (data_dir / "meta.json").exists():
        raise SystemExit(f"{data_dir}/meta.json not found: run prepare_data.py first")
    device = common.get_device(args.device)
    use_compile = device.type == "cuda" if args.compile is None else args.compile
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train = TokenWindows(data_dir / "train.bin", args.seq_len, args.seed)
    samples_per_step = args.batch_size * args.grad_accum
    max_steps = args.max_steps or int(args.epochs * train.n_windows) // samples_per_step

    torch.manual_seed(args.seed)
    cfg = ModelConfig(n_layer=args.n_layer, d_model=args.d_model, n_head=args.n_head, d_ff=args.d_ff, max_seq_len=args.seq_len)
    model = Transformer(cfg).to(device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay, device)
    step = 0
    ckpt_path = run_dir / "ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        changed = {k: (ckpt["args"][k], vars(args)[k]) for k in RECIPE_KEYS if ckpt["args"][k] != vars(args)[k]}
        if changed:
            raise SystemExit(f"{ckpt_path} was trained with different settings {changed}; use another --run-dir")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resumed from {ckpt_path} at step {step}")
        del ckpt
    config = {"args": vars(args), "model_config": asdict(cfg), "max_steps": max_steps, "n_params": model.num_params(),
              "data": json.loads((data_dir / "meta.json").read_text())}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    fwd = torch.compile(model) if use_compile else model

    tokens_per_step = samples_per_step * args.seq_len
    print(f"n_layer={args.n_layer} params={model.num_params() / 1e6:.1f}M device={device} compile={use_compile}")
    print(f"{max_steps} steps x {tokens_per_step:,} tokens = {max_steps * tokens_per_step / 1e9:.2f}B tokens "
          f"({max_steps * samples_per_step / train.n_windows:.2f} epochs of {train.n_windows:,} windows)")

    log = open(run_dir / "log.jsonl", "a")
    t_last, step_last = time.time(), step
    while step < max_steps:
        lr = lr_at(step, max_steps, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        first = step * samples_per_step
        micro = [train.batch(first + m * args.batch_size, args.batch_size, device) for m in range(args.grad_accum)]
        loss, grad_norm = train_step(fwd, model, optimizer, micro, args.grad_clip, device)
        step += 1

        record = None
        if step % args.log_every == 0 or step == max_steps:
            dt = time.time() - t_last
            tok_s = (step - step_last) * tokens_per_step / dt
            record = {"step": step, "loss": loss.item(), "lr": lr, "grad_norm": grad_norm.item(), "tok_per_s": round(tok_s)}
            eta_h = (max_steps - step) * tokens_per_step / tok_s / 3600
            print(f"step {step}/{max_steps} loss {record['loss']:.4f} lr {lr:.2e} gnorm {record['grad_norm']:.2f} "
                  f"{tok_s / 1e3:.0f}k tok/s eta {eta_h:.2f}h", flush=True)
            t_last, step_last = time.time(), step
        if step % args.eval_every == 0 or step == max_steps:
            record = record or {"step": step}
            record["val_loss"] = val_loss(fwd, data_dir / "val.bin", args.seq_len, args.batch_size, args.eval_batches, device)
            print(f"step {step} val_loss {record['val_loss']:.4f}", flush=True)
            t_last, step_last = time.time(), step  # keep eval time out of the throughput numbers
        if record:
            log.write(json.dumps(record) + "\n")
            log.flush()
        if step % args.ckpt_every == 0 or step == max_steps:
            save_checkpoint(ckpt_path, model, optimizer, step, args)
            t_last, step_last = time.time(), step
    print(f"done: {ckpt_path}")


if __name__ == "__main__":
    main()
