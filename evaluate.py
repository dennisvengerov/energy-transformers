"""Evaluate a checkpoint on the two metrics used for every model:

1. GSM8K test accuracy: greedy-decode a program for each question (prompt = TinyGSM docstring
   layout), execute it in a sandboxed subprocess and compare the returned number to the gold answer.
2. Solution perplexity on held-out TinyGSM rows: the model reads the problem (header + docstring)
   and is scored only on the solution tokens (function body + final EOS). Each example is scored on
   its own, without packing, in fp32, so the number is comparable across models.

    python evaluate.py --ckpt runs/L12/ckpt.pt     # writes runs/L12/eval/{results.json, gsm8k_generations.jsonl}
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import common
import grading
import torch
import torch.nn.functional as F
from common import EOS_ID, format_prompt, split_prompt_solution
from model import Transformer


@torch.no_grad()
def generate_greedy(model, prompts, max_new_tokens, batch_size, device, n_vocab):
    """Greedy continuation of each token-id prompt. Returns (tokens before EOS, whether EOS was emitted).

    Prompts are batched only with prompts of the same length, so no padding, attention masks or KV
    cache are needed: every step re-runs the full forward pass, which works for any model whose
    forward(idx) returns next-token logits.
    """
    outputs = [None] * len(prompts)
    by_len = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    for length, idxs in sorted(by_len.items()):
        steps = min(max_new_tokens, model.cfg.max_seq_len - length)
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start:start + batch_size]
            x = torch.tensor([prompts[i] for i in chunk], device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            for _ in range(steps):
                with common.autocast(device):
                    logits = model(x)[:, -1, :n_vocab]  # padded vocab rows are never valid tokens
                nxt = torch.where(done, EOS_ID, logits.argmax(-1))
                x = torch.cat([x, nxt[:, None]], dim=1)
                done |= nxt == EOS_ID
                if done.all():
                    break
            for i, row in zip(chunk, x[:, length:].tolist()):
                outputs[i] = (row[:row.index(EOS_ID)], True) if EOS_ID in row else (row, False)
    return outputs


def eval_gsm8k(model, tok, rows, device, args):
    grading.self_test()
    prompts = [format_prompt(r["question"]) for r in rows]
    token_prompts = [[EOS_ID] + tok.encode(p).ids for p in prompts]
    t0 = time.time()
    gens = generate_greedy(model, token_prompts, args.max_new_tokens, args.gen_batch_size, device, tok.get_vocab_size())
    t_gen = time.time() - t0
    programs = [p + tok.decode(g) for p, (g, _) in zip(prompts, gens)]
    results = grading.run_programs(programs, timeout=args.timeout, workers=args.workers)
    records = []
    for r, program, (_, hit_eos), res in zip(rows, programs, gens, results):
        records.append({
            "question": r["question"], "gold": r["gold"], "program": program, "status": res.status,
            "value": res.value, "detail": res.detail, "hit_eos": hit_eos,
            "correct": grading.is_correct(res, r["gold"]),
        })
    n_correct = sum(r["correct"] for r in records)
    summary = {"accuracy": n_correct / len(records), "n_correct": n_correct, "n": len(records),
               "generation_seconds": round(t_gen, 1)}
    for status in ("ok", "error", "non_numeric", "timeout", "crash"):
        summary[f"n_{status}"] = sum(r["status"] == status for r in records)
    summary["n_no_eos"] = sum(not r["hit_eos"] for r in records)
    return summary, records


@torch.no_grad()
def eval_solution_ppl(model, tok, rows, device, batch_size):
    examples, n_skipped = [], 0
    for r in rows:
        split = split_prompt_solution(r["question"], r["code"])
        if split is None:  # docstring isn't a clean copy of the question (~1% of rows)
            n_skipped += 1
            continue
        prompt_ids = [EOS_ID] + tok.encode(split[0]).ids
        full = [EOS_ID] + tok.encode(r["code"]).ids + [EOS_ID]
        if full[:len(prompt_ids)] != prompt_ids or len(full) - 1 > model.cfg.max_seq_len:
            n_skipped += 1
            continue
        examples.append((full, len(prompt_ids)))
    examples.sort(key=lambda e: len(e[0]))

    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        width = max(len(full) for full, _ in batch) - 1
        x = torch.full((len(batch), width), EOS_ID, dtype=torch.long)
        y = torch.full((len(batch), width), -1, dtype=torch.long)
        for row, (full, n_prompt) in enumerate(batch):
            seq = torch.tensor(full)
            x[row, :len(full) - 1] = seq[:-1]
            y[row, n_prompt - 1:len(full) - 1] = seq[n_prompt:]  # targets = solution tokens only
        x, y = x.to(device), y.to(device)
        logits = model(x)  # fp32 (no autocast): bf16 logits would add ~0.05-nat rounding noise
        total_nll += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction="sum").item()
        total_tokens += (y != -1).sum().item()
    nll = total_nll / total_tokens
    return {"ppl": math.exp(nll), "nll_per_token": nll, "n_examples": len(examples), "n_tokens": total_tokens,
            "n_skipped": n_skipped}


def read_jsonl(path, limit=None):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default=str(common.DATA_DIR))
    p.add_argument("--out-dir", default=None, help="default: <ckpt dir>/eval")
    p.add_argument("--limit", type=int, default=None, help="only the first N GSM8K problems")
    p.add_argument("--ppl-limit", type=int, default=None, help="only the first N held-out TinyGSM rows")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gen-batch-size", type=int, default=64)
    p.add_argument("--ppl-batch-size", type=int, default=32)
    p.add_argument("--timeout", type=float, default=5.0, help="seconds per program")
    p.add_argument("--workers", type=int, default=8, help="programs executed in parallel")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = common.get_device(args.device)
    model, ckpt = Transformer.from_checkpoint(args.ckpt, device)
    tok = common.get_tokenizer()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir or Path(args.ckpt).parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.ckpt}: n_layer={model.cfg.n_layer} params={model.num_params() / 1e6:.1f}M step={ckpt['step']}")

    ppl = eval_solution_ppl(model, tok, read_jsonl(data_dir / "val.jsonl", args.ppl_limit), device, args.ppl_batch_size)
    print(f"TinyGSM held-out solution perplexity: {ppl['ppl']:.4f} ({ppl['n_tokens']:,} tokens, {ppl['n_examples']:,} problems)")
    gsm8k, records = eval_gsm8k(model, tok, read_jsonl(data_dir / "gsm8k_test.jsonl", args.limit), device, args)
    print(f"GSM8K test accuracy: {gsm8k['accuracy']:.4f} ({gsm8k['n_correct']}/{gsm8k['n']})  "
          + " ".join(f"{k}={v}" for k, v in gsm8k.items() if k.startswith("n_") and k != "n_correct"))

    with open(out_dir / "gsm8k_generations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    results = {"ckpt": str(args.ckpt), "n_layer": model.cfg.n_layer, "n_params": model.num_params(), "step": ckpt["step"],
               "gsm8k": gsm8k, "tinygsm_solution_ppl": ppl, "eval_args": vars(args)}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
