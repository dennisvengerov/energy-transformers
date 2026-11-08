# Depth-gap baseline: Llama-style transformers trained from scratch on TinyGSM

Steps 1–2 of the energy-head project: train a 12-layer, 123.6M-parameter Llama-style decoder from
scratch on [TinyGSM](https://huggingface.co/datasets/TinyGSM/TinyGSM), train the identical model with
4 layers (6 as a backup), and measure the gap on two metrics:

1. **GSM8K test accuracy** (1,319 problems): the model writes `simple_math_problem()`, which is executed
   in a sandboxed subprocess and compared numerically with the gold answer.
2. **Held-out TinyGSM solution perplexity**: perplexity of the solution tokens given the problem.
   This is the continuous metric for measuring how much of the gap the energy head closes later.

Sanity target for the 12-layer baseline: the TinyGSM paper (arXiv 2312.09241, Table 2) reports
**53.1%** for a 125M model from random init. (The 63.1% in their Table 1 is for a pretrained model.)

## Commands on Armada, in order

```bash
# 1. Clone
git clone git@github.com:<you>/energy-transformer.git && cd energy-transformer

# 2. Environment: ./.venv with torch 2.14.1 built for CUDA 12.6 (~6-7 GB, nothing cached outside the project).
#    Uses python3.11 from PATH (e.g. after `module load python/3.11`), otherwise falls back to conda.
bash scripts/setup_env.sh

# 3. GPU benchmark (~5-10 min, mostly compilation): tokens/sec and estimated wall-clock of a full run for each depth
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
#    or through SLURM: srun --gres=gpu:1 --cpus-per-task=4 .venv/bin/python benchmark.py

# 4. Data (CPU only, needs internet, roughly 20-40 min). Downloads 3.75 GB of parquet into .cache/,
#    writes data/. Then the raw parquet can be deleted.
.venv/bin/python prepare_data.py
rm -rf .cache/huggingface/hub/datasets--TinyGSM--TinyGSM

# 5. Tests (optional, ~20 s on CPU)
.venv/bin/python -m pytest -q

# 6. Train + evaluate each depth, 12 layers first (it is the sanity check). Either SLURM...
mkdir -p logs
sbatch --job-name=L12 --export=ALL,N_LAYER=12 scripts/train.sbatch
sbatch --job-name=L4  --export=ALL,N_LAYER=4  scripts/train.sbatch
sbatch --job-name=L6  --export=ALL,N_LAYER=6  scripts/train.sbatch
#    ...or a plain background command, one depth after another on the GPU you were given:
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup bash -c 'bash scripts/run_depth.sh 12 && bash scripts/run_depth.sh 4 && bash scripts/run_depth.sh 6' > logs/depths.log 2>&1 &

# 7. Results table (accuracy, perplexity, and gaps relative to the 12-layer model)
.venv/bin/python summarize.py
```

- `scripts/run_depth.sh N` runs `train.py --n-layer N` and then `evaluate.py --ckpt runs/LN/ckpt.pt`.
  Evaluation alone: `.venv/bin/python evaluate.py --ckpt runs/L12/ckpt.pt`.
- **Resuming:** re-running the same command (or resubmitting the job) continues from `runs/LN/ckpt.pt`,
  which is saved every 1,000 steps (roughly 15–30 min of training). Resuming is exact, and a test checks this.
  If the lab's SLURM time limit is under 24 h, edit `--time` in `scripts/train.sbatch` and resubmit when a job hits it.
- **Progress:** `logs/*.out` (or `logs/depths.log`); per-step metrics in `runs/LN/log.jsonl`.
- **GPU use:** the code only uses `cuda:0`, the first device in `CUDA_VISIBLE_DEVICES`. SLURM sets this
  for you; with the plain command, set it yourself. Tokenization and code execution use 8 CPU threads.
- **Time:** run `benchmark.py` to get real numbers. A FLOP-based guess for 2 epochs is about 7 h for
  12 layers and 3–4 h each for 4 and 6 layers. Evaluation should take minutes per model, but that
  hasn't been measured on a GPU.

## What is identical across depths

Everything except `--n-layer`: width (768, 12 heads, SwiGLU 2048), tokenizer, data, token budget
(same number of steps), batch, learning-rate schedule, seed, data order (the batch at step *s* is a
fixed function of seed and *s*), init (no depth-dependent scaling), and evaluation. A test checks that
the 4- and 12-layer state dicts differ only in blocks 4–11.

| | 12 layers | 6 layers | 4 layers |
|---|---|---|---|
| parameters (tied embeddings) | 123.6M | 81.1M | 67.0M |
| non-embedding parameters | 85.0M | 42.5M | 28.3M |

## Decisions, and what was checked in the raw data

**Prompt format.** I streamed 48k rows from all 17 TinyGSM shards. Each `code` is
`def simple_math_problem() -> int:` followed by a docstring with the question split into one sentence
per line, indented 4 spaces. The split is done by the common regex `split_into_sentences` splitter
(it doesn't split after `Mr.`, `p.m.`, `$2.50` or single initials). `common.format_prompt` reproduces
the docstring of **47,550 / 47,550** rows whose docstring is a clean copy of `question`, byte for byte.
Tokenizing the prompt alone gives an exact token prefix of the training document for every one of
those rows. In the other ~1.2%, the generator mangled whitespace. Those rows stay in training but are
left out of the perplexity set. Quote style is `'''` in ~88% of rows and `"""` in the rest (found only
in shards 0–4), so GSM8K prompts use `'''`. Tests pin this format with verbatim TinyGSM rows.

**Data order.** The rows are grouped by generation batch (for example, shards 5–16 are all `'''`, and
~28% of their programs include `print(f'<<<{result}>>>')`). `prepare_data.py` therefore shuffles whole
documents before packing.

**Tokenizer.** Phi-1.5/CodeGen BPE (50,295 tokens, pinned revision), the tokenizer TinyGSM's own models
use. Its whitespace-run tokens make Python indentation cheap: ~152 tokens per program vs ~176 with
GPT-2. The full dataset is ≈1.81B tokens, matching the paper's 1.8B.

**Model.** Llama-style pre-norm decoder in plain PyTorch (`model.py`, ~130 lines): RMSNorm, RoPE,
SwiGLU, no biases, tied input/output embeddings, `F.scaled_dot_product_attention` (flash attention on
the A100), context 1024.

**Training.** Each document is `[EOS] + tokens(code)`. Documents are concatenated and cut into
1024-token windows, with no cross-document masking (standard GPT-style packing; neighbours are
unrelated after the shuffle). The loss covers all tokens. AdamW (β = 0.9/0.95, wd 0.1 on matrices),
peak LR 1e-3, 500 warmup steps, cosine decay to 1e-4, grad clip 1.0, bf16 autocast, `torch.compile`.
Batch is 32 × 8 accumulation × 1024 = 262k tokens. **2 epochs ≈ 3.6B tokens ≈ 13.8k steps** (the
paper's from-scratch run saw ~1.7 epochs: 20k steps × 1,024 examples). Fixed seed 0.

**Held-out split.** A row is held out iff `md5(question) % 1000 == 0` (~11.8k rows), so duplicate
questions never straddle train and validation.

**GSM8K evaluation.** The prompt is `[EOS] + format_prompt(question)`. Decoding is greedy (the paper
notes that sampling is >10 points worse), up to 512 new tokens, stopping at EOS. Same-length prompts
are batched together, so no padding or KV cache is needed. Evaluation first checks that the sandbox can run a
known-good program. The program (prompt + completion) runs in a fresh `python -I` subprocess: empty
temp dir, minimal environment, 5 s timeout with the process group
killed, plus CPU, memory (1 GB) and file-size (0) limits. Its own prints are discarded. The return
value must be a real number (bools, strings, None, NaN and inf count as wrong) and must match the gold
answer within 1e-4 (40 and 40.0 both match 40). Errors, timeouts and crashes count as wrong.
Per-problem programs and outcomes are written to `runs/LN/eval/gsm8k_generations.jsonl`. This sandbox
contains accidents from our own model; it isn't hardened against adversarial code.

**Solution perplexity.** For each held-out row, the input is `[EOS] + prompt + solution + [EOS]`,
scored on the solution tokens and the final EOS only. Each example is scored on its own (no packing,
no neighbouring documents) in fp32, so the number doesn't depend on batch size or bf16 rounding. A test checks this against a
per-example reference computation. `results.json` reports `ppl` and `nll_per_token` (mean nats per
token). Use `nll_per_token` differences when comparing models.

## Disk usage on Armada

| what | size |
|---|---|
| `.venv` (torch + CUDA libraries) | ~6–7 GB |
| `.cache/huggingface` (TinyGSM parquet; delete after step 4) | 3.75 GB |
| `data/` (`train.bin` 3.6 GB + validation + GSM8K) | ~3.7 GB, plus 3.6 GB temporarily during prep |
| `runs/L12`, `L6`, `L4`: one checkpoint each, overwritten in place (weights + Adam state) | 1.5 + 1.0 + 0.8 GB |
| `.cache/torchinductor`, `.cache/triton`, logs, eval outputs | < 1 GB |
| **total** | **~18 GB, ~14 GB after deleting the parquet; peak ~22 GB during prep** |

The Hugging Face, inductor and triton caches are inside the project. pip runs with `--no-cache-dir`, and
the conda package cache (only used if python3.11 isn't on PATH) is in `.cache/conda-pkgs`.

## For step 3 (energy head)

```python
model, ckpt = Transformer.from_checkpoint("runs/L4/ckpt.pt", device)
H = model.hidden(idx)        # (B, T, 768) final hidden states, after the last RMSNorm
logits = model.lm_head(H)    # lm_head.weight is W_U (tied to the input embedding)
# model(idx) == model.lm_head(model.hidden(idx))
```

Both evaluation paths (generation and perplexity) only call `model(idx)` and read `model.cfg.max_seq_len`.
A subclass whose `forward` returns `lm_head(energy_head(hidden(idx)))` is therefore evaluated by the
same code, unchanged. Adding the head's parameters to the checkpoint is the only loading change.

## Files

| file | purpose |
|---|---|
| `common.py` | project-local caches, device choice, tokenizer, TinyGSM prompt format |
| `model.py` | the Llama-style model |
| `prepare_data.py` | TinyGSM → `data/train.bin`, `val.bin`, `val.jsonl`; GSM8K test → `data/gsm8k_test.jsonl` |
| `train.py` | training (`--n-layer` is the experiment flag) |
| `evaluate.py` | GSM8K accuracy + solution perplexity → `runs/LN/eval/results.json` |
| `grading.py` | sandboxed execution and numeric grading |
| `benchmark.py` | throughput and wall-clock estimates |
| `summarize.py` | results table across depths |
| `scripts/` | `setup_env.sh`, `run_depth.sh`, `train.sbatch`, `smoke_test.sh` (local) |
| `tests/` | format, grading, model, and the full pipeline on synthetic data (CPU) |

## Local development (Mac, CPU/MPS)

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python torch==2.14.1 -r requirements.txt
.venv/bin/python -m pytest -q
bash scripts/smoke_test.sh   # streams ~17k TinyGSM rows into data/smoke, trains a tiny model, evaluates
```
