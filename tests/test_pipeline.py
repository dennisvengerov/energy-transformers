"""Data prep -> training -> evaluation on a tiny synthetic TinyGSM-style dataset (CPU, no network
beyond the cached tokenizer)."""

import json
from collections import Counter

import numpy as np
import pyarrow as pa
import pytest
import torch

import evaluate
import train
from common import EOS_ID, format_prompt, get_tokenizer, split_prompt_solution
from model import Transformer
from prepare_data import is_val, prepare_tinygsm

TINY_TRAIN = ["--n-layer", "1", "--d-model", "32", "--n-head", "2", "--d-ff", "64", "--seq-len", "128",
              "--batch-size", "4", "--grad-accum", "2", "--max-steps", "6", "--warmup-steps", "2",
              "--log-every", "1", "--eval-every", "3", "--eval-batches", "1", "--device", "cpu", "--no-compile"]


def fake_rows(n):
    rows = []
    for i in range(n):
        a, b = i % 17 + 1, i % 5 + 2  # questions repeat, so duplicates exist
        question = f"Sam has {a} apples. He buys {b} more. How many apples does Sam have?"
        body = f"    apples = {a}\n    bought = {b}\n    result = apples + bought\n    return result\n"
        rows.append({"question": question, "code": format_prompt(question, "'''" if i % 3 else '"""') + body})
    return rows


def split_docs(tokens):
    """uint16 token stream -> list of documents (each starts with EOS)."""
    starts = np.flatnonzero(tokens == EOS_ID).tolist() + [len(tokens)]
    return [tokens[s + 1:e].tolist() for s, e in zip(starts, starts[1:])]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    rows = fake_rows(400)
    batches = (pa.RecordBatch.from_pylist(rows[i:i + 64]) for i in range(0, len(rows), 64))
    stats = prepare_tinygsm(batches, out, val_every=5, seed=0)
    (out / "meta.json").write_text(json.dumps(stats))
    with open(out / "gsm8k_test.jsonl", "w") as f:
        for q, gold in [("Tom has 3 pens. He buys 4 more. How many pens does he have?", 7.0),
                        ("A box has 12 eggs. How many eggs are in 2 boxes?", 24.0)]:
            f.write(json.dumps({"question": q, "answer": f"#### {gold:g}", "gold": gold}) + "\n")
    return out


def test_prepare_split_and_shuffle(data_dir):
    tok, rows = get_tokenizer(), fake_rows(400)
    val_rows = [r for r in rows if is_val(r["question"], 5)]
    train_rows = [r for r in rows if not is_val(r["question"], 5)]
    assert 0 < len(val_rows) < len(rows) // 2
    assert not {r["question"] for r in val_rows} & {r["question"] for r in train_rows}  # duplicates stay together
    meta = json.loads((data_dir / "meta.json").read_text())
    assert (meta["n_train_docs"], meta["n_val_docs"]) == (len(train_rows), len(val_rows))

    train_docs = split_docs(np.fromfile(data_dir / "train.bin", dtype=np.uint16))
    expected = [tok.encode(r["code"]).ids for r in train_rows]
    assert Counter(map(tuple, train_docs)) == Counter(map(tuple, expected))  # same documents...
    assert train_docs != expected  # ...shuffled
    assert sum(len(d) + 1 for d in train_docs) == meta["n_train_tokens"]

    val_docs = split_docs(np.fromfile(data_dir / "val.bin", dtype=np.uint16))
    assert val_docs == [tok.encode(r["code"]).ids for r in val_rows]
    assert [json.loads(line) for line in open(data_dir / "val.jsonl")] == val_rows


def test_token_windows_cover_each_epoch_once(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(1001, dtype=np.uint16).tofile(path)
    windows = train.TokenWindows(path, seq_len=10, seed=3)
    assert windows.n_windows == 100
    for epoch in range(2):
        starts = sorted(int(windows.window(epoch * 100 + i)[0]) for i in range(100))
        assert starts == list(range(0, 1000, 10))
    assert len(windows.window(0)) == 11
    again = train.TokenWindows(path, seq_len=10, seed=3)
    assert [int(again.window(i)[0]) for i in range(150)] == [int(windows.window(i)[0]) for i in range(150)]


def test_resume_is_exact(data_dir, tmp_path, monkeypatch):
    common_args = TINY_TRAIN + ["--data-dir", str(data_dir)]
    train.main(common_args + ["--run-dir", str(tmp_path / "a"), "--ckpt-every", "6"])

    # Interrupt run b during step 4 (after the step-3 checkpoint), then resume it.
    real_step, calls = train.train_step, []

    def crashing_step(*args, **kwargs):
        calls.append(1)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return real_step(*args, **kwargs)

    monkeypatch.setattr(train, "train_step", crashing_step)
    with pytest.raises(KeyboardInterrupt):
        train.main(common_args + ["--run-dir", str(tmp_path / "b"), "--ckpt-every", "3"])
    monkeypatch.setattr(train, "train_step", real_step)
    assert torch.load(tmp_path / "b" / "ckpt.pt", weights_only=True)["step"] == 3
    train.main(common_args + ["--run-dir", str(tmp_path / "b"), "--ckpt-every", "3"])

    a = torch.load(tmp_path / "a" / "ckpt.pt", weights_only=True)
    b = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=True)
    assert a["step"] == b["step"] == 6
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])

    with pytest.raises(SystemExit, match="different settings"):
        train.main(common_args + ["--run-dir", str(tmp_path / "a"), "--lr", "5e-4"])


@pytest.fixture(scope="module")
def tiny_ckpt(data_dir, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run")
    train.main(TINY_TRAIN + ["--data-dir", str(data_dir), "--run-dir", str(run_dir)])
    return run_dir / "ckpt.pt"


def test_ppl_scores_only_solution_tokens_and_ignores_padding(data_dir, tiny_ckpt):
    model, _ = Transformer.from_checkpoint(tiny_ckpt, torch.device("cpu"))
    tok, rows = get_tokenizer(), evaluate.read_jsonl(data_dir / "val.jsonl")
    one = evaluate.eval_solution_ppl(model, tok, rows, torch.device("cpu"), batch_size=1)
    many = evaluate.eval_solution_ppl(model, tok, rows, torch.device("cpu"), batch_size=7)
    assert one["n_tokens"] == many["n_tokens"] and one["ppl"] == pytest.approx(many["ppl"], rel=1e-4)
    prompts = [split_prompt_solution(r["question"], r["code"])[0] for r in rows]
    body_tokens = sum(len(tok.encode(r["code"]).ids) - len(tok.encode(p).ids) + 1 for r, p in zip(rows, prompts))  # +1: final EOS
    assert one["n_tokens"] == body_tokens and one["n_examples"] == len(rows)

    # Reference: score each example's solution directly.
    nll, n = 0.0, 0
    for r, prompt_text in zip(rows, prompts):
        prompt = [EOS_ID] + tok.encode(prompt_text).ids
        full = torch.tensor([[EOS_ID] + tok.encode(r["code"]).ids + [EOS_ID]])
        with torch.no_grad():
            logp = model(full[:, :-1]).log_softmax(-1)[0]
        targets = full[0, 1:]
        nll -= logp[len(prompt) - 1:].gather(1, targets[len(prompt) - 1:, None]).sum().item()
        n += len(targets) - (len(prompt) - 1)
    assert one["nll_per_token"] == pytest.approx(nll / n, rel=1e-5)


def test_batched_generation_matches_one_at_a_time(tiny_ckpt):
    model, _ = Transformer.from_checkpoint(tiny_ckpt, torch.device("cpu"))
    prompts = [[EOS_ID, 5, 6], [EOS_ID, 7, 8], [EOS_ID, 9], [EOS_ID, 1, 2, 3, 4], [EOS_ID, 5, 6]]
    batched = evaluate.generate_greedy(model, prompts, 6, 8, torch.device("cpu"), 50295)
    single = [evaluate.generate_greedy(model, [p], 6, 1, torch.device("cpu"), 50295)[0] for p in prompts]
    assert batched == single


def test_evaluate_end_to_end(data_dir, tiny_ckpt, tmp_path):
    evaluate.main(["--ckpt", str(tiny_ckpt), "--data-dir", str(data_dir), "--out-dir", str(tmp_path),
                   "--max-new-tokens", "8", "--device", "cpu", "--workers", "2"])
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["gsm8k"]["n"] == 2 and 0 <= results["gsm8k"]["accuracy"] <= 1
    assert np.isfinite(results["tinygsm_solution_ppl"]["ppl"])
    records = [json.loads(line) for line in open(tmp_path / "gsm8k_generations.jsonl")]
    assert all(r["program"].startswith(format_prompt(r["question"])) for r in records)
