"""Tokenize TinyGSM into flat token files and fetch the GSM8K test set.

    python prepare_data.py                                        # full dataset (Armada)
    python prepare_data.py --sample-row-groups 1 --val-every 50   # ~17k streamed rows (local smoke test)

Writes to data/:
    train.bin         uint16 tokens; each document is [EOS] + tokens(code), documents shuffled
    val.bin           held-out documents in the same format (validation loss during training)
    val.jsonl         held-out (question, code) rows (solution perplexity in evaluate.py)
    gsm8k_test.jsonl  GSM8K test questions with numeric gold answers
    meta.json         counts and revisions; written last, so its presence means the data is complete

A row is held out iff md5(question) % val_every == 0, so duplicate questions never straddle the split.
TinyGSM rows are ordered by generation batch (later shards use a different docstring style), so
training documents are shuffled before packing.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import common  # noqa: F401  (project-local caches)
import numpy as np
import pyarrow.parquet as pq
from common import EOS_ID, GSM8K_REPO, GSM8K_REV, TINYGSM_REPO, TINYGSM_REV, TOKENIZER_REV, get_tokenizer
from grading import parse_gold

COLUMNS = ["question", "code"]


def tinygsm_batches(sample_row_groups=None):
    """Yield pyarrow record batches of (question, code).

    Full: download the pinned parquet shards (3.75 GB) into the project's HF cache.
    Sample: stream only the first `sample_row_groups` row groups (1,000 rows each) of every shard.
    """
    from huggingface_hub import HfFileSystem, snapshot_download

    if sample_row_groups is None:
        root = snapshot_download(TINYGSM_REPO, repo_type="dataset", revision=TINYGSM_REV, allow_patterns="data/*.parquet")
        for path in sorted(Path(root, "data").glob("*.parquet")):
            yield from pq.ParquetFile(path).iter_batches(batch_size=10_000, columns=COLUMNS)
    else:
        fs = HfFileSystem()
        prefix = f"datasets/{TINYGSM_REPO}@{TINYGSM_REV}/data"
        for name in sorted(Path(p).name for p in fs.ls(prefix, detail=False) if p.endswith(".parquet")):
            with fs.open(f"{prefix}/{name}", "rb") as f:
                pf = pq.ParquetFile(f)
                for rg in range(min(sample_row_groups, pf.num_row_groups)):
                    yield from pf.read_row_group(rg, columns=COLUMNS).to_batches()


def is_val(question, val_every):
    return int.from_bytes(hashlib.md5(question.encode("utf-8")).digest()[:8], "little") % val_every == 0


def prepare_tinygsm(batches, out_dir, val_every=1000, seed=0):
    tok = get_tokenizer()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "train.unshuffled.tmp"
    train_lengths, n_val_docs, n_val_tokens = [], 0, 0
    with open(tmp_path, "wb") as f_train, open(out_dir / "val.bin", "wb") as f_val, open(out_dir / "val.jsonl", "w") as f_rows:
        for batch in batches:
            questions, codes = batch.column("question").to_pylist(), batch.column("code").to_pylist()
            train_docs, val_docs = [], []
            for question, code, enc in zip(questions, codes, tok.encode_batch(codes)):
                doc = np.array([EOS_ID] + enc.ids, dtype=np.uint16)
                if is_val(question, val_every):
                    val_docs.append(doc)
                    f_rows.write(json.dumps({"question": question, "code": code}) + "\n")
                else:
                    train_docs.append(doc)
                    train_lengths.append(len(doc))
            if train_docs:
                f_train.write(np.concatenate(train_docs).tobytes())
            if val_docs:
                f_val.write(np.concatenate(val_docs).tobytes())
                n_val_docs += len(val_docs)
                n_val_tokens += sum(len(d) for d in val_docs)
            print(f"\rtokenized {len(train_lengths) + n_val_docs:,} documents", end="", flush=True)
    print()

    # Shuffle whole documents: gather them from the unshuffled file in a seeded random order.
    lengths = np.array(train_lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    order = np.random.default_rng(seed).permutation(len(lengths))
    src = np.memmap(tmp_path, dtype=np.uint16, mode="r")
    with open(out_dir / "train.bin", "wb") as f:
        for chunk in np.array_split(order, max(1, len(order) // 100_000)):
            f.write(np.concatenate([src[offsets[i]:offsets[i + 1]] for i in chunk]).tobytes())
    del src
    tmp_path.unlink()
    return {
        "n_train_docs": len(lengths),
        "n_train_tokens": int(lengths.sum()),
        "n_val_docs": n_val_docs,
        "n_val_tokens": n_val_tokens,
    }


def prepare_gsm8k(out_dir):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(GSM8K_REPO, "main/test-00000-of-00001.parquet", repo_type="dataset", revision=GSM8K_REV)
    rows = pq.read_table(path).to_pylist()
    with open(out_dir / "gsm8k_test.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"question": r["question"], "answer": r["answer"], "gold": parse_gold(r["answer"])}) + "\n")
    return len(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(common.DATA_DIR))
    p.add_argument("--sample-row-groups", type=int, default=None, help="stream only this many row groups per shard")
    p.add_argument("--val-every", type=int, default=1000, help="hold out ~1/val_every of the questions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=8, help="tokenizer threads (shared machine: keep modest)")
    args = p.parse_args(argv)
    os.environ["RAYON_NUM_THREADS"] = str(args.num_threads)  # read when the tokenizer first goes parallel

    out_dir = Path(args.out_dir)
    (out_dir / "meta.json").unlink(missing_ok=True)
    stats = prepare_tinygsm(tinygsm_batches(args.sample_row_groups), out_dir, args.val_every, args.seed)
    n_gsm8k = prepare_gsm8k(out_dir)
    meta = {
        **stats,
        "n_gsm8k_test": n_gsm8k,
        "val_every": args.val_every,
        "seed": args.seed,
        "sample_row_groups": args.sample_row_groups,
        "eos_id": EOS_ID,
        "tinygsm_revision": TINYGSM_REV,
        "gsm8k_revision": GSM8K_REV,
        "tokenizer_revision": TOKENIZER_REV,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
