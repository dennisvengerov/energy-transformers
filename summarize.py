"""Table of every evaluated run (runs/*/eval/results.json) and its gap to the deepest model.

    python summarize.py
"""

import json
import math
from pathlib import Path


def main():
    runs = []
    for path in Path(__file__).resolve().parent.glob("runs/*/eval/results.json"):
        runs.append((path.parent.parent.name, json.loads(path.read_text())))
    if not runs:
        raise SystemExit("no runs/*/eval/results.json yet")
    runs.sort(key=lambda r: -r[1]["n_layer"])
    ref = runs[0][1]
    print(f"{'run':<8} {'layers':>6} {'params':>8} {'step':>7} {'GSM8K acc':>10} {'Δ acc (pts)':>12} "
          f"{'sol. PPL':>9} {'Δ NLL/token':>12}")
    for name, r in runs:
        acc, nll = r["gsm8k"]["accuracy"], r["tinygsm_solution_ppl"]["nll_per_token"]
        d_acc = 100 * (acc - ref["gsm8k"]["accuracy"])
        d_nll = nll - ref["tinygsm_solution_ppl"]["nll_per_token"]
        print(f"{name:<8} {r['n_layer']:>6} {r['n_params'] / 1e6:>7.1f}M {r['step']:>7} {100 * acc:>9.1f}% {d_acc:>+12.1f} "
              f"{math.exp(nll):>9.4f} {d_nll:>+12.4f}")
    print(f"(gaps relative to {runs[0][0]}; GSM8K n={ref['gsm8k']['n']})")


if __name__ == "__main__":
    main()
