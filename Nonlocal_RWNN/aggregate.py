import json, glob, argparse, numpy as np, pandas as pd
from pathlib import Path

def bootstrap_ci(a, iters=10000, alpha=0.05, rng=None):
    rng = np.random.default_rng(rng)
    samples = [rng.choice(a, size=len(a), replace=True).mean() for _ in range(iters)]
    lo, hi = np.percentile(samples, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True,
                    help="directory, which contains the JSON for each seed, e.g.runs/cora/rum_uniform")
    ap.add_argument("--pattern", type=str, default="seed_*.json",
                    help="Wildcards to match JSON filenames")
    ap.add_argument("--save_csv", type=str, default=None,
                    help="Optional: save aggregation results to this CSV path")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.runs_dir) / args.pattern)))
    if not files:
        raise SystemExit(f"No files matched: {args.runs_dir}/{args.pattern}")
    rows = [json.load(open(f)) for f in files]
    df = pd.DataFrame(rows)

    metric = "best_test_acc"
    m = df[metric].mean()
    s = df[metric].std(ddof=1)
    lo, hi = bootstrap_ci(df[metric].to_numpy(), 10000, 0.05)

    print(df[["seed","best_val_acc","best_test_acc"]])
    print("\nSummary")
    print(f"runs_dir: {args.runs_dir}")
    print(f"n_seeds:  {len(df)}")
    print(f"{metric}: mean={m:.4f}  std={s:.4f}  95%CI=({lo:.4f}, {hi:.4f})")

    if args.save_csv:
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "runs_dir": [args.runs_dir],
            "n_seeds": [len(df)],
            "metric": [metric],
            "mean": [m], "std": [s], "ci_lo": [lo], "ci_hi": [hi]
        }
        pd.DataFrame(out).to_csv(args.save_csv, index=False)
        print(f"Saved: {args.save_csv}")

if __name__ == "__main__":
    main()
