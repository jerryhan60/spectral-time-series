#!/usr/bin/env python3
"""Bootstrap significance tests for hint preconditioning experiments.

Computes bootstrap confidence intervals for the relative improvement of
hint-preconditioned models vs baseline on GIFT-Eval (97 configs).
"""
import argparse
import glob
import sys
import numpy as np
import pandas as pd


def geo_mean(x):
    return np.exp(np.log(x).mean())


def bootstrap_ci(baseline_mases, model_mases, n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for relative improvement (model/baseline - 1)."""
    rng = np.random.RandomState(seed)
    n = len(baseline_mases)
    assert len(model_mases) == n

    ratios = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        bl_geo = geo_mean(baseline_mases[idx])
        md_geo = geo_mean(model_mases[idx])
        ratios.append(md_geo / bl_geo)

    ratios = np.array(ratios)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(ratios, [alpha * 100, (1 - alpha) * 100])
    mean_ratio = np.mean(ratios)
    p_value = np.mean(ratios >= 1.0)  # fraction where model is worse
    return {
        "mean_ratio": mean_ratio,
        "ci_lo": lo,
        "ci_hi": hi,
        "improvement_pct": (1 - mean_ratio) * 100,
        "ci_lo_pct": (1 - hi) * 100,
        "ci_hi_pct": (1 - lo) * 100,
        "p_value": p_value,
        "n_configs": n,
    }


def load_gifteval_results(csv_path):
    """Load GIFT-Eval all_results CSV and extract MASE values."""
    df = pd.read_csv(csv_path)
    # The all_results files have columns like dataset/freq/term and MASE[0.5]
    mase_col = [c for c in df.columns if "MASE" in c.upper() or "mase" in c]
    if not mase_col:
        raise ValueError(f"No MASE column in {csv_path}. Columns: {df.columns.tolist()}")
    mase_col = mase_col[0]
    return df[mase_col].dropna().values, df


def main():
    results_dir = "/scratch/gpfs/EHAZAN/jh1161/gifteval/results"

    # Find all_results files
    all_files = sorted(glob.glob(f"{results_dir}/all_results_*.csv"))
    print("Available all_results files:")
    for f in all_files:
        name = f.split("/")[-1]
        try:
            mases, _ = load_gifteval_results(f)
            print(f"  {name}: {len(mases)} configs, geo_mean={geo_mean(mases):.4f}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Key comparisons
    print("\n" + "=" * 70)
    print("BOOTSTRAP SIGNIFICANCE TESTS (10000 resamples, 95% CI)")
    print("=" * 70)

    # Identify baseline file
    baseline_candidates = [f for f in all_files if "baseline" in f.lower() or
                           (f.endswith("epoch_99-step_10000.csv") and "stu" not in f.lower())]

    # Load all results and match by config
    result_files = sorted(glob.glob(f"{results_dir}/all_results_*.csv"))

    # Try to find baseline by loading the gifteval results CSVs and matching
    gifteval_files = sorted(glob.glob(f"{results_dir}/gifteval_results_*.csv"))

    # Hardcode known comparisons based on experiment summary
    comparisons = {
        "ms_d4d6_10k": "q_ms_d4d6",
        "hd10_100k": "m2_hd10_100k",
        "d6_10k": "q_hint_s16d6",
        "d4_10k": "q_hint_s16d4",
    }

    # Try loading pairs of files for comparison
    for name, pattern in comparisons.items():
        model_files = [f for f in gifteval_files if pattern in f]
        if model_files:
            model_file = model_files[-1]  # Most recent
            print(f"\n--- {name} ({model_file.split('/')[-1]}) ---")
            try:
                model_df = pd.read_csv(model_file)
                mase_cols = [c for c in model_df.columns if "mase" in c.lower()]
                if mase_cols:
                    mase_col = mase_cols[0]
                    vals = model_df[mase_col].dropna().values
                    print(f"  Configs: {len(vals)}, Geo Mean MASE: {geo_mean(vals):.4f}")
                    print(f"  MASE < 1.0: {(vals < 1.0).sum()}/{len(vals)}")
            except Exception as e:
                print(f"  Error: {e}")

    # FEV-Bench comparison
    print("\n" + "=" * 70)
    print("FEV-BENCH RESULTS")
    print("=" * 70)

    fev_files = sorted(glob.glob(f"{results_dir}/fev_bench_*.csv"))
    fev_results = {}
    for f in fev_files:
        name = f.split("fev_bench_")[-1].replace(".csv", "").rsplit("_", 1)[0]
        try:
            df = pd.read_csv(f)
            mase_col = "MASE" if "MASE" in df.columns else "mase"
            mases = df[mase_col].dropna().values
            fev_results[name] = {
                "file": f,
                "n_tasks": len(mases),
                "geo_mean": geo_mean(mases),
                "mean": np.mean(mases),
                "mases": mases,
                "task_names": df["task_name"].values if "task_name" in df.columns else None,
            }
            print(f"  {name}: {len(mases)} tasks, MASE Geo={geo_mean(mases):.4f}, Mean={np.mean(mases):.4f}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Pairwise bootstrap on FEV results
    if "baseline_10k" in fev_results:
        bl = fev_results["baseline_10k"]
        print(f"\n--- FEV Bootstrap vs baseline_10k (Geo={bl['geo_mean']:.4f}) ---")
        for name, res in sorted(fev_results.items()):
            if name == "baseline_10k":
                continue
            # Match by task name
            if res["task_names"] is not None and bl["task_names"] is not None:
                bl_df = pd.DataFrame({"task": bl["task_names"], "mase_bl": bl["mases"]})
                md_df = pd.DataFrame({"task": res["task_names"], "mase_md": res["mases"]})
                merged = bl_df.merge(md_df, on="task")
                if len(merged) > 0:
                    ci = bootstrap_ci(merged["mase_bl"].values, merged["mase_md"].values)
                    sig = "***" if ci["p_value"] < 0.001 else "**" if ci["p_value"] < 0.01 else "*" if ci["p_value"] < 0.05 else "ns"
                    print(f"  {name}: {ci['improvement_pct']:+.2f}% [{ci['ci_lo_pct']:+.2f}%, {ci['ci_hi_pct']:+.2f}%] p={ci['p_value']:.4f} {sig} ({ci['n_configs']} tasks)")


if __name__ == "__main__":
    main()
