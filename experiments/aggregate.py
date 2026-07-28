"""Aggregate benchmark results into tables and figures.

Reports the interquartile mean with 95% stratified-bootstrap confidence
intervals (Agarwal et al. 2021) rather than bare means, plus solve rates,
sample efficiency, wall-clock and solution size.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neon.stats import (iqm, stratified_bootstrap_ci, bootstrap_ci, normalize,
                        performance_profile, load_reference_scores)
from neon.registry import METHODS
from neon.harness import TASK_SPEC

RUNS = "results/runs"
FIGS = "results/figures"


def load():
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def final_score(run, frac=3):
    """Mean of the last few evaluation points, to damp evaluation noise."""
    pts = [c["score"] for c in run["curve"][-frac:]]
    return float(np.mean(pts))


def build(runs, ref):
    """rows[(method, env)] = dict of per-seed arrays."""
    rows = {}
    for r in runs:
        k = (r["method"], r["env"])
        d = rows.setdefault(k, dict(norm=[], raw=[], solved=[], steps=[],
                                    wall=[], params=[], hidden=[]))
        rr = ref[r["env"]]
        fs = final_score(r)
        d["raw"].append(fs)
        d["norm"].append(normalize(fs, rr["random"], rr["solve"]))
        d["solved"].append(1.0 if r["solved"] else 0.0)
        d["steps"].append(r["solved"]["steps"] if r["solved"] else r["budget"])
        d["wall"].append(r["wall"])
        d["params"].append(r["curve"][-1].get("params", np.nan))
        d["hidden"].append(r["curve"][-1].get("hidden", np.nan))
    return rows


def table_per_env(rows, envs, methods):
    out = []
    for env in envs:
        out.append(f"\n### {env}\n")
        out.append("| method | final score | normalized IQM [95% CI] | "
                   "solve rate | median steps to solve | median wall (s) | "
                   "size (params) | n |")
        out.append("|---|---|---|---|---|---|---|---|")
        for m in methods:
            d = rows.get((m, env))
            if not d:
                continue
            n = len(d["raw"])
            raw = np.mean(d["raw"])
            pt, lo, hi = stratified_bootstrap_ci({env: d["norm"]}, iqm)
            sr = np.mean(d["solved"])
            ss = np.median(d["steps"])
            sstr = f"{ss:,.0f}" + ("" if sr == 1.0 else " (censored)")
            out.append(
                f"| {m} | {raw:.1f} | {pt:.3f} [{lo:.3f}, {hi:.3f}] | "
                f"{sr:.0%} | {sstr} | {np.median(d['wall']):.1f} | "
                f"{np.nanmedian(d['params']):.0f} | {n} |")
    return "\n".join(out)


def table_overall(rows, envs, methods):
    out = ["\n### Aggregate across all tasks\n",
           "| method | normalized IQM [95% CI] | mean solve rate | "
           "median wall (s) | median size |",
           "|---|---|---|---|---|"]
    for m in methods:
        by_task = {e: rows[(m, e)]["norm"] for e in envs if (m, e) in rows}
        if not by_task:
            continue
        pt, lo, hi = stratified_bootstrap_ci(by_task, iqm)
        sr = np.mean([np.mean(rows[(m, e)]["solved"]) for e in envs
                      if (m, e) in rows])
        wall = np.median(np.concatenate(
            [rows[(m, e)]["wall"] for e in envs if (m, e) in rows]))
        size = np.nanmedian(np.concatenate(
            [rows[(m, e)]["params"] for e in envs if (m, e) in rows]))
        out.append(f"| {m} | {pt:.3f} [{lo:.3f}, {hi:.3f}] | {sr:.0%} | "
                   f"{wall:.1f} | {size:.0f} |")
    return "\n".join(out)


def figures(runs, rows, ref, envs, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGS, exist_ok=True)

    # learning curves with bootstrap CI bands
    ncol = min(3, len(envs))
    nrow = int(np.ceil(len(envs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, env in zip(axes, envs):
        budget = TASK_SPEC[env]["budget"]
        grid = np.linspace(budget / 20, budget, 20)
        for m in methods:
            rs = [r for r in runs if r["method"] == m and r["env"] == env]
            if not rs:
                continue
            curves = []
            for r in rs:
                s = [c["steps"] for c in r["curve"]]
                v = [c["score"] for c in r["curve"]]
                curves.append(np.interp(grid, s, v, left=v[0], right=v[-1]))
            C = np.array(curves)
            med = np.median(C, axis=0)
            lo = np.percentile(C, 25, axis=0)
            hi = np.percentile(C, 75, axis=0)
            ln, = ax.plot(grid, med, label=m, lw=2 if m == "neon" else 1.2)
            ax.fill_between(grid, lo, hi, alpha=0.15, color=ln.get_color())
        ax.axhline(ref[env]["solve"], ls="--", c="k", lw=0.8)
        ax.set_title(env, fontsize=10)
        ax.set_xlabel("training env steps")
        ax.set_ylabel("held-out return")
    for ax in axes[len(envs):]:
        ax.axis("off")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/learning_curves.png", dpi=130)
    plt.close(fig)

    # performance profile
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in methods:
        allv = np.concatenate([rows[(m, e)]["norm"] for e in envs
                               if (m, e) in rows]) if any(
            (m, e) in rows for e in envs) else np.array([])
        if not len(allv):
            continue
        taus, frac = performance_profile(allv)
        ax.plot(taus, frac, label=m, lw=2 if m == "neon" else 1.2)
    ax.set_xlabel("normalized score threshold  (0 = random, 1 = solved)")
    ax.set_ylabel("fraction of runs above threshold")
    ax.set_title("Performance profile (all tasks, all seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/performance_profile.png", dpi=130)
    plt.close(fig)
    return [f"{FIGS}/learning_curves.png", f"{FIGS}/performance_profile.png"]


def main():
    runs = load()
    if not runs:
        print("no results yet; run experiments/run.py first")
        return
    ref = load_reference_scores()
    envs = [e for e in TASK_SPEC if any(r["env"] == e for r in runs)]
    methods = [m for m in METHODS if any(r["method"] == m for r in runs)]
    rows = build(runs, ref)
    md = ["# Benchmark results",
          f"\n{len(runs)} runs | {len(methods)} methods | {len(envs)} tasks",
          "\nNormalized score: 0 = uniform-random policy, 1 = task solve "
          "threshold. IQM = interquartile mean with 95% stratified-bootstrap "
          "CI over seeds (Agarwal et al. 2021).",
          table_overall(rows, envs, methods),
          table_per_env(rows, envs, methods)]
    os.makedirs("results", exist_ok=True)
    with open("results/RESULTS.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    try:
        figs = figures(runs, rows, ref, envs, methods)
        print("\nfigures:", ", ".join(figs))
    except Exception as e:
        print("figure generation skipped:", e)


if __name__ == "__main__":
    main()
