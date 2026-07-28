"""Statistical aggregation following Agarwal et al. 2021, "Deep RL at the
Edge of the Statistical Precipice": report the interquartile mean (IQM) with
stratified-bootstrap confidence intervals rather than bare means, because
per-seed variance in RL is large and means are dominated by outliers.
"""

from __future__ import annotations

import json
import os
import numpy as np


def iqm(x):
    """Interquartile mean: mean of the middle 50% of runs."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 4:
        return float(x.mean())
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(x[lo:hi].mean())


def stratified_bootstrap_ci(by_task, agg=iqm, n_boot=10_000, alpha=0.05,
                            seed=0):
    """CI for an aggregate over runs pooled across tasks, resampling seeds
    *within* each task (stratified) so no task dominates the resample."""
    rng = np.random.default_rng(seed)
    keys = list(by_task)
    arrs = [np.asarray(by_task[k], dtype=float) for k in keys]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pooled = [a[rng.integers(0, len(a), len(a))] for a in arrs if len(a)]
        stats[b] = agg(np.concatenate(pooled))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = agg(np.concatenate([a for a in arrs if len(a)]))
    return float(point), float(lo), float(hi)


def bootstrap_ci(x, agg=np.mean, n_boot=10_000, alpha=0.05, seed=0):
    """Simple percentile bootstrap CI for a single sample."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.array([agg(x[rng.integers(0, len(x), len(x))])
                      for _ in range(n_boot)])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(agg(x)), float(lo), float(hi)


def normalize(score, random_score, solve_score):
    """Map raw return to a common scale: 0 = random policy, 1 = solved."""
    denom = solve_score - random_score
    if abs(denom) < 1e-9:
        return float("nan")
    return (score - random_score) / denom


def performance_profile(norm_scores, taus=None):
    """Fraction of runs whose normalized score exceeds tau, for a range of
    tau (Agarwal et al. 2021). Robust to outliers and reveals the whole
    distribution rather than a single summary number."""
    x = np.asarray(norm_scores, dtype=float)
    taus = np.linspace(0.0, 1.2, 61) if taus is None else np.asarray(taus)
    return taus, np.array([(x >= t).mean() for t in taus])


# ------------------------------------------------------- reference scores
REF_PATH = "results/reference_scores.json"


def compute_reference_scores(path=REF_PATH):
    """Uniform-random-action return per task, used as the 0 point of the
    normalized scale. Evaluated on the same held-out seeds as every method."""
    from .harness import Task, N_EVAL
    from .envs import ENVS
    out = {}
    for env_id in ENVS:
        task = Task(env_id)
        rng = np.random.default_rng(0)

        def act(obs, task=task, rng=rng):
            return rng.uniform(-1, 1, (obs.shape[0], task.n_out))
        scores = [task.evaluate(act)[0] for _ in range(10)]
        out[env_id] = dict(random=float(np.mean(scores)),
                           solve=float(task.solve))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def load_reference_scores(path=REF_PATH):
    if not os.path.exists(path):
        return compute_reference_scores(path)
    with open(path) as f:
        return json.load(f)
