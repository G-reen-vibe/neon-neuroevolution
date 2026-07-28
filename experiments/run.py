"""Resumable benchmark runner.

Every (method, env, seed) cell is written to its own JSON file and skipped if
already present, so the full sweep can be executed in bounded chunks and
resumed indefinitely -- which is what makes this feasible on a single core
with wall-clock limits.

Seeds are iterated outermost so that partial sweeps are still balanced across
methods and tasks (after chunk 1 every cell has seed 0, etc.) rather than
fully finishing one method before starting the next.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neon.harness import Task, run_method, TASK_SPEC
from neon.registry import METHODS

OUT = "results/runs"


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cell_path(method, env, seed):
    return os.path.join(OUT, f"{method}__{env}__s{seed}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--envs", default=",".join(TASK_SPEC))
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--max-minutes", type=float, default=25.0,
                    help="stop launching new cells after this much wall time")
    ap.add_argument("--run-timeout", type=float, default=300.0,
                    help="per-cell wall-clock cap (seconds)")
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument("--n-evals", type=int, default=20)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    methods = args.methods.split(",")
    envs = args.envs.split(",")
    seeds = parse_seeds(args.seeds)
    os.makedirs(OUT, exist_ok=True)

    cells = [(m, e, s) for s in seeds for e in envs for m in methods]
    todo = [c for c in cells if not os.path.exists(cell_path(*c))]

    if args.status:
        print(f"total cells {len(cells)}  done {len(cells)-len(todo)}  "
              f"remaining {len(todo)}")
        by_m = {}
        for m, e, s in todo:
            by_m[m] = by_m.get(m, 0) + 1
        for m in sorted(by_m):
            print(f"  {m:10s} {by_m[m]:4d} remaining")
        return

    print(f"{len(cells)} cells, {len(cells)-len(todo)} already done, "
          f"{len(todo)} to run (budget {args.max_minutes:.0f} min)")
    t_start = time.time()
    ran = 0
    for method, env, seed in todo:
        if (time.time() - t_start) / 60 > args.max_minutes:
            print(f"\ntime budget reached; {len(todo)-ran} cells remain. "
                  f"Re-run the same command to continue.")
            break
        task = Task(env)
        budget = int(task.budget * args.budget_scale)
        t0 = time.time()
        try:
            m = METHODS[method](task, seed)
            res = run_method(m, task, budget=budget, n_evals=args.n_evals,
                             max_seconds=args.run_timeout)
            res.update(method=method, env=env, seed=seed, budget=budget,
                       truncated=res["steps"] < budget)
            with open(cell_path(method, env, seed), "w") as f:
                json.dump(res, f)
            flag = " [TRUNC]" if res["truncated"] else ""
            sv = (f"solved@{res['solved']['steps']:,}" if res["solved"]
                  else "unsolved")
            print(f"  {method:10s} {env:26s} s{seed}  "
                  f"final {res['final_score']:9.2f}  {sv:22s} "
                  f"{time.time()-t0:6.1f}s{flag}")
        except Exception:
            print(f"  {method:10s} {env:26s} s{seed}  FAILED")
            traceback.print_exc()
        ran += 1
    print(f"\nchunk done: {ran} cells in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
