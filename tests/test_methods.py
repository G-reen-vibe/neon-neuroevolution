"""Smoke test: every registered method runs, learns, and reports size."""
import numpy as np
from neon.harness import Task, run_method
from neon.registry import METHODS


def test_all(env_id="CartPole-v1", budget=40_000):
    task = Task(env_id)
    print(f"smoke test on {env_id}, budget {budget:,} steps")
    for name, M in METHODS.items():
        m = M(task, 0)
        r = run_method(m, task, budget=budget, n_evals=3)
        assert len(r["curve"]) >= 1, f"{name}: empty curve"
        assert np.isfinite(r["final_score"]), f"{name}: non-finite score"
        assert r["curve"][-1]["params"] > 0, f"{name}: no params reported"
        print(f"  {name:10s} final {r['final_score']:8.1f}  "
              f"size {r['curve'][-1]['params']:5d}  {r['wall']:5.1f}s")
    print("all methods OK")


if __name__ == "__main__":
    test_all()
