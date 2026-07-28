"""Shared evaluation harness.

Every method — NEON, the evolution baselines, NEAT and PPO — is evaluated
through this one code path, so comparisons are apples-to-apples:

  * identical environment dynamics (neon.envs, verified against gymnasium)
  * identical observation normalization
  * identical action decoding (argmax over n_actions / tanh-scaled continuous)
  * identical held-out evaluation protocol (same fixed eval seeds for all)
  * identical budget currency: *training* environment steps

Evaluation steps are recorded but never charged against the training budget,
which is the standard convention in the RL literature.
"""

from __future__ import annotations

import time
import numpy as np

from .envs import ENVS

# Held-out evaluation initial states. Identical for every method, every run.
EVAL_SEED = 20_240_115
N_EVAL = 32

# Per-task protocol constants.
TASK_SPEC = {
    "CartPole-v1":              dict(budget=500_000,   train_eps=2),
    "Acrobot-v1":               dict(budget=1_000_000, train_eps=2),
    "MountainCar-v0":           dict(budget=1_000_000, train_eps=2),
    "Pendulum-v1":              dict(budget=1_000_000, train_eps=3),
    "MountainCarContinuous-v0": dict(budget=2_000_000, train_eps=1),
}


class Task:
    """Wraps a vectorized env with the shared rollout / evaluation protocol."""

    def __init__(self, env_id: str):
        self.env_id = env_id
        self.cls = ENVS[env_id]
        self.n_in = self.cls.obs_dim
        # Discrete tasks use one output per action (argmax) for EVERY method,
        # so no method gets an encoding advantage.
        self.n_out = self.cls.n_actions
        self.discrete = self.cls.discrete
        self.solve = self.cls.solve
        self.horizon = self.cls.horizon
        self.obs_scale = self.cls.obs_scale
        self.act_scale = self.cls.act_scale
        spec = TASK_SPEC[env_id]
        self.budget = spec["budget"]
        self.train_eps = spec["train_eps"]

    def decode(self, raw):
        """raw (B, n_out) in [-1,1] -> environment actions."""
        if self.discrete:
            return np.argmax(raw, axis=1)
        return raw * self.act_scale

    def rollout(self, act_fn, B, seed, episodes, common=True):
        """Roll B policies in lockstep. Returns (returns (B,), env_steps)."""
        total = np.zeros(B)
        steps = 0
        for ep in range(episodes):
            env = self.cls(batch=B, rng=np.random.default_rng(seed + ep))
            obs = env.reset(seed=seed + 7919 * ep, common=common)
            done = np.zeros(B, dtype=bool)
            ret = np.zeros(B)
            while not done.all():
                steps += int(np.count_nonzero(~done))
                a = self.decode(act_fn(obs / self.obs_scale))
                obs, r, done = env.step(a)
                ret += r
            total += ret
        return total / episodes, steps

    def fitness(self, act_fn, B, seed):
        """Training fitness: common random numbers across the population."""
        return self.rollout(act_fn, B, seed, self.train_eps, common=True)

    def evaluate(self, act_fn):
        """Held-out evaluation of ONE policy tiled over N_EVAL independent
        initial states. act_fn must accept (N_EVAL, n_in)."""
        rets, steps = self.rollout(act_fn, N_EVAL, EVAL_SEED, 1, common=False)
        return float(rets.mean()), steps


# --------------------------------------------------------------------- MLP
class MLP:
    """Fixed-topology tanh MLP used by all parameter-vector baselines."""

    def __init__(self, n_in, n_out, hidden=16):
        self.n_in, self.n_out, self.h = n_in, n_out, hidden
        self.dim = n_in * hidden + hidden + hidden * n_out + n_out
        self._cuts = np.cumsum([n_in * hidden, hidden, hidden * n_out, n_out])

    def act_fn(self, params):
        """params (P, dim) -> callable(obs (P,n_in)) -> raw (P,n_out)."""
        P = params.shape[0]
        a, b, c, _ = self._cuts
        W1 = params[:, :a].reshape(P, self.n_in, self.h)
        b1 = params[:, a:b].reshape(P, self.h)
        W2 = params[:, b:c].reshape(P, self.h, self.n_out)
        b2 = params[:, c:].reshape(P, self.n_out)

        def act(obs):
            hid = np.tanh(np.einsum("pi,pih->ph", obs, W1) + b1)
            return np.tanh(np.einsum("ph,pho->po", hid, W2) + b2)
        return act

    def tiled_act_fn(self, theta, B):
        return self.act_fn(np.repeat(theta[None, :], B, axis=0))


# ------------------------------------------------------------------ driver
def run_method(method, task: Task, budget=None, n_evals=40,
               max_seconds=None, log=None):
    """Drive any method to a fixed training-step budget, recording a learning
    curve of held-out performance. Methods implement:

        method.step()          -> training env-steps consumed by one iteration
        method.act_fn(B)       -> callable for B copies of the incumbent
        method.size()          -> dict describing the incumbent's complexity
    """
    budget = budget or task.budget
    marks = np.linspace(budget / n_evals, budget, n_evals)
    t0 = time.time()
    steps = 0
    nxt = 0
    curve = []
    solved = None
    while steps < budget:
        steps += method.step()
        if max_seconds and time.time() - t0 > max_seconds:
            break
        while nxt < len(marks) and steps >= marks[nxt]:
            score, _ = task.evaluate(method.act_fn(N_EVAL))
            rec = dict(steps=int(steps), score=score,
                       wall=round(time.time() - t0, 3), **method.size())
            curve.append(rec)
            if solved is None and score >= task.solve:
                solved = dict(steps=int(steps), wall=rec["wall"], score=score)
            if log:
                log(f"    {steps:>9,} steps  score {score:9.2f}  "
                    f"{method.size()}  {rec['wall']:.1f}s")
            nxt += 1
    if not curve:  # budget smaller than one iteration
        score, _ = task.evaluate(method.act_fn(N_EVAL))
        curve.append(dict(steps=int(steps), score=score,
                          wall=round(time.time() - t0, 3), **method.size()))
    return dict(curve=curve, solved=solved,
                final_score=curve[-1]["score"],
                wall=round(time.time() - t0, 3), steps=int(steps))
