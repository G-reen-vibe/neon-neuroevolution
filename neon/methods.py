"""Baseline and proposed methods, all behind one interface.

Each method implements:
    step()      -> training env-steps consumed by one iteration
    act_fn(B)   -> callable for B copies of the current incumbent policy
    size()      -> dict(params=..., hidden=...) describing incumbent complexity

Baselines use literature-standard hyperparameters (cited per class). All
population methods share popsize=64 so that generation counts are comparable;
the budget is denominated in environment steps regardless.
"""

from __future__ import annotations

import numpy as np

from .core import NEON, NEONConfig
from .harness import MLP

POPSIZE = 64
MLP_HIDDEN = 16


def _rank_utilities(f):
    """Zero-sum rank utilities (OpenAI-ES fitness shaping)."""
    n = len(f)
    r = np.empty(n)
    r[np.argsort(f)] = np.arange(n)
    return r / (n - 1) - 0.5


def _nes_utilities(n):
    """Standard NES utility weights (Wierstra et al. 2014)."""
    k = np.arange(1, n + 1)
    u = np.maximum(0.0, np.log(n / 2 + 1) - np.log(k))
    u = u / u.sum() - 1.0 / n
    return u


class Adam:
    def __init__(self, dim, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0

    def __call__(self, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


# ============================================================== NEON (ours)
class NEONMethod:
    name = "neon"

    def __init__(self, task, seed, hp=None):
        self.task, self.seed = task, seed
        cfg = dict(n_inputs=task.n_in, n_outputs=task.n_out,
                   popsize=POPSIZE, seed=seed)
        cfg.update(hp or {})
        self.algo = NEON(NEONConfig(**cfg))
        self.gen = 0

    def step(self):
        masks, weights, eps, node_on = self.algo.sample_population()
        fit, steps = self.task.fitness(
            lambda o: self.algo.act_batch(weights, o), POPSIZE,
            self.seed * 977 + self.gen * 31)
        self.algo.update(masks, eps, fit, node_on)
        self.gen += 1
        return steps

    def act_fn(self, B):
        w = np.repeat(self.algo.mean_network(), B, axis=0)
        return lambda o: self.algo.act_batch(w, o)

    def size(self):
        st = self.algo.stats()
        return dict(params=st["active_edges"], hidden=st["used_hidden"])


# ================================================= parameter-vector methods
class _VectorMethod:
    """Common scaffolding for methods that search a fixed-length vector."""

    def __init__(self, task, seed, hp=None):
        self.task, self.seed = task, seed
        self.net = MLP(task.n_in, task.n_out, MLP_HIDDEN)
        self.d = self.net.dim
        self.rng = np.random.default_rng(seed)
        self.gen = 0
        self.hp = hp or {}

    def _eval(self, pop):
        fit, steps = self.task.fitness(self.net.act_fn(pop), pop.shape[0],
                                       self.seed * 977 + self.gen * 31)
        self.gen += 1
        return fit, steps

    def act_fn(self, B):
        return self.net.tiled_act_fn(self.incumbent, B)

    def size(self):
        return dict(params=self.d, hidden=MLP_HIDDEN)


class OpenAIES(_VectorMethod):
    """Salimans et al. 2017, 'Evolution Strategies as a Scalable Alternative
    to Reinforcement Learning'. Mirrored sampling, rank shaping, Adam."""
    name = "openai_es"

    def __init__(self, task, seed, hp=None):
        super().__init__(task, seed, hp)
        self.sigma = self.hp.get("sigma", 0.1)
        self.theta = self.rng.normal(0, 0.1, self.d)
        self.opt = Adam(self.d, self.hp.get("lr", 0.03))
        self.wd = self.hp.get("weight_decay", 0.005)

    def step(self):
        half = POPSIZE // 2
        e = self.rng.normal(0, 1, (half, self.d))
        eps = np.concatenate([e, -e], axis=0)
        fit, steps = self._eval(self.theta + self.sigma * eps)
        u = _rank_utilities(fit)
        grad = (u @ eps) / (POPSIZE * self.sigma) - self.wd * self.theta
        self.theta += self.opt(grad)
        return steps

    @property
    def incumbent(self):
        return self.theta


class SNES(_VectorMethod):
    """Schaul et al. 2011, Separable Natural Evolution Strategies: adapts a
    per-parameter step size by natural gradient on the search distribution."""
    name = "snes"

    def __init__(self, task, seed, hp=None):
        super().__init__(task, seed, hp)
        self.mu = self.rng.normal(0, 0.1, self.d)
        self.sig = np.full(self.d, self.hp.get("sigma0", 0.2))
        self.eta_mu = 1.0
        self.eta_sig = (3 + np.log(self.d)) / (5 * np.sqrt(self.d))
        self.u = _nes_utilities(POPSIZE)

    def step(self):
        s = self.rng.normal(0, 1, (POPSIZE, self.d))
        fit, steps = self._eval(self.mu + self.sig * s)
        order = np.argsort(-fit)
        us = np.zeros(POPSIZE)
        us[order] = self.u
        g_mu = us @ s
        g_sig = us @ (s**2 - 1.0)
        self.mu += self.eta_mu * self.sig * g_mu
        self.sig *= np.exp(0.5 * self.eta_sig * g_sig)
        np.clip(self.sig, 1e-4, 3.0, out=self.sig)
        return steps

    @property
    def incumbent(self):
        return self.mu


class CMAES(_VectorMethod):
    """Hansen's CMA-ES (reference `cma` package), the standard strong
    black-box optimizer baseline."""
    name = "cmaes"

    def __init__(self, task, seed, hp=None):
        super().__init__(task, seed, hp)
        import cma
        self.es = cma.CMAEvolutionStrategy(
            np.zeros(self.d), self.hp.get("sigma0", 0.5),
            {"popsize": POPSIZE, "seed": int(seed) + 1, "verbose": -9})

    def step(self):
        sols = self.es.ask()
        pop = np.asarray(sols)
        fit, steps = self._eval(pop)
        self.es.tell(sols, list(-fit))  # cma minimizes
        return steps

    @property
    def incumbent(self):
        return np.asarray(self.es.result.xfavorite)


class DeepGA(_VectorMethod):
    """Such et al. 2017, 'Deep Neuroevolution': truncation selection plus
    Gaussian mutation, no crossover, with elitism."""
    name = "ga"

    def __init__(self, task, seed, hp=None):
        super().__init__(task, seed, hp)
        self.sigma = self.hp.get("sigma", 0.02)
        self.trunc = max(2, POPSIZE // 10)
        self.pop = self.rng.normal(0, 0.5, (POPSIZE, self.d))
        self.best = self.pop[0].copy()
        self.best_fit = -np.inf

    def step(self):
        fit, steps = self._eval(self.pop)
        order = np.argsort(-fit)
        if fit[order[0]] > self.best_fit:
            self.best_fit = float(fit[order[0]])
            self.best = self.pop[order[0]].copy()
        parents = self.pop[order[: self.trunc]]
        idx = self.rng.integers(0, self.trunc, POPSIZE - 1)
        children = parents[idx] + self.sigma * self.rng.normal(
            0, 1, (POPSIZE - 1, self.d))
        self.pop = np.concatenate([self.best[None, :], children], axis=0)
        return steps

    @property
    def incumbent(self):
        return self.best


class RandomSearch(_VectorMethod):
    """Uniform-random parameter search: the sanity floor every method must
    clear to demonstrate it is learning anything at all."""
    name = "random"

    def __init__(self, task, seed, hp=None):
        super().__init__(task, seed, hp)
        self.scale = self.hp.get("scale", 1.0)
        self.best = self.rng.normal(0, self.scale, self.d)
        self.best_fit = -np.inf

    def step(self):
        pop = self.rng.normal(0, self.scale, (POPSIZE, self.d))
        fit, steps = self._eval(pop)
        i = int(np.argmax(fit))
        if fit[i] > self.best_fit:
            self.best_fit = float(fit[i])
            self.best = pop[i].copy()
        return steps

    @property
    def incumbent(self):
        return self.best
