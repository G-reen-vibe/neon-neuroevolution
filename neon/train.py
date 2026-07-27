"""Training loop: NEON on vectorized classic-control tasks.

Entire population is evaluated as one batch: each env step is a single numpy
array op, each policy step one batched matmul. No per-genome interpretation.
"""

from __future__ import annotations

import time
import numpy as np

from .core import NEON, NEONConfig
from .envs import ENVS

EPISODES = {"CartPole-v1": 2, "Acrobot-v1": 2, "Pendulum-v1": 4}


def rollout(algo: NEON, weights: np.ndarray, env_cls, seed: int,
            episodes: int) -> tuple[np.ndarray, int]:
    """Evaluate P networks in lockstep with common random numbers: every
    individual faces the same initial states."""
    P = weights.shape[0]
    total = np.zeros(P)
    steps = 0
    rng = np.random.default_rng(seed)
    scale = env_cls.obs_scale
    for ep in range(episodes):
        env = env_cls(batch=P, rng=rng)
        obs = env.reset(seed=int(seed + 7919 * ep))
        done = np.zeros(P, dtype=bool)
        ret = np.zeros(P)
        while not done.all():
            out = algo.act_batch(weights, obs / scale)
            if env_cls.discrete:
                if algo.nO == 1:
                    actions = (out[:, 0] > 0).astype(int)
                else:
                    actions = np.argmax(out, axis=1)
            else:
                actions = out * getattr(env_cls, "act_scale", 1.0)
            obs, r, done = env.step(actions)
            ret += r
            steps += P  # lockstep batch: P sim-steps per tick
        total += ret
    return total / episodes, steps


def n_io(env_cls):
    if env_cls.discrete:
        out = 1 if env_cls.n_actions == 2 else env_cls.n_actions
    else:
        out = env_cls.n_actions
    return env_cls.obs_dim, out


def train(env_id: str, seed: int = 0, max_generations: int = 300,
          log=print, cfg_overrides: dict | None = None, quiet=False,
          fixed_eval: bool = False, episodes: int | None = None):
    env_cls = ENVS[env_id]
    n_in, n_out = n_io(env_cls)
    kw = dict(n_inputs=n_in, n_outputs=n_out, seed=seed)
    if cfg_overrides:
        kw.update(cfg_overrides)
    cfg = NEONConfig(**kw)
    algo = NEON(cfg)
    episodes = EPISODES[env_id] if episodes is None else episodes
    t0 = time.time()
    total_steps = 0
    history = []
    solved_at = None
    for gen in range(max_generations):
        masks, weights, eps, node_on = algo.sample_population()
        ep_seed = (seed * 977 if fixed_eval else seed * 977 + gen * 31)
        fitness, steps = rollout(algo, weights, env_cls, ep_seed, episodes)
        total_steps += steps  # env-steps consumed across the batch
        algo.update(masks, eps, fitness, node_on)
        map_w = np.repeat(algo.mean_network(), 8, axis=0)
        map_ret, s2 = rollout(algo, map_w, env_cls, 555_000 + seed + gen, 1)
        total_steps += s2
        st = algo.stats()
        history.append(dict(gen=gen, pop_mean=float(fitness.mean()),
                            map_mean=float(map_ret.mean()),
                            steps=total_steps, wall=time.time() - t0, **st))
        if not quiet and (gen % 10 == 0 or map_ret.mean() >= env_cls.solve):
            log(f"[{env_id} s{seed}] gen {gen:3d} pop {fitness.mean():8.1f} "
                f"map {map_ret.mean():8.1f} edges {st['active_edges']:3d} "
                f"hid {st['used_hidden']} pnode {st['p_node_mean']:.2f} "
                f"t {time.time()-t0:5.1f}s")
        if map_ret.mean() >= env_cls.solve and solved_at is None:
            map_w = np.repeat(algo.mean_network(), 24, axis=0)
            conf, s3 = rollout(algo, map_w, env_cls, 999_000 + seed + gen, 2)
            total_steps += s3
            if conf.mean() >= env_cls.solve:
                solved_at = dict(gen=gen, steps=total_steps,
                                 wall=time.time() - t0,
                                 confirm=float(conf.mean()), **st)
                if not quiet:
                    log(f"[{env_id} s{seed}] SOLVED gen {gen} "
                        f"confirm {conf.mean():.1f} wall {time.time()-t0:.1f}s "
                        f"edges {st['active_edges']} hid {st['used_hidden']}")
                break
    return dict(env=env_id, seed=seed, solved=solved_at, history=history,
                final_stats=algo.stats())
