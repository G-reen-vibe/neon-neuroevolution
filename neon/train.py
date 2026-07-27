"""Training loop: NEON on gymnasium classic-control tasks.

The whole population steps its environments in lockstep so the policy forward
pass stays one batched matmul per generation-step (SyncVectorEnv would add
per-env python overhead for autoreset we don't want; we manage a simple
batch of envs directly).
"""

from __future__ import annotations

import time
import numpy as np
import gymnasium as gym

from .core import NEON, NEONConfig

TASKS = {
    "CartPole-v1": dict(discrete=True, solve=475.0, max_steps=500,
                        obs_scale=[2.4, 3.0, 0.21, 3.0]),
    "Acrobot-v1": dict(discrete=True, solve=-90.0, max_steps=500,
                       obs_scale=[1, 1, 1, 1, 12.57, 28.27]),
    "Pendulum-v1": dict(discrete=False, solve=-180.0, max_steps=200,
                        act_scale=2.0, obs_scale=[1, 1, 8.0], episodes=4),
}


def make_batch(env_id, n, seed):
    """Common random numbers: every individual faces the SAME initial state,
    so fitness ranks compare policies, not luck."""
    envs = [gym.make(env_id) for _ in range(n)]
    obs = np.stack([e.reset(seed=int(seed))[0] for e in envs])
    return envs, obs


def rollout(algo: NEON, weights: np.ndarray, env_id: str, seed: int,
            episodes: int = 1) -> tuple[np.ndarray, int]:
    """Evaluate P networks, each on its own env instance. Returns mean
    returns over `episodes` and total env steps consumed."""
    spec = TASKS[env_id]
    scale = np.asarray(spec["obs_scale"], dtype=np.float64)
    P = weights.shape[0]
    total = np.zeros(P)
    steps = 0
    for ep in range(episodes):
        envs, obs = make_batch(env_id, P, seed + 10_000 * ep)
        done = np.zeros(P, dtype=bool)
        ret = np.zeros(P)
        while not done.all():
            out = algo.act_batch(weights, obs / scale)  # (P, nO) in [-1, 1]
            if spec["discrete"]:
                n_act = envs[0].action_space.n
                if algo.nO == 1:
                    actions = (out[:, 0] > 0).astype(int)
                else:
                    actions = np.argmax(out[:, :n_act], axis=1)
            else:
                actions = out * spec.get("act_scale", 1.0)
            for i, e in enumerate(envs):
                if done[i]:
                    continue
                a = actions[i] if not spec["discrete"] else int(actions[i])
                o, r, term, trunc, _ = e.step(a)
                obs[i] = o
                ret[i] += r
                done[i] |= term or trunc
                steps += 1
        for e in envs:
            e.close()
        total += ret
    return total / episodes, steps


def n_outputs_for(env_id):
    spec = TASKS[env_id]
    e = gym.make(env_id)
    if spec["discrete"]:
        n = e.action_space.n
        out = 1 if n == 2 else n
    else:
        out = e.action_space.shape[0]
    n_in = e.observation_space.shape[0]
    e.close()
    return n_in, out


def train(env_id: str, seed: int = 0, max_generations: int = 120,
          eval_episodes: int | None = None, log=print, cfg_overrides: dict | None = None):
    n_in, n_out = n_outputs_for(env_id)
    spec = TASKS[env_id]
    kw = dict(n_inputs=n_in, n_outputs=n_out, seed=seed)
    if cfg_overrides:
        kw.update(cfg_overrides)
    cfg = NEONConfig(**kw)
    algo = NEON(cfg)
    if eval_episodes is None:
        eval_episodes = spec.get("episodes", 1)
    t0 = time.time()
    total_steps = 0
    history = []
    solved_at = None
    for gen in range(max_generations):
        masks, weights, eps, node_on = algo.sample_population()
        fitness, steps = rollout(algo, weights, env_id, seed * 977 + gen * 31,
                                 episodes=eval_episodes)
        total_steps += steps
        algo.update(masks, eps, fitness, node_on)
        # deterministic MAP-network check on fresh seeds
        map_w = np.repeat(algo.mean_network(), 8, axis=0)
        map_ret, s2 = rollout(algo, map_w, env_id, 555_000 + seed + gen)
        total_steps += s2
        st = algo.stats()
        history.append(dict(gen=gen, pop_mean=float(fitness.mean()),
                            pop_best=float(fitness.max()),
                            map_mean=float(map_ret.mean()),
                            steps=total_steps, wall=time.time() - t0, **st))
        if gen % 5 == 0 or map_ret.mean() >= spec["solve"]:
            log(f"[{env_id} s{seed}] gen {gen:3d} pop {fitness.mean():8.1f} "
                f"map {map_ret.mean():8.1f} edges {st['active_edges']:3d} "
                f"hid {st['used_hidden']} steps {total_steps} "
                f"t {time.time()-t0:5.1f}s")
        if map_ret.mean() >= spec["solve"] and solved_at is None:
            # confirm with a stricter evaluation
            map_w = np.repeat(algo.mean_network(), 20, axis=0)
            conf, s3 = rollout(algo, map_w, env_id, 999_000 + seed + gen)
            total_steps += s3
            if conf.mean() >= spec["solve"]:
                solved_at = dict(gen=gen, steps=total_steps,
                                 wall=time.time() - t0,
                                 confirm=float(conf.mean()))
                log(f"[{env_id} s{seed}] SOLVED gen {gen} "
                    f"confirm {conf.mean():.1f} steps {total_steps} "
                    f"wall {time.time()-t0:.1f}s")
                break
    return dict(env=env_id, seed=seed, solved=solved_at, history=history,
                final_stats=algo.stats())
