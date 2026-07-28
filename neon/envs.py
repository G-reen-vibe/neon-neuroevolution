"""Natively vectorized classic-control environments.

Ports of gymnasium's classic-control dynamics that step a whole batch of B
independent instances as numpy array ops, removing the per-env python
`step()` overhead that dominates wall-clock for population methods on CPU.
Verified numerically against gymnasium in tests/test_envs.py.

Two reset modes:
  common=True   all rows get the SAME initial state (common random numbers,
                so fitness ranks compare policies rather than luck)
  common=False  independent initial states per row (used by PPO, and by the
                held-out evaluation protocol)
"""

from __future__ import annotations

import numpy as np


class VecEnv:
    """Base class. Subclasses define state layout, init sampler, obs and step."""

    discrete: bool = True
    act_scale: float = 1.0

    def __init__(self, batch: int, rng=None):
        self.B = batch
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.s = np.zeros((batch, self.state_dim))
        self.done = np.zeros(batch, dtype=bool)
        self.last_term = np.zeros(batch, dtype=bool)
        self.t = np.zeros(batch, dtype=np.int64)

    # -- to implement -------------------------------------------------------
    def _sample_init(self, n, rng):        # -> (n, state_dim)
        raise NotImplementedError

    def _obs(self):                        # -> (B, obs_dim)
        raise NotImplementedError

    def _step(self, action, live):         # -> (reward (B,), terminated (B,))
        raise NotImplementedError

    # -- shared -------------------------------------------------------------
    def reset(self, seed=None, common=True):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        if common:
            self.s = np.repeat(self._sample_init(1, rng), self.B, axis=0)
        else:
            self.s = self._sample_init(self.B, rng)
        self.done[:] = False
        self.t[:] = 0
        return self._obs()

    def reset_rows(self, idx, rng=None):
        """Reset a subset of rows in place (auto-reset for on-policy RL)."""
        rng = rng if rng is not None else self.rng
        n = int(np.count_nonzero(idx))
        if n == 0:
            return
        self.s[idx] = self._sample_init(n, rng)
        self.done[idx] = False
        self.t[idx] = 0

    def step(self, action):
        live = ~self.done
        reward, term = self._step(action, live)
        reward = np.where(live, reward, 0.0)
        self.t += live
        # true termination (not horizon truncation) -- PPO needs this to
        # decide whether to bootstrap the value of the next state
        self.last_term = term & live
        self.done |= self.last_term
        self.done |= self.t >= self.horizon
        return self._obs(), reward, self.done.copy()


class VecCartPole(VecEnv):
    name = "CartPole-v1"
    state_dim, obs_dim, n_actions = 4, 4, 2
    discrete, horizon, solve = True, 500, 475.0
    obs_scale = np.array([2.4, 3.0, 0.2095, 3.0])

    def _sample_init(self, n, rng):
        return rng.uniform(-0.05, 0.05, (n, 4))

    def _obs(self):
        return self.s.copy()

    def _step(self, action, live):
        g, mc, mp, l, fmag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
        mt, pml = mc + mp, mp * l
        x, xd, th, thd = self.s.T
        force = np.where(action == 1, fmag, -fmag)
        ct, st = np.cos(th), np.sin(th)
        temp = (force + pml * thd**2 * st) / mt
        thacc = (g * st - ct * temp) / (l * (4.0 / 3.0 - mp * ct**2 / mt))
        xacc = temp - pml * thacc * ct / mt
        ns = np.stack([x + tau * xd, xd + tau * xacc,
                       th + tau * thd, thd + tau * thacc], axis=1)
        self.s[live] = ns[live]
        term = (np.abs(self.s[:, 0]) > 2.4) | (np.abs(self.s[:, 2]) > 0.2095)
        return np.ones(self.B), term


class VecPendulum(VecEnv):
    name = "Pendulum-v1"
    state_dim, obs_dim, n_actions = 2, 3, 1
    discrete, horizon, solve = False, 200, -180.0
    act_scale = 2.0
    obs_scale = np.array([1.0, 1.0, 8.0])

    def _sample_init(self, n, rng):
        return np.stack([rng.uniform(-np.pi, np.pi, n),
                         rng.uniform(-1.0, 1.0, n)], axis=1)

    def _obs(self):
        th, thd = self.s.T
        return np.stack([np.cos(th), np.sin(th), thd], axis=1)

    def _step(self, action, live):
        g, m, l, dt = 10.0, 1.0, 1.0, 0.05
        u = np.clip(action[:, 0], -2.0, 2.0)
        th, thd = self.s.T
        th_n = ((th + np.pi) % (2 * np.pi)) - np.pi
        cost = th_n**2 + 0.1 * thd**2 + 0.001 * u**2
        newthd = np.clip(thd + (3 * g / (2 * l) * np.sin(th)
                                + 3.0 / (m * l**2) * u) * dt, -8.0, 8.0)
        ns = np.stack([th + newthd * dt, newthd], axis=1)
        self.s[live] = ns[live]
        return -cost, np.zeros(self.B, dtype=bool)


class VecAcrobot(VecEnv):
    name = "Acrobot-v1"
    state_dim, obs_dim, n_actions = 4, 6, 3
    discrete, horizon, solve = True, 500, -90.0
    obs_scale = np.array([1, 1, 1, 1, 4 * np.pi, 9 * np.pi])

    def _sample_init(self, n, rng):
        return rng.uniform(-0.1, 0.1, (n, 4))

    def _obs(self):
        th1, th2, d1, d2 = self.s.T
        return np.stack([np.cos(th1), np.sin(th1),
                         np.cos(th2), np.sin(th2), d1, d2], axis=1)

    def _dsdt(self, sa):
        m1 = m2 = l1 = 1.0
        lc1 = lc2 = 0.5
        I1 = I2 = 1.0
        g = 9.8
        th1, th2, dth1, dth2, a = sa.T
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2
                                 + 2 * l1 * lc2 * np.cos(th2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(th2)) + I2
        phi2 = m2 * lc2 * g * np.cos(th1 + th2 - np.pi / 2.0)
        phi1 = (-m2 * l1 * lc2 * dth2**2 * np.sin(th2)
                - 2 * m2 * l1 * lc2 * dth2 * dth1 * np.sin(th2)
                + (m1 * lc1 + m2 * l1) * g * np.cos(th1 - np.pi / 2) + phi2)
        ddth2 = ((a + d2 / d1 * phi1
                  - m2 * l1 * lc2 * dth1**2 * np.sin(th2) - phi2)
                 / (m2 * lc2**2 + I2 - d2**2 / d1))
        ddth1 = -(d2 * ddth2 + phi1) / d1
        out = np.zeros_like(sa)
        out[:, 0], out[:, 1], out[:, 2], out[:, 3] = dth1, dth2, ddth1, ddth2
        return out

    def _step(self, action, live):
        torque = action.astype(np.float64) - 1.0
        sa = np.concatenate([self.s, torque[:, None]], axis=1)
        dt = 0.2
        k1 = self._dsdt(sa)
        k2 = self._dsdt(sa + dt / 2 * k1)
        k3 = self._dsdt(sa + dt / 2 * k2)
        k4 = self._dsdt(sa + dt * k3)
        ns = sa + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        w = lambda x: ((x + np.pi) % (2 * np.pi)) - np.pi
        ns = np.stack([w(ns[:, 0]), w(ns[:, 1]),
                       np.clip(ns[:, 2], -4 * np.pi, 4 * np.pi),
                       np.clip(ns[:, 3], -9 * np.pi, 9 * np.pi)], axis=1)
        self.s[live] = ns[live]
        term = (-np.cos(self.s[:, 0])
                - np.cos(self.s[:, 1] + self.s[:, 0])) > 1.0
        reward = np.where(term, 0.0, -1.0)
        return reward, term


class VecMountainCar(VecEnv):
    name = "MountainCar-v0"
    state_dim, obs_dim, n_actions = 2, 2, 3
    discrete, horizon, solve = True, 200, -110.0
    obs_scale = np.array([0.9, 0.07])

    def _sample_init(self, n, rng):
        return np.stack([rng.uniform(-0.6, -0.4, n), np.zeros(n)], axis=1)

    def _obs(self):
        return self.s.copy()

    def _step(self, action, live):
        pos, vel = self.s.T
        v = vel + (action - 1) * 0.001 + np.cos(3 * pos) * (-0.0025)
        v = np.clip(v, -0.07, 0.07)
        p = np.clip(pos + v, -1.2, 0.6)
        v = np.where((p <= -1.2) & (v < 0), 0.0, v)
        ns = np.stack([p, v], axis=1)
        self.s[live] = ns[live]
        term = (self.s[:, 0] >= 0.5) & (self.s[:, 1] >= 0.0)
        return -np.ones(self.B), term


class VecMountainCarContinuous(VecEnv):
    name = "MountainCarContinuous-v0"
    state_dim, obs_dim, n_actions = 2, 2, 1
    # horizon shortened from gymnasium's 999 to 500: the dynamics are
    # unchanged and verified, but a 999-step horizon makes each fitness
    # evaluation so expensive that population methods get too few
    # generations within any budget PPO can also be run at.
    discrete, horizon, solve = False, 500, 90.0
    act_scale = 1.0
    obs_scale = np.array([0.9, 0.07])

    def _sample_init(self, n, rng):
        return np.stack([rng.uniform(-0.6, -0.4, n), np.zeros(n)], axis=1)

    def _obs(self):
        return self.s.copy()

    def _step(self, action, live):
        force = np.clip(action[:, 0], -1.0, 1.0)
        pos, vel = self.s.T
        v = vel + force * 0.0015 - 0.0025 * np.cos(3 * pos)
        v = np.clip(v, -0.07, 0.07)
        p = np.clip(pos + v, -1.2, 0.6)
        v = np.where((p <= -1.2) & (v < 0), 0.0, v)
        ns = np.stack([p, v], axis=1)
        self.s[live] = ns[live]
        term = (self.s[:, 0] >= 0.45) & (self.s[:, 1] >= 0.0)
        reward = np.where(term, 100.0, 0.0) - 0.1 * force**2
        return reward, term


ENVS = {c.name: c for c in [VecCartPole, VecPendulum, VecAcrobot,
                            VecMountainCar, VecMountainCarContinuous]}
