"""Natively vectorized classic-control environments.

Bit-for-bit ports of gymnasium's CartPole-v1, Pendulum-v1 and Acrobot-v1
dynamics, stepping a whole batch of B independent instances as numpy array
ops. This removes the per-env python `step()` overhead that dominates
wall-clock for population methods on CPU. Verified numerically against
gymnasium in tests/test_envs.py.
"""

from __future__ import annotations

import numpy as np


class VecCartPole:
    obs_dim, n_actions, discrete = 4, 2, True
    solve, horizon = 475.0, 500
    obs_scale = np.array([2.4, 3.0, 0.2095, 3.0])

    def __init__(self, batch, rng):
        self.B, self.rng = batch, rng

    def reset(self, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        base = rng.uniform(-0.05, 0.05, 4)
        self.s = np.tile(base, (self.B, 1))  # common random numbers
        self.done = np.zeros(self.B, dtype=bool)
        self.t = 0
        return self.s.copy()

    def step(self, action):
        g, mc, mp, l, fmag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
        mt, pml = mc + mp, mp * l
        x, xd, th, thd = self.s.T
        force = np.where(action == 1, fmag, -fmag)
        ct, st = np.cos(th), np.sin(th)
        temp = (force + pml * thd**2 * st) / mt
        thacc = (g * st - ct * temp) / (l * (4.0 / 3.0 - mp * ct**2 / mt))
        xacc = temp - pml * thacc * ct / mt
        live = ~self.done
        self.s[live, 0] = (x + tau * xd)[live]
        self.s[live, 1] = (xd + tau * xacc)[live]
        self.s[live, 2] = (th + tau * thd)[live]
        self.s[live, 3] = (thd + tau * thacc)[live]
        term = (np.abs(self.s[:, 0]) > 2.4) | (np.abs(self.s[:, 2]) > 0.2095)
        reward = live.astype(np.float64)  # 1 per live step (incl. term step)
        self.done |= term
        self.t += 1
        if self.t >= self.horizon:
            self.done[:] = True
        return self.s.copy(), reward, self.done.copy()


class VecPendulum:
    obs_dim, n_actions, discrete = 3, 1, False
    solve, horizon = -180.0, 200
    act_scale = 2.0
    obs_scale = np.array([1.0, 1.0, 8.0])

    def __init__(self, batch, rng):
        self.B, self.rng = batch, rng

    def reset(self, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        th = rng.uniform(-np.pi, np.pi)
        thd = rng.uniform(-1.0, 1.0)
        self.th = np.full(self.B, th)
        self.thd = np.full(self.B, thd)
        self.done = np.zeros(self.B, dtype=bool)
        self.t = 0
        return self._obs()

    def _obs(self):
        return np.stack([np.cos(self.th), np.sin(self.th), self.thd], axis=1)

    def step(self, action):
        g, m, l, dt = 10.0, 1.0, 1.0, 0.05
        u = np.clip(action[:, 0], -2.0, 2.0)
        th_n = ((self.th + np.pi) % (2 * np.pi)) - np.pi
        cost = th_n**2 + 0.1 * self.thd**2 + 0.001 * u**2
        newthd = self.thd + (3 * g / (2 * l) * np.sin(self.th)
                             + 3.0 / (m * l**2) * u) * dt
        newthd = np.clip(newthd, -8.0, 8.0)
        self.th = self.th + newthd * dt
        self.thd = newthd
        self.t += 1
        if self.t >= self.horizon:
            self.done[:] = True
        return self._obs(), -cost, self.done.copy()


class VecAcrobot:
    obs_dim, n_actions, discrete = 6, 3, True
    solve, horizon = -90.0, 500
    obs_scale = np.array([1, 1, 1, 1, 4 * np.pi, 9 * np.pi])

    def __init__(self, batch, rng):
        self.B, self.rng = batch, rng

    def reset(self, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        base = rng.uniform(-0.1, 0.1, 4)
        self.s = np.tile(base, (self.B, 1))  # [th1, th2, thd1, thd2]
        self.done = np.zeros(self.B, dtype=bool)
        self.t = 0
        return self._obs()

    def _obs(self):
        th1, th2, d1, d2 = self.s.T
        return np.stack([np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2),
                         d1, d2], axis=1)

    def _dsdt(self, s_aug):
        m1 = m2 = 1.0
        l1 = 1.0
        lc1 = lc2 = 0.5
        I1 = I2 = 1.0
        g = 9.8
        a = s_aug[:, 4]
        th1, th2, dth1, dth2 = s_aug[:, 0], s_aug[:, 1], s_aug[:, 2], s_aug[:, 3]
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
        out = np.zeros_like(s_aug)
        out[:, 0], out[:, 1], out[:, 2], out[:, 3] = dth1, dth2, ddth1, ddth2
        return out

    def step(self, action):
        torque = action.astype(np.float64) - 1.0  # {0,1,2} -> {-1,0,1}
        s_aug = np.concatenate([self.s, torque[:, None]], axis=1)
        dt = 0.2
        # RK4, matching gymnasium's rk4 with dt=0.2 single interval
        k1 = self._dsdt(s_aug)
        k2 = self._dsdt(s_aug + dt / 2 * k1)
        k3 = self._dsdt(s_aug + dt / 2 * k2)
        k4 = self._dsdt(s_aug + dt * k3)
        ns = s_aug + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        th1 = self._wrap(ns[:, 0])
        th2 = self._wrap(ns[:, 1])
        d1 = np.clip(ns[:, 2], -4 * np.pi, 4 * np.pi)
        d2 = np.clip(ns[:, 3], -9 * np.pi, 9 * np.pi)
        live = ~self.done
        self.s[live] = np.stack([th1, th2, d1, d2], axis=1)[live]
        term = (-np.cos(self.s[:, 0])
                - np.cos(self.s[:, 1] + self.s[:, 0])) > 1.0
        reward = np.where(live & ~term, -1.0, 0.0)
        self.done |= term
        self.t += 1
        if self.t >= self.horizon:
            self.done[:] = True
        return self._obs(), reward, self.done.copy()

    @staticmethod
    def _wrap(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi


ENVS = {"CartPole-v1": VecCartPole, "Pendulum-v1": VecPendulum,
        "Acrobot-v1": VecAcrobot}
