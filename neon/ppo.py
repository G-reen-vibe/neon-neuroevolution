"""PPO (Schulman et al. 2017) in pure NumPy with manual backpropagation.

Included as the gradient-based deep-RL reference point. Implemented from
scratch rather than pulled from a framework because the only available torch
wheel carries ~2GB of CUDA dependencies onto a single-core box; a hand-written
version is small, fast for these network sizes, and keeps the whole benchmark
on one dependency-light stack.

Standard classic-control hyperparameters: 64x64 tanh actor and critic,
gamma 0.99, GAE lambda 0.95, clip 0.2, lr 3e-4, 10 epochs of minibatch 64 over
2048-step rollouts. Truncation vs termination is handled correctly (the value
of a time-limit-truncated state is bootstrapped, a terminated one is not).
"""

from __future__ import annotations

import numpy as np


class Net:
    """MLP with tanh hidden layers, linear output, manual fwd/bwd."""

    def __init__(self, sizes, rng, last_scale=0.01):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = (np.sqrt(2.0 / sizes[i]) if i < len(sizes) - 2
                     else last_scale)
            self.W.append(rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, x):
        acts = [x]
        h = x
        for i in range(len(self.W) - 1):
            h = np.tanh(h @ self.W[i] + self.b[i])
            acts.append(h)
        return h @ self.W[-1] + self.b[-1], acts

    def backward(self, dout, acts):
        L = len(self.W)
        gW = [None] * L
        gb = [None] * L
        gW[L - 1] = acts[-1].T @ dout
        gb[L - 1] = dout.sum(0)
        d = dout @ self.W[L - 1].T
        for i in range(L - 2, -1, -1):
            d = d * (1 - acts[i + 1] ** 2)
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = d @ self.W[i].T
        return gW, gb

    def plist(self):
        return self.W + self.b


class AdamList:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-5):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def apply(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class PPO:
    name = "ppo"

    def __init__(self, task, seed, hp=None):
        hp = hp or {}
        self.task = task
        self.rng = np.random.default_rng(seed)
        self.n_envs = hp.get("n_envs", 8)
        self.T = hp.get("T", 256)
        self.gamma = hp.get("gamma", 0.99)
        self.lam = hp.get("lam", 0.95)
        self.clip = hp.get("clip", 0.2)
        self.epochs = hp.get("epochs", 10)
        self.mb = hp.get("minibatch", 64)
        self.vf_coef = hp.get("vf_coef", 0.5)
        self.ent_coef = hp.get("ent_coef", 0.0)
        self.max_norm = hp.get("max_grad_norm", 0.5)
        h = hp.get("hidden", 64)
        self.discrete = task.discrete
        self.adim = task.n_out
        self.actor = Net([task.n_in, h, h, self.adim], self.rng)
        self.critic = Net([task.n_in, h, h, 1], self.rng, last_scale=1.0)
        self.log_std = np.zeros(self.adim)
        self.opt_a = AdamList(self.actor.plist(), hp.get("lr", 3e-4))
        self.opt_c = AdamList(self.critic.plist(), hp.get("lr", 3e-4))
        self.opt_s = AdamList([self.log_std], hp.get("lr", 3e-4))
        self.env = task.cls(batch=self.n_envs,
                            rng=np.random.default_rng(seed + 1))
        self.obs = self.env.reset(seed=seed + 12345, common=False)

    # ------------------------------------------------------------ distribution
    def _logp_entropy(self, out, act):
        if self.discrete:
            z = out - out.max(axis=1, keepdims=True)
            logp_all = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            p = np.exp(logp_all)
            logp = logp_all[np.arange(len(act)), act]
            ent = -(p * logp_all).sum(axis=1)
            return logp, ent, (p, logp_all)
        std = np.exp(self.log_std)
        z = (act - out) / std
        logp = (-0.5 * z**2 - self.log_std - 0.5 * np.log(2 * np.pi)).sum(1)
        ent = (self.log_std + 0.5 * np.log(2 * np.pi * np.e)).sum()
        return logp, np.full(len(act), ent), (z, std)

    def _sample(self, out):
        if self.discrete:
            z = out - out.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            u = self.rng.random((len(out), 1))
            act = (p.cumsum(axis=1) < u).sum(axis=1)
            return np.clip(act, 0, self.adim - 1), None
        std = np.exp(self.log_std)
        raw = out + std * self.rng.normal(0, 1, out.shape)
        return raw, raw

    # -------------------------------------------------------------- collection
    def step(self):
        T, E, sc = self.T, self.n_envs, self.task.obs_scale
        obs_b = np.zeros((T, E, self.task.n_in))
        nobs_b = np.zeros((T, E, self.task.n_in))
        act_b = (np.zeros((T, E), dtype=np.int64) if self.discrete
                 else np.zeros((T, E, self.adim)))
        logp_b = np.zeros((T, E))
        rew_b = np.zeros((T, E))
        done_b = np.zeros((T, E))
        term_b = np.zeros((T, E))
        for t in range(T):
            on = self.obs / sc
            out, _ = self.actor.forward(on)
            act, raw = self._sample(out)
            logp, _, _ = self._logp_entropy(out, act if self.discrete else raw)
            env_act = (act if self.discrete
                       else np.clip(raw, -1, 1) * self.task.act_scale)
            nobs, rew, done = self.env.step(env_act)
            obs_b[t], act_b[t], logp_b[t] = on, act, logp
            rew_b[t], done_b[t] = rew, done
            term_b[t] = self.env.last_term
            nobs_b[t] = nobs / sc
            if done.any():
                self.env.reset_rows(done, self.rng)
            self.obs = self.env._obs()
        # ------------------------------------------------------------- GAE
        flat_o = obs_b.reshape(-1, self.task.n_in)
        flat_n = nobs_b.reshape(-1, self.task.n_in)
        vals = self.critic.forward(flat_o)[0].reshape(T, E)
        nvals = self.critic.forward(flat_n)[0].reshape(T, E)
        adv = np.zeros((T, E))
        last = np.zeros(E)
        for t in reversed(range(T)):
            nonterm = 1.0 - term_b[t]      # bootstrap unless truly terminated
            notdone = 1.0 - done_b[t]      # GAE chain breaks at any boundary
            delta = rew_b[t] + self.gamma * nvals[t] * nonterm - vals[t]
            last = delta + self.gamma * self.lam * notdone * last
            adv[t] = last
        ret = adv + vals
        # ---------------------------------------------------------- optimize
        N = T * E
        o = flat_o
        a = act_b.reshape(N) if self.discrete else act_b.reshape(N, self.adim)
        lp_old = logp_b.reshape(N)
        A = adv.reshape(N)
        R = ret.reshape(N)
        A = (A - A.mean()) / (A.std() + 1e-8)
        idx = np.arange(N)
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for s in range(0, N, self.mb):
                j = idx[s: s + self.mb]
                self._update(o[j], a[j], lp_old[j], A[j], R[j])
        return T * E

    def _update(self, o, a, lp_old, A, R):
        n = len(o)
        out, acts_a = self.actor.forward(o)
        logp, ent, extra = self._logp_entropy(out, a)
        ratio = np.exp(np.clip(logp - lp_old, -20, 20))
        s1 = ratio * A
        s2 = np.clip(ratio, 1 - self.clip, 1 + self.clip) * A
        unclipped = (s1 <= s2).astype(np.float64)
        dlogp = -(unclipped * ratio * A) / n
        if self.discrete:
            p, logp_all = extra
            oh = np.zeros_like(p)
            oh[np.arange(n), a] = 1.0
            dout = dlogp[:, None] * (oh - p)
            if self.ent_coef:
                dH = -p * (logp_all + ent[:, None])
                dout += -self.ent_coef / n * dH
            dstd = None
        else:
            z, std = extra
            dout = dlogp[:, None] * (z / std)
            dstd = (dlogp[:, None] * (z**2 - 1.0)).sum(0)
            if self.ent_coef:
                dstd -= self.ent_coef * np.ones(self.adim)
        gW, gb = self.actor.backward(dout, acts_a)
        v, acts_c = self.critic.forward(o)
        dv = self.vf_coef * (v[:, 0] - R)[:, None] / n
        cW, cb = self.critic.backward(dv, acts_c)
        self._clip_apply(self.actor, self.opt_a, gW + gb)
        self._clip_apply(self.critic, self.opt_c, cW + cb)
        if dstd is not None:
            self.opt_s.apply([self.log_std], [np.clip(dstd, -10, 10)])
            np.clip(self.log_std, -3.0, 1.0, out=self.log_std)

    def _clip_apply(self, net, opt, grads):
        norm = np.sqrt(sum((g * g).sum() for g in grads))
        if norm > self.max_norm:
            grads = [g * (self.max_norm / (norm + 1e-8)) for g in grads]
        opt.apply(net.plist(), grads)

    # ------------------------------------------------------------- incumbent
    def act_fn(self, B):
        def act(obs):
            out, _ = self.actor.forward(obs)
            return out if self.discrete else np.tanh(out)
        return act

    def size(self):
        n = sum(p.size for p in self.actor.plist())
        return dict(params=int(n), hidden=64)
