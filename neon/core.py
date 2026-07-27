"""NEON: NeuroEvolution Over Network distributions.

A distributional successor to NEAT. Instead of evolving a population of
discrete graphs with mutation/crossover/speciation, NEON maintains a
parameterized probability distribution over sparse DAGs:

    per potential edge e:   gate logit a_e   (P(edge exists) = sigmoid(a_e))
                            weight mean mu_e (weight = mu_e + sigma * eps)

and ascends expected fitness with estimated natural-gradient steps computed
from a sampled population. NEAT's three mechanisms become principles:

    complexification -> minimality prior on gate probabilities
    speciation       -> entropy floor on gates (population is a topology cloud)
    crossover        -> the update itself (edge statistics pooled across pop)
    innovation nums  -> unneeded (fixed scaffold = shared coordinates)

The whole population is evaluated with batched masked matmuls; no per-genome
graph interpretation ever happens.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


def rank_utilities(fitness: np.ndarray) -> np.ndarray:
    """Zero-sum rank-based utilities (OpenAI-ES style fitness shaping)."""
    n = len(fitness)
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(fitness)] = np.arange(n)
    u = ranks / (n - 1) - 0.5  # in [-0.5, 0.5], zero mean
    return u


@dataclass
class NEONConfig:
    n_inputs: int
    n_outputs: int
    hidden_pool: int = 16          # ceiling on hidden nodes (not a target)
    popsize: int = 128             # must be even (antithetic pairs)
    sigma: float = 0.12            # weight perturbation scale
    lr_mu: float = 0.06            # weight-mean learning rate
    lr_gate: float = 0.35          # gate-logit learning rate
    conn_cost: float = 0.012       # minimality prior strength
    gate_init_io: float = 0.5      # initial logit for input->output edges
    gate_init_hidden: float = -3.0 # initial logit for edges via hidden pool
    logit_clip: float = 4.5        # entropy floor: |a_e| <= clip
    weight_clip: float = 6.0
    mu_l2: float = 0.002           # slight weight decay on means
    seed: int = 0


class NEON:
    """Distributional topology+weight search on a fixed DAG scaffold.

    Node ordering: [inputs][hidden pool][outputs]. An edge (i -> j) is
    potential iff order(i) < order(j) and j is not an input. Hidden and output
    nodes carry a bias (modeled as an always-orderable pseudo-input of 1).
    """

    def __init__(self, cfg: NEONConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        nI, nH, nO = cfg.n_inputs, cfg.hidden_pool, cfg.n_outputs
        self.nI, self.nH, self.nO = nI, nH, nO
        self.N = nI + 1 + nH + nO  # +1 = bias node (index nI), constant 1.0
        # potential-edge mask: strict lower topological order -> higher,
        # targets only hidden/output nodes.
        order = np.arange(self.N)
        src = order[:, None]
        dst = order[None, :]
        target_ok = dst >= nI + 1
        self.potential = (src < dst) & target_ok            # (N, N) bool
        # parameters
        self.mu = np.zeros((self.N, self.N))
        self.gate = np.full((self.N, self.N), -np.inf)
        io_direct = self.potential & (src <= nI) & (dst >= nI + 1 + nH)
        via_hidden = self.potential & ~io_direct
        self.gate[io_direct] = cfg.gate_init_io
        self.gate[via_hidden] = cfg.gate_init_hidden
        # small init on direct-path means so gradient signal exists at t=0
        self.mu[io_direct] = self.rng.normal(0.0, 0.3, io_direct.sum())
        self.hidden_slice = slice(nI + 1, nI + 1 + nH)
        self.output_slice = slice(nI + 1 + nH, self.N)

    # ---------------------------------------------------------------- sample
    def sample_population(self):
        """Sample masks and weights for the whole population.

        Weight noise is antithetic (pairs +eps/-eps); gate masks are sampled
        independently per individual (they are the structural exploration).
        Returns (masks, weights): each (P, N, N).
        """
        cfg = self.cfg
        P = cfg.popsize
        p_gate = self._gate_probs()
        masks = (self.rng.random((P, self.N, self.N)) < p_gate) & self.potential
        half = P // 2
        eps_half = self.rng.normal(0.0, 1.0, (half, self.N, self.N))
        eps = np.concatenate([eps_half, -eps_half], axis=0)
        weights = np.where(masks, self.mu + cfg.sigma * eps, 0.0)
        return masks, weights, eps

    def _gate_probs(self):
        p = np.zeros((self.N, self.N))
        pe = self.potential
        p[pe] = 1.0 / (1.0 + np.exp(-self.gate[pe]))
        return p

    # --------------------------------------------------------------- forward
    def act_batch(self, weights: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Forward pass for the whole population at once.

        weights: (P, N, N) — weights[p, i, j] is edge i->j for individual p.
        obs:     (P, nI)   — one observation per individual.
        returns: (P, nO)   — tanh outputs.
        """
        P = weights.shape[0]
        acts = np.zeros((P, self.N))
        acts[:, : self.nI] = obs
        acts[:, self.nI] = 1.0  # bias
        # hidden nodes advance in topological order; each depends only on
        # lower-indexed nodes, so a per-node masked dot suffices.
        for j in range(self.nI + 1, self.nI + 1 + self.nH):
            pre = np.einsum("pi,pi->p", acts[:, :j], weights[:, :j, j])
            acts[:, j] = np.tanh(pre)
        out_w = weights[:, :, self.output_slice]          # (P, N, nO)
        out = np.einsum("pi,pio->po", acts, out_w)
        return np.tanh(out)

    # ---------------------------------------------------------------- update
    def update(self, masks, eps, fitness):
        cfg = self.cfg
        u = rank_utilities(np.asarray(fitness, dtype=np.float64))
        P = len(u)
        p_gate = self._gate_probs()
        # ES gradient on weight means (only through active edges)
        g_mu = np.einsum("p,pij->ij", u, masks * eps) / (P * cfg.sigma)
        self.mu += cfg.lr_mu * g_mu - cfg.lr_mu * cfg.mu_l2 * self.mu
        np.clip(self.mu, -cfg.weight_clip, cfg.weight_clip, out=self.mu)
        # score-function gradient on gate logits
        g_gate = np.einsum("p,pij->ij", u, masks - p_gate) / P
        # minimality prior: pull probabilities down, scaled by bernoulli var
        prior = cfg.conn_cost * p_gate * (1.0 - p_gate)
        self.gate[self.potential] += cfg.lr_gate * (
            g_gate[self.potential] - prior[self.potential]
        )
        np.clip(self.gate, -cfg.logit_clip, cfg.logit_clip, out=self.gate)
        # keep -inf on non-potential entries (clip touched them)
        self.gate[~self.potential] = -np.inf

    # ------------------------------------------------------------------ MAP
    def mean_network(self):
        """Deterministic MAP network: edges with p>0.5 at mean weights."""
        mask = self._gate_probs() > 0.5
        return np.where(mask, self.mu, 0.0)[None, :, :]

    def stats(self):
        p = self._gate_probs()
        active = int((p > 0.5).sum())
        # hidden node is 'used' if it has an active incoming and outgoing edge
        m = p > 0.5
        used_hidden = 0
        for j in range(self.nI + 1, self.nI + 1 + self.nH):
            if m[:, j].any() and m[j, :].any():
                used_hidden += 1
        expected_edges = float(p[self.potential].sum())
        return {
            "active_edges": active,
            "expected_edges": expected_edges,
            "used_hidden": used_hidden,
        }
