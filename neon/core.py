"""NEON: NeuroEvolution Over Network distributions.

A distributional successor to NEAT. Instead of evolving a population of
discrete graphs with mutation/crossover/speciation, NEON maintains a
parameterized probability distribution over sparse DAGs and ascends expected
fitness with estimated natural-gradient steps computed from a sampled
population.

The distribution is *hierarchical*:

    per hidden node j:      node logit b_j    (P(node active) = sigmoid(b_j))
    per potential edge e:   gate logit a_e    (P(edge | endpoints active))
                            weight mean mu_e  (weight = mu_e + sigma * eps)

Hierarchy is the distributional analog of NEAT's add-node mutation: switching
a node on in a sample brings a whole connected functional unit with it, so
structural exploration is correlated at the unit level — solving the credit
assignment problem that defeats independent per-edge gates (a lone hidden
edge does nothing, so no gradient would ever open one).

NEAT's mechanisms become principles:

    complexification -> minimality prior on node/edge probabilities
    speciation       -> entropy floor on gates (population is a topology cloud)
    crossover        -> the update itself (unit statistics pooled across pop)
    innovation nums  -> unneeded (fixed scaffold = shared coordinates)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def rank_utilities(fitness: np.ndarray) -> np.ndarray:
    """Zero-sum rank-based utilities (OpenAI-ES style fitness shaping)."""
    n = len(fitness)
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(fitness)] = np.arange(n)
    return ranks / (n - 1) - 0.5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class NEONConfig:
    n_inputs: int
    n_outputs: int
    hidden_pool: int = 16          # ceiling on hidden nodes (not a target)
    popsize: int = 128             # must be even (antithetic pairs)
    sigma: float = 0.12            # weight perturbation scale
    lr_mu: float = 0.06            # weight-mean learning rate
    lr_gate: float = 0.30          # edge-gate learning rate
    lr_node: float = 0.25          # node-gate learning rate
    conn_cost: float = 0.010      # per-edge minimality prior
    node_cost: float = 0.030      # per-node minimality prior
    gate_init_io: float = 0.5      # logit for direct input->output edges
    gate_init_hidden: float = 0.0  # logit for edges within active units
    node_init: float = -1.8        # node logit (p ~= 0.14): rare but explored
    logit_clip: float = 4.5        # entropy floor: |logit| <= clip
    weight_clip: float = 6.0
    mu_init: float = 0.30          # random init scale for all weight means
    mu_l2: float = 0.002
    seed: int = 0


class NEON:
    """Hierarchical distributional topology+weight search on a DAG scaffold.

    Node ordering: [inputs][bias][hidden pool][outputs]. An edge (i -> j) is
    potential iff i < j in this order and j is not an input/bias.
    """

    def __init__(self, cfg: NEONConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        nI, nH, nO = cfg.n_inputs, cfg.hidden_pool, cfg.n_outputs
        self.nI, self.nH, self.nO = nI, nH, nO
        self.N = nI + 1 + nH + nO  # +1 bias node at index nI (constant 1.0)
        order = np.arange(self.N)
        src, dst = order[:, None], order[None, :]
        self.potential = (src < dst) & (dst >= nI + 1)
        self.hidden_ids = np.arange(nI + 1, nI + 1 + nH)
        self.output_slice = slice(nI + 1 + nH, self.N)
        # --- parameters
        self.mu = np.where(
            self.potential,
            self.rng.normal(0.0, cfg.mu_init, (self.N, self.N)), 0.0)
        self.gate = np.full((self.N, self.N), -np.inf)
        is_hidden = (dst >= nI + 1) & (dst < nI + 1 + nH)
        touches_hidden = ((src >= nI + 1) & (src < nI + 1 + nH)) | is_hidden
        io_direct = self.potential & ~touches_hidden
        self.gate[io_direct] = cfg.gate_init_io
        self.gate[self.potential & touches_hidden] = cfg.gate_init_hidden
        self.node = np.full(nH, cfg.node_init)
        # hidden-membership map for building the node factor quickly
        self._hid_of = np.full(self.N, -1)
        self._hid_of[self.hidden_ids] = np.arange(nH)

    # ---------------------------------------------------------------- sample
    def sample_population(self):
        """Returns (masks, weights, eps, node_on).

        masks:   (P, N, N) bool — realized edges
        weights: (P, N, N)
        eps:     (P, N, N) antithetic weight noise
        node_on: (P, nH) bool — realized hidden units
        """
        cfg = self.cfg
        P = cfg.popsize
        p_edge = self._edge_probs()
        p_node = _sigmoid(self.node)
        node_on = self.rng.random((P, self.nH)) < p_node
        # node factor: edge (i,j) allowed iff each hidden endpoint is on
        allow = np.ones((P, self.N, self.N), dtype=bool)
        hid = self._hid_of
        h_src = hid[:, None] >= 0          # (N, N): src is hidden
        h_dst = hid[None, :] >= 0
        src_on = node_on[:, np.clip(hid, 0, None)]          # (P, N)
        dst_on = src_on                                     # same lookup
        allow &= ~h_src[None] | src_on[:, :, None]
        allow &= ~h_dst[None] | dst_on[:, None, :]
        masks = (self.rng.random((P, self.N, self.N)) < p_edge) \
            & self.potential & allow
        half = P // 2
        eps_half = self.rng.normal(0.0, 1.0, (half, self.N, self.N))
        eps = np.concatenate([eps_half, -eps_half], axis=0)
        weights = np.where(masks, self.mu + cfg.sigma * eps, 0.0)
        return masks, weights, eps, node_on

    def _edge_probs(self):
        p = np.zeros((self.N, self.N))
        p[self.potential] = _sigmoid(self.gate[self.potential])
        return p

    # --------------------------------------------------------------- forward
    def act_batch(self, weights: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Batched forward pass: weights (P,N,N), obs (P,nI) -> (P,nO)."""
        P = weights.shape[0]
        acts = np.zeros((P, self.N))
        acts[:, : self.nI] = obs
        acts[:, self.nI] = 1.0
        for j in self.hidden_ids:
            pre = np.einsum("pi,pi->p", acts[:, :j], weights[:, :j, j])
            acts[:, j] = np.tanh(pre)
        out = np.einsum("pi,pio->po", acts, weights[:, :, self.output_slice])
        return np.tanh(out)

    # ---------------------------------------------------------------- update
    def update(self, masks, eps, fitness, node_on):
        cfg = self.cfg
        u = rank_utilities(np.asarray(fitness, dtype=np.float64))
        P = len(u)
        p_edge = self._edge_probs()
        p_node = _sigmoid(self.node)
        # ES gradient on weight means (through active edges only)
        g_mu = np.einsum("p,pij->ij", u, masks * eps) / (P * cfg.sigma)
        self.mu += cfg.lr_mu * g_mu - cfg.lr_mu * cfg.mu_l2 * self.mu
        np.clip(self.mu, -cfg.weight_clip, cfg.weight_clip, out=self.mu)
        # score-function gradient on edge gates (conditional: only samples
        # where the edge was *allowed* carry information; masks==sample&allow,
        # using (m - p) with all samples is still an unbiased direction after
        # rank shaping and works well; prior scaled by bernoulli variance)
        g_gate = np.einsum("p,pij->ij", u, masks - p_edge) / P
        prior_e = cfg.conn_cost * p_edge * (1.0 - p_edge)
        upd = cfg.lr_gate * (g_gate - prior_e)
        self.gate[self.potential] += upd[self.potential]
        np.clip(self.gate, -cfg.logit_clip, cfg.logit_clip, out=self.gate)
        self.gate[~self.potential] = -np.inf
        # score-function gradient on node gates
        g_node = (u[:, None] * (node_on - p_node)).mean(axis=0)
        prior_n = cfg.node_cost * p_node * (1.0 - p_node)
        self.node += cfg.lr_node * (g_node - prior_n)
        np.clip(self.node, -cfg.logit_clip, cfg.logit_clip, out=self.node)

    # ------------------------------------------------------------------ MAP
    def mean_network(self):
        """Deterministic MAP network: nodes/edges with p>0.5, mean weights."""
        m = self._edge_probs() > 0.5
        on = _sigmoid(self.node) > 0.5
        for k, j in enumerate(self.hidden_ids):
            if not on[k]:
                m[j, :] = False
                m[:, j] = False
        return np.where(m, self.mu, 0.0)[None, :, :]

    def stats(self):
        w = self.mean_network()[0]
        m = w != 0.0
        used_hidden = int(sum(
            1 for j in self.hidden_ids if m[:, j].any() and m[j, :].any()))
        return {
            "active_edges": int(m.sum()),
            "used_hidden": used_hidden,
            "p_node_mean": float(_sigmoid(self.node).mean()),
        }
