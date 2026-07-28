"""NEAT baseline (Stanley & Miikkulainen 2002) via the reference
`neat-python` implementation.

The algorithm is untouched -- only the *evaluator* is accelerated: the
environment side is stepped as a batch while genome activations remain the
reference per-genome python calls. This gives NEAT the same fast environment
code every other method gets, so the comparison measures search quality
rather than implementation speed.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import neat

CONFIG_TEMPLATE = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1e18
pop_size              = {pop}
reset_on_extinction   = True
no_fitness_termination = True

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = {n_in}
num_outputs             = {n_out}
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
single_structural_mutation = False
structural_mutation_surer  = default
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


class NEATMethod:
    name = "neat"

    def __init__(self, task, seed, hp=None):
        from .methods import POPSIZE
        self.task, self.seed = task, seed
        self.pop_size = (hp or {}).get("pop_size", POPSIZE)
        text = CONFIG_TEMPLATE.format(pop=self.pop_size, n_in=task.n_in,
                                      n_out=task.n_out)
        fd, path = tempfile.mkstemp(suffix=".cfg")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation, path)
        os.unlink(path)
        import random
        random.seed(seed)
        np.random.seed(seed)
        self.pop = neat.Population(self.config)
        self.best = None
        self.best_fit = -np.inf
        self.gen = 0
        self._steps = 0

    # ------------------------------------------------------------ evaluation
    def _nets_act_fn(self, nets):
        def act(obs):
            return np.asarray([n.activate(o) for n, o in zip(nets, obs)])
        return act

    def _eval_genomes(self, genomes, config):
        nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]
        fit, steps = self.task.fitness(self._nets_act_fn(nets), len(nets),
                                       self.seed * 977 + self.gen * 31)
        self._steps += steps
        for (gid, genome), f in zip(genomes, fit):
            genome.fitness = float(f)
        i = int(np.argmax(fit))
        if fit[i] > self.best_fit:
            self.best_fit = float(fit[i])
            self.best = genomes[i][1]

    def step(self):
        self._steps = 0
        try:
            self.pop.run(self._eval_genomes, 1)
        except Exception:      # complete extinction -> restart population
            self.pop = neat.Population(self.config)
        self.gen += 1
        return self._steps

    # ------------------------------------------------------------- incumbent
    def act_fn(self, B):
        net = neat.nn.FeedForwardNetwork.create(self.best, self.config)
        return lambda obs: np.asarray([net.activate(o) for o in obs])

    def size(self):
        if self.best is None:
            return dict(params=0, hidden=0)
        conns = sum(1 for c in self.best.connections.values() if c.enabled)
        hidden = len(self.best.nodes) - self.task.n_out
        return dict(params=conns, hidden=max(0, hidden))
