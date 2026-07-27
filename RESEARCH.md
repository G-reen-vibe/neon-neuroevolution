# Research Notes: Modernizing NEAT

## 1. What NEAT actually is

NEAT (Stanley & Miikkulainen, 2002) evolves both weights and topology. Its three
mechanisms, and the problem each solves:

| Mechanism | Problem it solves |
|---|---|
| Innovation numbers | Aligning variable-shape genomes so crossover is meaningful |
| Speciation + fitness sharing | Protecting structural innovations from immediate competition |
| Complexification (start minimal) | Searching small topology spaces before large ones |

## 2. Why it stalled

1. **Noisy search signal.** Each genome receives one scalar fitness; structural
   mutations are accepted/rejected essentially by tournament luck. No information
   is shared across the population except through crossover, which is itself
   unreliable (the permutation problem is only partially solved by innovation
   numbers).
2. **Fragility.** Compatibility threshold, c1/c2/c3 coefficients, stagnation
   limits, elitism fractions — speciation alone is ~5 coupled hyperparameters.
3. **No vectorization.** Every genome has a different shape, so evaluation is a
   Python loop over irregular graphs. On modern hardware this wastes 100–1000x.
4. **Crossover is often harmful.** Ablations in the literature repeatedly find
   NEAT-without-crossover performs comparably or better.

## 3. Directions considered

### D1. Tensorized NEAT (GPU-batch the classic algorithm)
Pad genomes into fixed-size tensors, batch the forward passes.
**Strength:** large constant-factor speedup, proven by recent tensorized
implementations. **Weakness:** the *search dynamics* remain 2002-noisy; this is
engineering, not an algorithm. **Verdict:** rejected as the core contribution;
the vectorization lesson is absorbed into whatever we build.

### D2. Quality-Diversity NEAT (MAP-Elites archive replaces speciation)
Keep an archive of elites over a behavior space; structural innovation survives
by being *different*, not by species protection.
**Strength:** principled diversity, strong on deceptive tasks.
**Weakness:** requires per-task behavior descriptors; mutation/crossover
untouched, so the core signal-to-noise problem remains. **Verdict:** rejected as
core (partial renovation), though NEON's entropy control achieves the same goal
domain-independently.

### D3. Latent-space topology evolution
Learn an autoencoder over graphs; evolve in the latent space.
**Strength:** smooth search space. **Weakness:** exactly the "model bolted on
top" hybrid the brief warns against; needs graph training data; opaque.
**Verdict:** rejected.

### D4. Gradient-grown networks (cascade-correlation revival)
Grow units greedily, train by backprop.
**Weakness:** abandons the gradient-free, reward-only setting that makes
neuroevolution useful (sparse/deceptive rewards, non-differentiable sims).
**Verdict:** rejected — changes the problem instead of solving it.

### D5. Single-lineage annealed structural search
One graph, simulated annealing on structure, ES on weights.
**Strength:** minimal and cheap. **Weakness:** discards population-level
information aggregation entirely; historically collapses on deceptive tasks.
**Verdict:** rejected.

### D6. Distributional reformulation → **NEON** (chosen)
**Key observation:** all three NEAT mechanisms are ad-hoc approximations of
maintaining and updating a *probability distribution over topologies*:

- speciation ≈ preventing distribution collapse (entropy control)
- crossover ≈ recombining information across samples (a distribution update
  does this exactly, at the sufficient-statistic level)
- innovation numbers ≈ a shared coordinate system (a fixed scaffold provides
  one for free)

So make the distribution first-class. Search over a parameterized distribution
on sparse DAGs and ascend the expected fitness by estimated natural gradient —
the NES/OpenAI-ES treatment, extended from weights to *structure itself*.

## 4. NEON specification

**Scaffold.** A fixed maximal DAG: input nodes, a pool of H potential hidden
nodes in topological order, output nodes. Every forward edge (input→hidden,
hidden→hidden with lower→higher order, input→output, hidden→output) is a
*potential* edge. The scaffold is the shared coordinate system; genomes never
need aligning because they are all points in the same space.

**Distribution.** Each potential edge e carries a gate logit a_e and a weight
mean mu_e. Each hidden node's activation is fixed (tanh). A network is sampled
as:

    m_e ~ Bernoulli(sigmoid(a_e))        # does the edge exist?
    w_e = mu_e + sigma * eps_e           # if it exists, its weight

**Update (one rule).** Evaluate N samples, rank-normalize fitness u_i (as in
OpenAI-ES: fixed utilities, zero-sum), then

    mu  += lr_mu / (N*sigma) * sum_i u_i * m_i * eps_i          # ES gradient
    a   += lr_a  / N         * sum_i u_i * (m_i - sigmoid(a))    # score function
    a   -= lr_a * c_cost * sigmoid(a) * (1 - sigmoid(a))         # minimality prior

Antithetic sampling on eps halves weight-gradient variance; rank shaping acts
as a baseline for the gate gradient.

**Where NEAT's mechanisms went:**

| NEAT mechanism | NEON principle |
|---|---|
| Complexification | Prior: gate logits init strongly negative (except input→output); connection-cost term keeps unused structure improbable |
| Speciation | Entropy: gate probabilities clamped away from 0/1 saturation, so every generation *is* a diverse cloud of topologies |
| Crossover | The update: per-edge statistics aggregated across the whole population each generation |
| Innovation numbers | Unneeded: fixed scaffold = shared coordinates |

**Complexity/speed.** The entire population evaluation is batched: activations
for all N networks advance through the topological order with masked matmuls.
No per-genome graph interpretation.

**Honest risks.**
1. Score-function gradients for gates are higher variance than weight
   gradients → mitigated by rank utilities and antithetic weight noise; verified
   empirically.
2. The hidden pool caps maximal size → make the pool generous (it is only a
   *ceiling*; the prior keeps realized networks minimal).
3. Bernoulli gates could saturate early → entropy floor via logit clamping.

## 5. Evaluation plan

- Tasks: CartPole-v1 (solve = 475 avg/100), Acrobot-v1, Pendulum-v1
  (continuous actions — NEAT handles these too via neat-python).
- Baselines: neat-python NEAT (community-standard configs), OpenAI-ES on a
  fixed dense MLP (to show structure search isn't dead weight).
- Metrics: episodes to solve, wall-clock to solve, final performance,
  realized network size (active edges), across >= 5 seeds.
