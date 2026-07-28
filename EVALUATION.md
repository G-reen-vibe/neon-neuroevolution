# Experimental protocol

## Stack

Pure **Python 3.12 + NumPy**, plus `cma` (reference CMA-ES), `neat-python`
(reference NEAT), `matplotlib`/`scipy` for analysis, and `gymnasium` used
**only** to verify environment correctness.

Rationale for a NumPy-only stack on this machine (1 CPU core, no GPU):

* **JAX/XLA** buys little for 20-node networks on one core, and pays a JIT
  compile cost per configuration.
* **PyTorch** is unavailable in a useful form: the only reachable wheel is
  526 MB and drags in ~2 GB of CUDA dependencies for a machine with no GPU.
  PPO is therefore implemented from scratch in NumPy with manual
  backpropagation (`neon/ppo.py`), validated to solve CartPole in 20k steps
  and Pendulum in ~150k.
* Keeping every method on one stack means all of them share **identical**
  environment code, so measured differences reflect search quality rather
  than implementation speed.

Environments are re-implemented as natively vectorized NumPy batches
(`neon/envs.py`), which is what makes population methods affordable here:
~90k-500k env-steps/second versus ~5k for per-instance gymnasium stepping.

## Benchmarks

Five tasks spanning distinct difficulty regimes, not five variations of one:

| task | regime | horizon | budget | solve |
|---|---|---|---|---|
| CartPole-v1 | dense reward, trivial | 500 | 0.5M | 475 |
| Acrobot-v1 | dense reward, underactuated swing | 500 | 1.5M | -90 |
| MountainCar-v0 | sparse reward, deceptive, discrete | 200 | 2M | -110 |
| Pendulum-v1 | continuous control, strong local optimum | 200 | 3M | -180 |
| MountainCarContinuous-v0 | sparse reward, exploration-limited | 500 | 3M | 90 |

**Correctness.** Every environment is verified step-for-step against
gymnasium (`tests/test_envs.py`); maximum observed discrepancy is 1.7e-6,
which is float32 rounding in gymnasium. The single intentional deviation is
MountainCarContinuous' horizon, shortened 999 → 500 (dynamics unchanged) so
that population methods get a workable number of generations within a budget
PPO can also be run at; this is documented in the source.

**CartPole is retained but is not discriminative** — uniform random search
over MLP weights solves it in ~14k steps. It is reported because it is the
canonical NEAT benchmark, and its non-discrimination is itself a result.

## Baselines

| method | reference | family |
|---|---|---|
| NEAT | Stanley & Miikkulainen 2002 | topology-evolving neuroevolution |
| OpenAI-ES | Salimans et al. 2017 | evolution strategy, fixed topology |
| SNES | Schaul et al. 2011 | natural ES, per-parameter step sizes |
| CMA-ES | Hansen 2003 (`cma` package) | full-covariance black-box optimizer |
| Deep GA | Such et al. 2017 | genetic algorithm, no crossover |
| PPO | Schulman et al. 2017 | gradient-based deep RL |
| Random search | — | sanity floor every method must clear |

All use literature-standard hyperparameters. All population methods share
`popsize=64`; all fixed-topology baselines use the same 16-unit tanh MLP.

## Protocol

* **Budget currency: training environment steps.** Evaluation steps are
  recorded but never charged, per standard convention.
* **Fitness uses common random numbers** — every individual in a generation
  faces identical initial states, so fitness ranks compare policies, not luck.
* **Held-out evaluation.** Performance is always measured on 24 initial
  states drawn from a fixed seed (`EVAL_SEED = 20240115`), identical for
  every method, every seed and every evaluation point. Training never sees
  them, so no method can be measured on its own training conditions.
* **Identical action decoding.** Discrete tasks use one output per action and
  argmax for *every* method — including NEON, which could otherwise use a
  single output on binary tasks and gain an unearned size advantage.
* **20 evaluation points per run**, giving learning curves on a common grid.
* **Solved** = held-out score first reaches the task threshold.
* **Seeds:** 10 per (method, task) for all methods except PPO, which uses 5
  (it is ~15x more expensive in wall-clock and serves as a reference point
  rather than a primary comparison). Unequal n is handled correctly by the
  stratified bootstrap.

## Statistics

Following Agarwal et al. 2021, *"Deep RL at the Edge of the Statistical
Precipice"*, point estimates are **interquartile means (IQM)** with **95%
stratified-bootstrap confidence intervals** (10,000 resamples, seeds
resampled within each task so no task dominates). Bare means over few seeds
are not reported as headline numbers.

Scores are normalized per task as
`(score - random_policy_score) / (solve_threshold - random_policy_score)`,
so 0 = uniform-random policy and 1 = solved. The random-policy reference is
measured empirically on the same held-out seeds
(`results/reference_scores.json`).

Reported per task: final score, normalized IQM with CI, solve rate, median
steps-to-solve (censored at budget when unsolved), median wall-clock, and
median solution size. Aggregated across tasks: IQM with CI, plus a
**performance profile** (fraction of runs above each normalized threshold),
which shows the whole distribution instead of a single summary.

## Running

```bash
python experiments/run.py --seeds 0-9 --max-minutes 25   # resumable chunk
python experiments/run.py --status                       # progress
python experiments/aggregate.py                          # tables + figures
```

Each (method, task, seed) cell is written to its own JSON file and skipped if
present, so the sweep survives interruption and resumes where it stopped.
