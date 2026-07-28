"""Central registry of every method under comparison."""
from .methods import (NEONMethod, OpenAIES, SNES, CMAES, DeepGA, RandomSearch)
from .neat_method import NEATMethod
from .ppo import PPO

METHODS = {
    "neon": NEONMethod,          # ours
    "neat": NEATMethod,          # Stanley & Miikkulainen 2002
    "openai_es": OpenAIES,       # Salimans et al. 2017
    "snes": SNES,                # Schaul et al. 2011
    "cmaes": CMAES,              # Hansen 2003
    "ga": DeepGA,                # Such et al. 2017
    "ppo": PPO,                  # Schulman et al. 2017
    "random": RandomSearch,      # sanity floor
}

BASELINES = [m for m in METHODS if m != "neon"]
