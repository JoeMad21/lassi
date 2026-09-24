"""Scoring: code similarity measures and score profiles.

- lassi.scoring.similarity: the faithful Sim-T and Sim-L, and the C-aware Sim-T.
- lassi.scoring.df_v0: the ScoreProfile `df-v0` (bible Training Module, Reward
  Function), its weights read from assets/scoring/df-v0.yaml.

Importing this package registers the ScoreProfile `df-v0` in the default
registry. Wiring the similarity values into metrics is P2's work.
"""

from lassi.scoring.df_v0 import DfV0Profile, Weights, guard_state, load_weights, warning_count
from lassi.scoring.similarity import Similarity, c_tokens, measure, sim_l, sim_t, sim_t_c

__all__ = [
    "DfV0Profile",
    "Similarity",
    "Weights",
    "c_tokens",
    "guard_state",
    "load_weights",
    "measure",
    "sim_l",
    "sim_t",
    "sim_t_c",
    "warning_count",
]
