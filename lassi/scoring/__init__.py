"""Scoring: code similarity measures and score profiles.

- lassi.scoring.similarity: the faithful Sim-T and Sim-L, and the C-aware Sim-T.
- lassi.scoring.df_v0: the ScoreProfile `df-v0` (bible Training Module, Reward
  Function), its weights read from assets/scoring/df-v0.yaml.
- lassi.scoring.lassi_profile: the ScoreProfile `lassi` (bible Evaluation
  Protocol, LASSI reproduction row), its component order and notes read from
  assets/scoring/lassi.yaml.

Importing this package registers the ScoreProfiles `df-v0` and `lassi` in the
default registry.
"""

from lassi.scoring.df_v0 import DfV0Profile, Weights, guard_state, load_weights, warning_count
from lassi.scoring.lassi_profile import LassiProfile, ProfileFile, load_profile
from lassi.scoring.similarity import Similarity, c_tokens, measure, sim_l, sim_t, sim_t_c

__all__ = [
    "DfV0Profile",
    "LassiProfile",
    "ProfileFile",
    "Similarity",
    "Weights",
    "c_tokens",
    "guard_state",
    "load_profile",
    "load_weights",
    "measure",
    "sim_l",
    "sim_t",
    "sim_t_c",
    "warning_count",
]
