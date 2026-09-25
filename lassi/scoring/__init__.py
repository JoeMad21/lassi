"""Scoring: code similarity measures, score profiles, and the scoring pass over a finished run.

- lassi.scoring.similarity: the faithful Sim-T and Sim-L, and the C-aware Sim-T.
- lassi.scoring.df_v0: the ScoreProfile `df-v0` (bible Training Module, Reward
  Function), its weights read from assets/scoring/df-v0.yaml.
- lassi.scoring.lassi_profile: the ScoreProfile `lassi` (bible Evaluation
  Protocol, LASSI reproduction row), its component order and notes read from
  assets/scoring/lassi.yaml.
- lassi.scoring.profiles: build_profile, which builds a registered
  ScoreProfile by name with the bench root it needs (the capability
  reads_bench_sources), for the scoring pass and the runner alike.
- lassi.scoring.score_run: `lassi score`, which scores a finished run tree
  and writes its review packet under <runs root>/scores/<score id>. It is
  not imported here; import it by name.
- lassi.scoring.run_scoring: the recipe's `score` and `metrics` in `lassi run`
  (the runner imports it).

Importing this package registers the ScoreProfiles `df-v0` and `lassi` in the
default registry.
"""

from lassi.scoring.df_v0 import DfV0Profile, Weights, guard_state, load_weights, warning_count
from lassi.scoring.lassi_profile import LassiProfile, ProfileFile, load_profile
from lassi.scoring.profiles import build_profile, reads_bench_sources
from lassi.scoring.similarity import Similarity, c_tokens, measure, sim_l, sim_t, sim_t_c

__all__ = [
    "DfV0Profile",
    "LassiProfile",
    "ProfileFile",
    "Similarity",
    "Weights",
    "build_profile",
    "c_tokens",
    "guard_state",
    "load_profile",
    "load_weights",
    "measure",
    "reads_bench_sources",
    "sim_l",
    "sim_t",
    "sim_t_c",
    "warning_count",
]
