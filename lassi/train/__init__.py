"""The train layer: `lassi train` on a train recipe, behind the Trainer interface (bible Training Module; task P17.8).

lassi.train.run runs a train recipe (run_training) and lassi.train.data reads
its data and takes the data split hash. A Trainer (lassi.core.interfaces) is
the replaceable backend; trainer modules join this package and are imported
here, eagerly, so that importing lassi.train registers every trainer. Every
module imports only the standard library and lassi at module level, and a
framework only inside a function when a trainer is built or trains
(PHASE-NOTES P0 and P17, the registration rule). One trainer is registered:
"trl" (lassi.train.trl_trainer; task P17.9), which writes its checkpoint
records through lassi.train.checkpoint and scores grpo with
lassi.train.fixture_reward.
"""

from __future__ import annotations

from lassi.train import checkpoint, data, fixture_reward, run, trl_trainer

__all__ = ["checkpoint", "data", "fixture_reward", "run", "trl_trainer"]
