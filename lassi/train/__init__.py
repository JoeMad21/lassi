"""The train layer: `lassi train` on a train recipe, behind the Trainer interface (bible Training Module; task P17.8).

lassi.train.run runs a train recipe (run_training) and lassi.train.data reads
its data and takes the data split hash. A Trainer (lassi.core.interfaces) is
the replaceable backend; trainer modules join this package and are imported
here, eagerly, so that importing lassi.train registers every trainer. Every
module imports only the standard library and lassi at module level, and a
framework only inside a function when a trainer is built or trains
(PHASE-NOTES P0 and P17, the registration rule). Two trainers are
registered: "trl" (lassi.train.trl_trainer; task P17.9), which writes its
checkpoint records through lassi.train.checkpoint, scores grpo with
lassi.train.fixture_reward, and reads its base models through
lassi.train.bases (2026-10-23); and "ttxla" (lassi.train.ttxla_trainer;
task P17.38), which trains sft on a Tenstorrent card through the tt-xla
wheel, running lassi/train/ttxla_worker.py in the wheel's own venv, under
the card guard of `lassi train --silicon` (lassi.train.silicon).
lassi.train.prepare is the separate step that downloads a Hub base into
$HF_HOME/hub (tools/prepare_train.py).
"""

from __future__ import annotations

from lassi.train import bases, checkpoint, data, fixture_reward, prepare, run, silicon, trl_trainer, ttxla_trainer

__all__ = [
    "bases", "checkpoint", "data", "fixture_reward", "prepare", "run", "silicon", "trl_trainer", "ttxla_trainer",
]
