"""The train layer: `lassi train` on a train recipe, behind the Trainer interface (bible Training Module; task P17.8).

lassi.train.run runs a train recipe (run_training) and lassi.train.data reads
its data and takes the data split hash. A Trainer (lassi.core.interfaces) is
the replaceable backend; trainer modules join this package and are imported
here, eagerly, so that importing lassi.train registers every trainer. Every
module imports only the standard library and lassi at module level, and a
framework only inside a function when a trainer is built or trains
(PHASE-NOTES P0 and P17, the registration rule). No trainer is registered
yet.
"""

from __future__ import annotations

from lassi.train import data, run

__all__ = ["data", "run"]
