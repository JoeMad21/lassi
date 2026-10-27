"""Analysis: run metrics, intervals, and paper comparisons.

- lassi.analysis.stats: the Wilson score interval and the unbiased pass@k.
- lassi.analysis.paper: a paper's reference values, read from
  assets/scoring/lassi-paper.yaml, with their Wilson intervals and the B0
  interval.
- lassi.analysis.metrics: one metric table per arm and direction from the
  lassi profile's scores (bible Evaluation Protocol).
- lassi.analysis.tables: those tables as Markdown and Parquet.
- lassi.analysis.guard_coverage: how many attempts the host-compute guard
  checked, and why the rest were not checked, per run and per target
  language (task P17.14).
- lassi.analysis.adversary: the adversary metrics per arm, suite,
  adversary, and direction from the trials' adversary records (task
  P16.6): the confirmed-divergence rate, the invalid-input rejection rate,
  and the contract-finding count, in Markdown and Parquet.
- lassi.analysis.tt_rates: Tenstorrent rates (task P17.42): an item's
  analytic work over a silicon run's measured device kernel time, each rate
  naming its numerator and denominator, a percent of the vendor's peak
  labelled with its source, and a training run's step summary; none from a
  simulator (Agent Rule 2).
"""

from lassi.analysis.adversary import (
    ADVERSARY_METRICS,
    AdversaryRow,
    AdversaryTable,
    adversary_markdown,
    adversary_rows,
    adversary_tables,
    read_adversary_parquet,
    write_adversary_parquet,
)
from lassi.analysis.metrics import METRIC_NAMES, MetricRow, MetricTable, metric_tables
from lassi.analysis.paper import (
    PAPER_FILE,
    PaperCount,
    PaperMetric,
    PaperValues,
    b0_interval,
    load_paper,
    paper_interval,
)
from lassi.analysis.stats import pass_at_k, wilson_interval
from lassi.analysis.tables import metrics_markdown, read_metrics_parquet, table_markdown, write_metrics_parquet

__all__ = [
    "ADVERSARY_METRICS",
    "METRIC_NAMES",
    "PAPER_FILE",
    "AdversaryRow",
    "AdversaryTable",
    "MetricRow",
    "MetricTable",
    "PaperCount",
    "PaperMetric",
    "PaperValues",
    "adversary_markdown",
    "adversary_rows",
    "adversary_tables",
    "b0_interval",
    "load_paper",
    "metric_tables",
    "metrics_markdown",
    "paper_interval",
    "pass_at_k",
    "read_adversary_parquet",
    "read_metrics_parquet",
    "table_markdown",
    "wilson_interval",
    "write_adversary_parquet",
    "write_metrics_parquet",
]
