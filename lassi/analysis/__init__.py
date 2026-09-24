"""Analysis: run metrics, intervals, and paper comparisons.

- lassi.analysis.stats: the Wilson score interval and the unbiased pass@k.
- lassi.analysis.paper: a paper's reference values, read from
  assets/scoring/lassi-paper.yaml, with their Wilson intervals and the B0
  interval.
- lassi.analysis.metrics: one metric table per arm and direction from the
  lassi profile's scores (bible Evaluation Protocol).
- lassi.analysis.tables: those tables as Markdown and Parquet.
"""

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
    "METRIC_NAMES",
    "PAPER_FILE",
    "MetricRow",
    "MetricTable",
    "PaperCount",
    "PaperMetric",
    "PaperValues",
    "b0_interval",
    "load_paper",
    "metric_tables",
    "metrics_markdown",
    "paper_interval",
    "pass_at_k",
    "read_metrics_parquet",
    "table_markdown",
    "wilson_interval",
    "write_metrics_parquet",
]
