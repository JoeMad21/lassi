"""Judges: rubrics, the model family table, and (from task P15.2) the registered Judges (bible Agents And Judges).

A judge screens or scores a candidate with a rubric and returns a Verdict
(lassi.core.interfaces), always reported as [JUDGED] and never as a
measurement (Agent Rule 14). This package holds the generic pieces judges
share (task P15.1): the rubric loader (lassi.judges.rubric), which reads a
versioned prompt and an output schema under assets/rubrics/<name>/, and the
model family table's reader (lassi.judges.families), which says which
family a model id belongs to, so a recipe can keep a judge's family apart
from the generator's.

Judges here are generic components: no module names a project, a suite, a
model, or a host, and none imports the projects package (Agent Rule 3).
Importing this package registers no component yet.
"""
