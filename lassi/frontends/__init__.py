"""Language frontends: source code raised to the hub level (bible Language Frontends, Frontend Rules).

Importing this package registers the Frontend python-subset (lassi.frontends.python_subset) in the default
registry. hub_tag (lassi.frontends.hub) is the tag rule every frontend applies to the module it returns, from the
dialects the module holds, never from the language. A source outside a frontend's published subset raises
lassi.core.interfaces.FrontendRefusal, the one refusal every frontend raises.
"""

from lassi.frontends.hub import HUB_LEVEL, LOW, LOW_DIALECTS, STRUCTURED, hub_tag
from lassi.frontends.python_subset import PythonSubsetFrontend

__all__ = ["HUB_LEVEL", "LOW", "LOW_DIALECTS", "STRUCTURED", "PythonSubsetFrontend", "hub_tag"]
