"""Register hyphenated algorithm modules under importable aliases.

Python cannot import filenames that start with a digit or contain hyphens
(e.g. ``linear_models/01-closed-form-regression.py``). This conftest loads
each such file at test-collection time and registers it in ``sys.modules``
under a clean alias that tests can import normally.

Add one ``_register`` call here each time a new algorithm file is added.
"""

import importlib.util
import pathlib
import sys

_ROOT = pathlib.Path(__file__).parent


def _register(alias: str, rel_path: str) -> None:
    path = _ROOT / rel_path
    if not path.exists():
        return
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)


_register(
    "linear_models.closed_form_regression",
    "linear_models/01-closed-form-regression.py",
)
