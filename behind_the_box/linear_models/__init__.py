"""Linear models family.

Implementation files here are named ``NN-algorithm-name.py`` so they sort
next to their paired ``NN-algorithm-name.md`` Explanation. Python cannot
import such names directly (they start with a digit and contain hyphens),
so this package auto-discovers them at import time:

1. Every ``NN-*.py`` file in this directory is loaded and registered in
   ``sys.modules`` as ``behind_the_box.linear_models.<snake_case_stem>``
   (e.g. ``01-closed-form-regression.py`` → ``.closed_form_regression``).
2. Each public class defined in those files is re-exported here, so
   ``from behind_the_box.linear_models import LinearRegression`` works.
3. ``MODELS`` maps each file's stem to its model class, for tooling that
   iterates over every model (e.g. ``experiments/benchmark.py``).

Adding a new algorithm file requires no change to this module.
"""

import importlib.util
import inspect
import pathlib
import sys

_PACKAGE_DIR = pathlib.Path(__file__).parent

#: Maps algorithm file stem → model class, e.g.
#: {"01-closed-form-regression": LinearRegression}
MODELS: dict[str, type] = {}

__all__ = ["MODELS"]


def _load(path: pathlib.Path) -> dict[str, type]:
    """Import one ``NN-name.py`` file; return the classes it defines."""
    # "01-closed-form-regression" → alias "closed_form_regression"
    alias_stem = path.stem.split("-", 1)[1].replace("-", "_")
    alias = f"{__name__}.{alias_stem}"

    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)

    # Classes defined in the file itself (not imported into it).
    return {
        name: obj
        for name, obj in vars(module).items()
        if inspect.isclass(obj) and obj.__module__ == alias
    }


for _path in sorted(_PACKAGE_DIR.glob("[0-9][0-9]-*.py")):
    _classes = _load(_path)
    globals().update(_classes)
    __all__ += list(_classes)
    for _cls in _classes.values():
        MODELS[_path.stem] = _cls
