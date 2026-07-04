"""Behind the Box — ML algorithms implemented from scratch, NumPy only.

Each family subpackage (``linear_models``, later ``kernel_methods``, …)
holds paired Explanation (``.md``) and Implementation (``.py``) files.
``MODELS`` aggregates every family's registry so tooling can iterate over
all implemented models: keys are algorithm file stems, values are classes.
"""

from behind_the_box import linear_models

#: All models across all families, keyed by algorithm file stem.
MODELS: dict[str, type] = {
    **linear_models.MODELS,
}

__all__ = ["MODELS", "linear_models"]
