"""Dataset loaders for the benchmark harness.

Each loader returns (X, y, problem) where problem is "regression" or
"classification" — used to match datasets to compatible models.
Classification datasets arrive with model 05 (logistic regression).
"""

import pathlib

import numpy as np
from sklearn.datasets import fetch_california_housing

_ADVERTISING_CSV = (
    pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "advertising.csv"
)

Dataset = tuple[np.ndarray, np.ndarray, str]


def synthetic(seed: int = 42) -> Dataset:
    """500-sample, 5-feature regression data with known ground truth."""
    rng = np.random.default_rng(seed)
    true_coef = np.array([2.0, -1.5, 0.8, 0.0, 3.2])
    X = rng.standard_normal((500, 5))
    y = X @ true_coef + 5.0 + rng.standard_normal(500) * 0.5
    return X, y, "regression"


def california(seed: int = 42) -> Dataset:  # noqa: ARG001 - uniform loader signature
    """California Housing: 20 640 samples, 8 features (needs internet once)."""
    data = fetch_california_housing()
    return data.data, data.target, "regression"


def advertising(seed: int = 42) -> Dataset:  # noqa: ARG001 - uniform loader signature
    """Advertising: 200 samples, 3 features (TV, radio, newspaper → sales)."""
    if not _ADVERTISING_CSV.exists():
        msg = f"{_ADVERTISING_CSV} not found — download from Kaggle"
        raise FileNotFoundError(msg)

    with open(_ADVERTISING_CSV) as f:
        header = f.readline().strip()
    cols = [c.strip(' "').lower() for c in header.split(",")]
    feature_cols = [cols.index("tv"), cols.index("radio"), cols.index("newspaper")]
    target_col = cols.index("sales")

    data = np.genfromtxt(_ADVERTISING_CSV, delimiter=",", skip_header=1)
    return data[:, feature_cols], data[:, target_col], "regression"


LOADERS = {
    "synthetic": synthetic,
    "california": california,
    "advertising": advertising,
}
