"""Dataset loaders — the single source of truth for benchmark and test data.

Analogue of ``sklearn.datasets``. Consumed by ``tests/conftest.py`` (which
wraps these loaders as pytest fixtures) and ``experiments/benchmark.py``.

Each loader returns (X, y, problem) where problem is "regression" or
"classification" — used to match datasets to compatible models.
Classification datasets arrive with model 05 (logistic regression).
"""

import os
import pathlib
import urllib.request

import numpy as np
from sklearn.datasets import fetch_california_housing

Dataset = tuple[np.ndarray, np.ndarray, str]

# Local fixture kept for pre-existing manual downloads; only found when
# running from a repo checkout.
_ADVERTISING_FIXTURE = (
    pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "advertising.csv"
)
# Public source: the ISLR textbook's own hosting (same dataset as Kaggle).
_ADVERTISING_URL = "https://www.statlearning.com/s/Advertising.csv"


def _cache_dir() -> pathlib.Path:
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    base = pathlib.Path(xdg) if xdg else pathlib.Path.home() / ".cache"
    return base / "behind_the_box"


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
    """Advertising: 200 samples, 3 features (TV, radio, newspaper → sales).

    Resolution order: repo fixture (tests/fixtures/advertising.csv) →
    cached copy (~/.cache/behind_the_box/) → download from statlearning.com
    into the cache. Raises urllib.error.URLError when offline on first use.
    """
    csv = _ADVERTISING_FIXTURE
    if not csv.exists():
        csv = _cache_dir() / "advertising.csv"
        if not csv.exists():
            csv.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_ADVERTISING_URL, csv)  # noqa: S310 - fixed https URL

    with open(csv) as f:
        header = f.readline().strip()
    cols = [c.strip(' "').lower() for c in header.split(",")]
    feature_cols = [cols.index("tv"), cols.index("radio"), cols.index("newspaper")]
    target_col = cols.index("sales")

    data = np.genfromtxt(csv, delimiter=",", skip_header=1)
    return data[:, feature_cols], data[:, target_col], "regression"


LOADERS = {
    "synthetic": synthetic,
    "california": california,
    "advertising": advertising,
}
