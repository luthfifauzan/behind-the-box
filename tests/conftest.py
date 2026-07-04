"""Fixtures shared by every algorithm's test suite."""

import pathlib

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing

_Synthetic = tuple[np.ndarray, np.ndarray, np.ndarray, float]

_FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic(rng: np.random.Generator) -> _Synthetic:
    """100-sample, 3-feature dataset with known ground-truth parameters.

    Returns (X, y, true_coef, true_intercept).
    """
    true_coef = np.array([2.0, -1.5, 0.8])
    true_intercept = 5.0
    X = rng.standard_normal((100, 3))
    y = X @ true_coef + true_intercept + rng.standard_normal(100) * 0.1
    return X, y, true_coef, true_intercept


@pytest.fixture(scope="session")
def california_housing() -> tuple[np.ndarray, np.ndarray]:
    """California Housing (X, y). Session-scoped: fetched/parsed once per run."""
    data = fetch_california_housing()
    return data.data, data.target


@pytest.fixture(scope="session")
def advertising() -> tuple[np.ndarray, np.ndarray]:
    """Advertising dataset (X, y) from tests/fixtures/advertising.csv.

    The CSV is gitignored (downloaded from Kaggle); tests that use this
    fixture are skipped when it is absent.
    """
    fixture = _FIXTURES_DIR / "advertising.csv"
    if not fixture.exists():
        pytest.skip("tests/fixtures/advertising.csv not found — download from Kaggle")

    # Detect column positions from the header.
    with open(fixture) as f:
        header = f.readline().strip()
    cols = [c.strip(' "').lower() for c in header.split(",")]
    feature_cols = [cols.index("tv"), cols.index("radio"), cols.index("newspaper")]
    target_col = cols.index("sales")

    data = np.genfromtxt(fixture, delimiter=",", skip_header=1)
    return data[:, feature_cols], data[:, target_col]
