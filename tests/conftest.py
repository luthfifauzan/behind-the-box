"""Fixtures shared by every algorithm's test suite.

Real datasets come from behind_the_box.datasets (the single source of
truth); the fixtures here only add pytest concerns — session caching and
skip-when-offline.
"""

from urllib.error import URLError

import numpy as np
import pytest

from behind_the_box import datasets

_Synthetic = tuple[np.ndarray, np.ndarray, np.ndarray, float]


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic(rng: np.random.Generator) -> _Synthetic:
    """100-sample, 3-feature dataset with known ground-truth parameters.

    Returns (X, y, true_coef, true_intercept) — unlike datasets.synthetic,
    the truth is returned so tests can assert parameter recovery.
    """
    true_coef = np.array([2.0, -1.5, 0.8])
    true_intercept = 5.0
    X = rng.standard_normal((100, 3))
    y = X @ true_coef + true_intercept + rng.standard_normal(100) * 0.1
    return X, y, true_coef, true_intercept


@pytest.fixture(scope="session")
def california_housing() -> tuple[np.ndarray, np.ndarray]:
    """California Housing (X, y). Session-scoped: fetched/parsed once per run."""
    X, y, _ = datasets.california()
    return X, y


@pytest.fixture(scope="session")
def advertising() -> tuple[np.ndarray, np.ndarray]:
    """Advertising dataset (X, y); auto-downloaded and cached on first use."""
    try:
        X, y, _ = datasets.advertising()
    except URLError:
        pytest.skip("offline and advertising.csv not cached")
    return X, y
