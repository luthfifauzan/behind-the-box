"""Tests for closed-form (OLS) linear regression.

Generic regressor behaviour (API contract, sklearn oracle, edge cases,
real datasets) comes from the shared suites in regressor_checks.py —
configured here with tight tolerances, since the Normal Equation is an
exact solver. Only the OLS-specific mathematical invariants are defined
in this file.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLR

from behind_the_box.linear_models import LinearRegression
from tests.linear_models.regressor_checks import (
    RegressorAPIContract,
    RegressorEdgeCases,
    RegressorOracleChecks,
    RegressorRealDataChecks,
)

_Synthetic = tuple[np.ndarray, np.ndarray, np.ndarray, float]


class _Config:
    """Shared configuration: exact solver → tight tolerances, no scaling."""

    model_factory = staticmethod(LinearRegression)
    sklearn_factory = staticmethod(SklearnLR)
    rtol = 1e-6
    scale_features = False
    exact_interpolation = True


class TestAPIContract(_Config, RegressorAPIContract):
    """Public interface contract of LinearRegression."""


class TestOracleComparison(_Config, RegressorOracleChecks):
    """Numerical agreement with scikit-learn LinearRegression."""


class TestEdgeCases(_Config, RegressorEdgeCases):
    """Corner-case inputs."""


class TestRealDatasets(_Config, RegressorRealDataChecks):
    """California Housing + Advertising performance."""


# ── OLS-specific mathematical properties ──────────────────────────────────────


class TestMathematicalProperties:
    """Invariants specific to the exact OLS solution."""

    def test_residuals_orthogonal_to_feature_space(self, synthetic: _Synthetic) -> None:
        """OLS residuals must lie in the null space of X̃ᵀ (normal equations)."""
        X, y, *_ = synthetic
        model = LinearRegression().fit(X, y)
        X_b = np.column_stack([np.ones(len(X)), X])
        residuals = y - model.predict(X)
        np.testing.assert_allclose(X_b.T @ residuals, 0.0, atol=1e-8)

    def test_noiseless_data_gives_perfect_r2(self, rng: np.random.Generator) -> None:
        """With no noise and n_samples > n_features, R² must equal 1."""
        X = rng.standard_normal((60, 4))
        y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + 7.0
        r2 = LinearRegression().fit(X, y).score(X, y)
        np.testing.assert_allclose(r2, 1.0, atol=1e-10)

    def test_recovers_true_coefficients(self, synthetic: _Synthetic) -> None:
        """Low-noise data: fitted parameters must be close to the truth."""
        X, y, true_coef, true_intercept = synthetic
        model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.coef_, true_coef, atol=0.05)
        np.testing.assert_allclose(model.intercept_, true_intercept, atol=0.05)

    def test_score_on_training_predictions_is_one(self, synthetic: _Synthetic) -> None:
        """score(X, predict(X)) must always return 1.0."""
        X, y, *_ = synthetic
        model = LinearRegression().fit(X, y)
        assert model.score(X, model.predict(X)) == pytest.approx(1.0)

    def test_linear_part_is_additive(
        self, synthetic: _Synthetic, rng: np.random.Generator,
    ) -> None:
        """(predict(X1+X2) - b) == (predict(X1)-b) + (predict(X2)-b)."""
        X, y, *_ = synthetic
        model = LinearRegression().fit(X, y)
        X1 = rng.standard_normal((10, X.shape[1]))
        X2 = rng.standard_normal((10, X.shape[1]))
        b = model.intercept_
        lhs = model.predict(X1 + X2) - b
        rhs = (model.predict(X1) - b) + (model.predict(X2) - b)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    def test_zero_feature_matrix_predicts_mean(self, rng: np.random.Generator) -> None:
        """When all features are zero, the only signal is the intercept = mean(y)."""
        y = rng.standard_normal(100)
        X = np.zeros((100, 3))
        model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.intercept_, np.mean(y), atol=1e-8)
        np.testing.assert_allclose(model.coef_, np.zeros(3), atol=1e-8)
