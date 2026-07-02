"""Tests for closed-form (OLS) linear regression.

Five categories:
  - API contract     : interface shape and return types
  - Math properties  : invariants the algorithm must satisfy
  - Oracle           : numerical match against scikit-learn
  - Edge cases       : corner inputs the implementation must survive
  - Real datasets    : california_housing (inline) + advertising.csv (fixture)
"""

import os

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression as SklearnLR


# ── import under test ─────────────────────────────────────────────────────────
# conftest.py registers "linear_models.closed_form_regression" at collection
# time, so this import will work once the file exists.
from linear_models.closed_form_regression import LinearRegression

_Synthetic = tuple[np.ndarray, np.ndarray, np.ndarray, float]


# ── shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic(
    rng: np.random.Generator,
) -> _Synthetic:
    """100-sample, 3-feature dataset with known ground-truth parameters."""
    true_coef = np.array([2.0, -1.5, 0.8])
    true_intercept = 5.0
    X = rng.standard_normal((100, 3))
    y = X @ true_coef + true_intercept + rng.standard_normal(100) * 0.1
    return X, y, true_coef, true_intercept


# ── api contract ──────────────────────────────────────────────────────────────


class TestAPIContract:
    """Verify the public interface contract of LinearRegression."""

    def test_fit_returns_self(self, synthetic: _Synthetic) -> None:
        """fit() must return self to allow method chaining."""
        X, y, *_ = synthetic
        model = LinearRegression()
        assert model.fit(X, y) is model

    def test_coef_not_set_before_fit(self) -> None:
        """coef_ must not be accessible before fit() is called."""
        with pytest.raises(AttributeError):
            _ = LinearRegression().coef_

    def test_intercept_not_set_before_fit(self) -> None:
        """intercept_ must not be accessible before fit() is called."""
        with pytest.raises(AttributeError):
            _ = LinearRegression().intercept_

    def test_coef_shape(self, synthetic: _Synthetic) -> None:
        """coef_ must be a 1-D array with one entry per feature."""
        X, y, *_ = synthetic
        model = LinearRegression().fit(X, y)
        assert model.coef_.shape == (X.shape[1],)

    def test_intercept_is_scalar(self, synthetic: _Synthetic) -> None:
        """intercept_ must be a 0-D scalar value."""
        X, y, *_ = synthetic
        model = LinearRegression().fit(X, y)
        assert np.ndim(model.intercept_) == 0

    def test_predict_output_shape(
        self, synthetic: _Synthetic, rng: np.random.Generator,
    ) -> None:
        """predict() must return a 1-D array with one entry per sample."""
        X, y, *_ = synthetic
        X_test = rng.standard_normal((25, X.shape[1]))
        model = LinearRegression().fit(X, y)
        assert model.predict(X_test).shape == (25,)

    def test_score_returns_float(self, synthetic: _Synthetic) -> None:
        """score() must return a Python float."""
        X, y, *_ = synthetic
        model = LinearRegression().fit(X, y)
        assert isinstance(model.score(X, y), float)


# ── mathematical properties ───────────────────────────────────────────────────


class TestMathematicalProperties:
    """Verify OLS mathematical invariants."""

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


# ── oracle comparison (vs scikit-learn) ───────────────────────────────────────


class TestOracleComparison:
    """Verify numerical agreement with scikit-learn LinearRegression."""

    def test_coef_matches_sklearn(self, synthetic: _Synthetic) -> None:
        """coef_ must match scikit-learn to within rtol=1e-6."""
        X, y, *_ = synthetic
        ref = SklearnLR().fit(X, y)
        model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.coef_, ref.coef_, rtol=1e-6)

    def test_intercept_matches_sklearn(self, synthetic: _Synthetic) -> None:
        """intercept_ must match scikit-learn to within rtol=1e-6."""
        X, y, *_ = synthetic
        ref = SklearnLR().fit(X, y)
        model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.intercept_, ref.intercept_, rtol=1e-6)

    def test_predictions_match_sklearn(
        self, synthetic: _Synthetic, rng: np.random.Generator,
    ) -> None:
        """predict() must match scikit-learn on held-out data to within rtol=1e-6."""
        X, y, *_ = synthetic
        X_test = rng.standard_normal((30, X.shape[1]))
        ref = SklearnLR().fit(X, y)
        model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.predict(X_test), ref.predict(X_test), rtol=1e-6)

    def test_score_matches_sklearn(self, synthetic: _Synthetic) -> None:
        """score() R² must match scikit-learn to within rel=1e-6."""
        X, y, *_ = synthetic
        ref = SklearnLR().fit(X, y)
        model = LinearRegression().fit(X, y)
        assert model.score(X, y) == pytest.approx(ref.score(X, y), rel=1e-6)

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_sklearn_across_random_seeds(self, seed: int) -> None:
        """coef_ and intercept_ must match scikit-learn across multiple random seeds."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((200, 8))
        y = rng.standard_normal(200)
        ref = SklearnLR().fit(X, y)
        model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.coef_, ref.coef_, rtol=1e-5)
        np.testing.assert_allclose(model.intercept_, ref.intercept_, rtol=1e-5)


# ── edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Verify correct behaviour on corner-case inputs."""

    def test_single_feature(self, rng: np.random.Generator) -> None:
        """Single-feature regression must recover slope and intercept accurately."""
        X = rng.standard_normal((80, 1))
        y = 3.0 * X.ravel() + 1.0 + rng.standard_normal(80) * 0.01
        model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.coef_[0], 3.0, atol=0.05)
        np.testing.assert_allclose(model.intercept_, 1.0, atol=0.05)

    def test_many_features(self, rng: np.random.Generator) -> None:
        """High-dimensional regression must match scikit-learn to within rtol=1e-5."""
        X = rng.standard_normal((500, 50))
        true_coef = rng.standard_normal(50)
        y = X @ true_coef + rng.standard_normal(500) * 0.1
        model = LinearRegression().fit(X, y)
        ref = SklearnLR().fit(X, y)
        np.testing.assert_allclose(model.coef_, ref.coef_, rtol=1e-5)

    def test_collinear_features_no_nan(self, rng: np.random.Generator) -> None:
        """lstsq must not raise and must return finite predictions under collinearity."""
        X = rng.standard_normal((100, 3))
        X[:, 2] = X[:, 0] + X[:, 1]  # exact linear dependence
        y = X[:, 0] + rng.standard_normal(100) * 0.1
        model = LinearRegression().fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))

    def test_two_samples_one_feature(self) -> None:
        """Minimum viable dataset: 2 points determine a line exactly."""
        X = np.array([[0.0], [1.0]])
        y = np.array([1.0, 3.0])
        model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.predict(X), y, atol=1e-10)

    def test_predictions_on_unseen_data_are_finite(
        self, synthetic: _Synthetic, rng: np.random.Generator,
    ) -> None:
        """Predictions on out-of-distribution data must all be finite."""
        X, y, *_ = synthetic
        model = LinearRegression().fit(X, y)
        X_new = rng.standard_normal((50, X.shape[1])) * 10
        assert np.all(np.isfinite(model.predict(X_new)))


# ── real datasets ─────────────────────────────────────────────────────────────


class TestRealDatasets:
    """Verify performance and correctness on real-world datasets."""

    def test_california_housing_r2(self) -> None:
        """R² on California Housing must exceed 0.58 and match scikit-learn."""
        data = fetch_california_housing()
        X, y = data.data, data.target
        model = LinearRegression().fit(X, y)
        ref = SklearnLR().fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.58, f"R² on California Housing too low: {r2:.4f}"
        np.testing.assert_allclose(r2, ref.score(X, y), atol=0.01)

    def test_california_housing_predictions_are_finite(self) -> None:
        """Predictions on California Housing must all be finite."""
        data = fetch_california_housing()
        X, y = data.data, data.target
        model = LinearRegression().fit(X, y)
        assert np.all(np.isfinite(model.predict(X)))

    def test_advertising_csv(self) -> None:
        """R² on Advertising must exceed 0.89 and match scikit-learn."""
        fixture = os.path.join(
            os.path.dirname(__file__), "..", "fixtures", "advertising.csv"
        )
        if not os.path.exists(fixture):
            pytest.skip("tests/fixtures/advertising.csv not found — download from Kaggle")

        # Detect column positions from header
        with open(fixture) as f:
            header = f.readline().strip()
        cols = [c.strip(' "').lower() for c in header.split(",")]
        feature_cols = [cols.index("tv"), cols.index("radio"), cols.index("newspaper")]
        target_col = cols.index("sales")

        data = np.genfromtxt(fixture, delimiter=",", skip_header=1)
        X = data[:, feature_cols]
        y = data[:, target_col]

        model = LinearRegression().fit(X, y)
        ref = SklearnLR().fit(X, y)
        r2 = model.score(X, y)

        assert r2 > 0.89, f"R² on Advertising too low: {r2:.4f}"
        np.testing.assert_allclose(r2, ref.score(X, y), atol=0.01)
        assert np.all(np.isfinite(model.predict(X)))
