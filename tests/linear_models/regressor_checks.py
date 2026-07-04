"""Reusable check suites that any regressor's test file can subclass.

Four suites, mirroring the repo's test categories (API contract, oracle
comparison, edge cases, real datasets). Model-specific mathematical
properties do NOT belong here — they live in each model's own test file.

Usage — subclass in ``test_NN_<name>.py`` and set class attributes:

    class TestAPIContract(RegressorAPIContract):
        model_factory = staticmethod(LinearRegression)

    class TestOracle(RegressorOracleChecks):
        model_factory = staticmethod(LinearRegression)
        sklearn_factory = staticmethod(SklearnLR)
        rtol = 1e-6          # loosen for iterative solvers

Configuration attributes:
    model_factory    — zero-arg callable returning an unfitted model
    sklearn_factory  — zero-arg callable returning the sklearn oracle
    rtol             — relative tolerance vs the oracle (exact solvers: 1e-6)
    scale_features   — standardize X before fitting (iterative solvers need
                       well-conditioned inputs; both model and oracle see the
                       same scaled data, so comparisons stay valid)
    exact_interpolation — whether n_samples == n_params data is fitted
                       exactly (true for closed-form solvers; iterative
                       solvers stopped early may not interpolate)
    min_r2_california / min_r2_advertising — real-dataset R² floors

This file has no ``test_`` prefix, so pytest only collects these checks
through the subclasses that configure them.
"""

from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest

_Synthetic = tuple[np.ndarray, np.ndarray, np.ndarray, float]


class _RegressorConfig:
    """Class attributes that parametrize the check suites below."""

    model_factory: ClassVar[Callable]
    sklearn_factory: ClassVar[Callable]
    rtol: ClassVar[float] = 1e-6
    scale_features: ClassVar[bool] = False
    exact_interpolation: ClassVar[bool] = True
    min_r2_california: ClassVar[float] = 0.58
    min_r2_advertising: ClassVar[float] = 0.89

    @classmethod
    def _scale(
        cls, X_train: np.ndarray, *X_others: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Standardize using train statistics (no-op unless scale_features)."""
        if not cls.scale_features:
            return (X_train, *X_others)
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return tuple((X - mean) / std for X in (X_train, *X_others))


# ── api contract ──────────────────────────────────────────────────────────────


class RegressorAPIContract(_RegressorConfig):
    """The public interface contract every regressor must satisfy."""

    def test_fit_returns_self(self, synthetic: _Synthetic) -> None:
        """fit() must return self to allow method chaining."""
        X, y, *_ = synthetic
        model = self.model_factory()
        assert model.fit(X, y) is model

    def test_coef_not_set_before_fit(self) -> None:
        """coef_ must not be accessible before fit() is called."""
        with pytest.raises(AttributeError):
            _ = self.model_factory().coef_

    def test_intercept_not_set_before_fit(self) -> None:
        """intercept_ must not be accessible before fit() is called."""
        with pytest.raises(AttributeError):
            _ = self.model_factory().intercept_

    def test_coef_shape(self, synthetic: _Synthetic) -> None:
        """coef_ must be a 1-D array with one entry per feature."""
        X, y, *_ = synthetic
        model = self.model_factory().fit(X, y)
        assert model.coef_.shape == (X.shape[1],)

    def test_intercept_is_scalar(self, synthetic: _Synthetic) -> None:
        """intercept_ must be a 0-D scalar value."""
        X, y, *_ = synthetic
        model = self.model_factory().fit(X, y)
        assert np.ndim(model.intercept_) == 0

    def test_predict_output_shape(
        self, synthetic: _Synthetic, rng: np.random.Generator,
    ) -> None:
        """predict() must return a 1-D array with one entry per sample."""
        X, y, *_ = synthetic
        X_test = rng.standard_normal((25, X.shape[1]))
        model = self.model_factory().fit(X, y)
        assert model.predict(X_test).shape == (25,)

    def test_score_returns_float(self, synthetic: _Synthetic) -> None:
        """score() must return a Python float."""
        X, y, *_ = synthetic
        model = self.model_factory().fit(X, y)
        assert isinstance(model.score(X, y), float)


# ── oracle comparison (vs scikit-learn) ───────────────────────────────────────


class RegressorOracleChecks(_RegressorConfig):
    """Numerical agreement with the scikit-learn counterpart."""

    def test_coef_matches_sklearn(self, synthetic: _Synthetic) -> None:
        """coef_ must match scikit-learn to within rtol."""
        X, y, *_ = synthetic
        (X,) = self._scale(X)
        ref = self.sklearn_factory().fit(X, y)
        model = self.model_factory().fit(X, y)
        np.testing.assert_allclose(model.coef_, ref.coef_, rtol=self.rtol)

    def test_intercept_matches_sklearn(self, synthetic: _Synthetic) -> None:
        """intercept_ must match scikit-learn to within rtol."""
        X, y, *_ = synthetic
        (X,) = self._scale(X)
        ref = self.sklearn_factory().fit(X, y)
        model = self.model_factory().fit(X, y)
        np.testing.assert_allclose(model.intercept_, ref.intercept_, rtol=self.rtol)

    def test_predictions_match_sklearn(
        self, synthetic: _Synthetic, rng: np.random.Generator,
    ) -> None:
        """predict() must match scikit-learn on held-out data to within rtol."""
        X, y, *_ = synthetic
        X_test = rng.standard_normal((30, X.shape[1]))
        X, X_test = self._scale(X, X_test)
        ref = self.sklearn_factory().fit(X, y)
        model = self.model_factory().fit(X, y)
        np.testing.assert_allclose(
            model.predict(X_test), ref.predict(X_test), rtol=self.rtol,
        )

    def test_score_matches_sklearn(self, synthetic: _Synthetic) -> None:
        """score() R² must match scikit-learn to within rtol."""
        X, y, *_ = synthetic
        (X,) = self._scale(X)
        ref = self.sklearn_factory().fit(X, y)
        model = self.model_factory().fit(X, y)
        assert model.score(X, y) == pytest.approx(ref.score(X, y), rel=self.rtol)

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_sklearn_across_random_seeds(self, seed: int) -> None:
        """coef_ and intercept_ must match scikit-learn across random seeds."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((200, 8))
        y = rng.standard_normal(200)
        (X,) = self._scale(X)
        ref = self.sklearn_factory().fit(X, y)
        model = self.model_factory().fit(X, y)
        np.testing.assert_allclose(model.coef_, ref.coef_, rtol=self.rtol * 10)
        np.testing.assert_allclose(
            model.intercept_, ref.intercept_, rtol=self.rtol * 10,
        )


# ── edge cases ────────────────────────────────────────────────────────────────


class RegressorEdgeCases(_RegressorConfig):
    """Correct behaviour on corner-case inputs."""

    def test_single_feature(self, rng: np.random.Generator) -> None:
        """Single-feature regression must recover slope and intercept accurately."""
        X = rng.standard_normal((80, 1))
        y = 3.0 * X.ravel() + 1.0 + rng.standard_normal(80) * 0.01
        model = self.model_factory().fit(X, y)
        np.testing.assert_allclose(model.coef_[0], 3.0, atol=0.05)
        np.testing.assert_allclose(model.intercept_, 1.0, atol=0.05)

    def test_many_features(self, rng: np.random.Generator) -> None:
        """High-dimensional regression must match scikit-learn to within rtol."""
        X = rng.standard_normal((500, 50))
        true_coef = rng.standard_normal(50)
        y = X @ true_coef + rng.standard_normal(500) * 0.1
        (X,) = self._scale(X)
        model = self.model_factory().fit(X, y)
        ref = self.sklearn_factory().fit(X, y)
        np.testing.assert_allclose(model.coef_, ref.coef_, rtol=self.rtol * 10)

    def test_collinear_features_no_nan(self, rng: np.random.Generator) -> None:
        """Under exact collinearity, must not raise and predictions must be finite."""
        X = rng.standard_normal((100, 3))
        X[:, 2] = X[:, 0] + X[:, 1]  # exact linear dependence
        y = X[:, 0] + rng.standard_normal(100) * 0.1
        model = self.model_factory().fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))

    def test_two_samples_one_feature(self) -> None:
        """Minimum viable dataset: 2 points determine a line exactly."""
        if not self.exact_interpolation:
            pytest.skip("solver does not guarantee exact interpolation")
        X = np.array([[0.0], [1.0]])
        y = np.array([1.0, 3.0])
        model = self.model_factory().fit(X, y)
        np.testing.assert_allclose(model.predict(X), y, atol=1e-10)

    def test_predictions_on_unseen_data_are_finite(
        self, synthetic: _Synthetic, rng: np.random.Generator,
    ) -> None:
        """Predictions on out-of-distribution data must all be finite."""
        X, y, *_ = synthetic
        model = self.model_factory().fit(X, y)
        X_new = rng.standard_normal((50, X.shape[1])) * 10
        assert np.all(np.isfinite(model.predict(X_new)))


# ── real datasets ─────────────────────────────────────────────────────────────


class RegressorRealDataChecks(_RegressorConfig):
    """Performance and correctness on real-world datasets."""

    @pytest.mark.network
    def test_california_housing_r2(
        self, california_housing: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """R² on California Housing must exceed the floor and match scikit-learn."""
        X, y = california_housing
        (X,) = self._scale(X)
        model = self.model_factory().fit(X, y)
        ref = self.sklearn_factory().fit(X, y)
        r2 = model.score(X, y)
        assert r2 > self.min_r2_california, f"R² too low: {r2:.4f}"
        np.testing.assert_allclose(r2, ref.score(X, y), atol=0.01)

    @pytest.mark.network
    def test_california_housing_predictions_are_finite(
        self, california_housing: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Predictions on California Housing must all be finite."""
        X, y = california_housing
        (X,) = self._scale(X)
        model = self.model_factory().fit(X, y)
        assert np.all(np.isfinite(model.predict(X)))

    @pytest.mark.network
    def test_advertising_csv(
        self, advertising: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """R² on Advertising must exceed the floor and match scikit-learn."""
        X, y = advertising
        (X,) = self._scale(X)
        model = self.model_factory().fit(X, y)
        ref = self.sklearn_factory().fit(X, y)
        r2 = model.score(X, y)
        assert r2 > self.min_r2_advertising, f"R² too low: {r2:.4f}"
        np.testing.assert_allclose(r2, ref.score(X, y), atol=0.01)
        assert np.all(np.isfinite(model.predict(X)))
