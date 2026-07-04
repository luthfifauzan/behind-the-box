"""Shared behaviour mixed into every model in this package.

Analogue of ``sklearn.base``. Only behaviour that is *identical* across
models lives here — anything algorithm-specific stays in the model's own
file so each Explanation/Implementation pair remains readable on its own.

Current contents: ``RegressorMixin.score`` (R²), which every regressor
computes the exact same way regardless of how it was fitted.
"""

import numpy as np


class RegressorMixin:
    """Mixin providing ``score()`` for any model with a ``predict()`` method.

    A regressor gains R² scoring simply by inheriting from this class:

        class SomeRegressor(RegressorMixin):
            def fit(self, X, y): ...
            def predict(self, X): ...
            # score() arrives via inheritance — no code needed

    The mixin has no state of its own; it only calls ``self.predict``.
    """

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the R² coefficient of determination.

        R² measures what fraction of the variance in y is explained by the model:

            R² = 1 − SS_res / SS_tot

        where:
            SS_res = Σ (yᵢ − ŷᵢ)²   (sum of squared residuals — model's error)
            SS_tot = Σ (yᵢ − ȳ)²    (total variance — baseline error)
            ȳ      = mean(y)

        Interpretation:
            R² = 1.0  →  perfect predictions.
            R² = 0.0  →  no better than always predicting the mean.
            R² < 0.0  →  worse than the mean baseline (possible on test data).

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True target values of shape (n_samples,).

        Returns:
            R² as a Python float.
        """
        y_pred = self.predict(X)

        model_error = np.sum((y - y_pred) ** 2)
        baseline_error = np.sum((y - np.mean(y)) ** 2)

        return 1 - (model_error / baseline_error)
