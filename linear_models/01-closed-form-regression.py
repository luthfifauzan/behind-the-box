"""Closed-form (OLS) linear regression via the Normal Equation."""

import numpy as np


class LinearRegression:
    """Ordinary Least Squares regression solved via the Normal Equation.

    The model predicts a continuous target as a weighted sum of input features
    plus a scalar bias (intercept):

        ŷ = θ₁x₁ + θ₂x₂ + … + θₙxₙ + θ₀

    Fitting finds the parameter vector θ* that minimises mean squared error
    between predictions and targets. See 01-closed-form-regression.md for the
    full derivation.

    Attributes set after fit():
        coef_ (np.ndarray): Shape (n_features,). One weight per input feature.
        intercept_ (float): Scalar bias term θ₀.
    """

    def __init__(self) -> None:
        """Some docstring to put here."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Solve for the optimal parameters using the Normal Equation.

        The Normal Equation (equation 8 in the .md) gives the exact closed-form
        solution:

            θ* = (X̃ᵀX̃)⁻¹ X̃ᵀy

        where X̃ is X with a prepended column of ones that absorbs the intercept
        (turning θ₀ into just another weight).

        Implementation outline:
            1. Build X̃ by prepending a column of ones to X.
            2. Solve the linear system X̃ᵀX̃ θ = X̃ᵀy for θ.
               Use np.linalg.lstsq rather than inverting X̃ᵀX̃ explicitly —
               lstsq is numerically stable even when features are collinear.
            3. Split θ: the first element becomes intercept_, the rest coef_.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).

        Returns:
            self — allows method chaining, e.g. model.fit(X, y).predict(X).

        Sets:
            self.coef_:      np.ndarray of shape (n_features,).
            self.intercept_: float scalar.
        """

        X_prepended = np.column_stack((np.ones(X.shape[0]), X))
        theta = np.linalg.lstsq(X_prepended, y)[0]

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute predicted target values for X.

        Applies the linear map from equation (2) in the .md:

            ŷ = X @ coef_ + intercept_

        Each row of X is one sample; the output is one scalar prediction per row.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
               Must have the same number of features as the training data.

        Returns:
            np.ndarray of shape (n_samples,) containing predicted values.
        """

        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the R² coefficient of determination (equation 9 in the .md).

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

        model_error = sum((y - y_pred) ** 2)
        baseline_error = sum((y - np.mean(y)) ** 2)

        return 1 - (model_error / baseline_error)


if __name__ == "__main__":
    X = np.array([[1, 4], [2, 6], [3, 4], [1, 6]])  # following the example
    y = np.array([40, 60, 60, 50])

    lr = LinearRegression().fit(X, y)

    y_pred = lr.predict(X)
    r2_score = lr.score(X, y)
