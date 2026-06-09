import numpy as np


class LinearRegression:
    """Ordinary Least Squares via the normal equations.

    Solves theta = (X'X)^{-1} X'y using lstsq for numerical stability.
    See 01-closed-form-regression.md for the derivation.
    """

    coef_: np.ndarray
    intercept_: float

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X_b = np.column_stack([np.ones(len(X)), X])
        theta, *_ = np.linalg.lstsq(X_b, y, rcond=None)
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)
