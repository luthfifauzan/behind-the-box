# Linear Regression — Closed-Form Solution

## Motivation

Linear regression is the foundation of supervised machine learning. Before neural networks, before ensembles, nearly every practitioner's first instinct is to ask: "can I model this as a linear relationship?"

Beyond its utility, it is worth studying carefully because it has something rare in machine learning: an *exact* algebraic solution. This means we can understand the model completely — not just run it. Its failure modes also directly motivate the techniques that follow: iterative optimisation, regularisation, and probabilistic modelling.

Some familiarity with linear algebra (matrix multiplication, transpose) and calculus (partial derivatives) is helpful but not required — each step below is worked through explicitly.

---

## Derivation

**Setup.** Given $m$ training examples with $n$ features, we have $X \in \mathbb{R}^{m \times n}$ and targets $y \in \mathbb{R}^m$. We prepend a column of ones to $X$ to absorb the intercept, giving $\tilde{X} \in \mathbb{R}^{m \times (n+1)}$ and parameters $\theta \in \mathbb{R}^{n+1}$.

**Loss.** We want $\theta$ that minimises the mean squared error between predictions $\tilde{X}\theta$ and targets $y$:

$$\mathcal{L}(\theta) = \frac{1}{m}\|\tilde{X}\theta - y\|^2 = \frac{1}{m}(\tilde{X}\theta - y)^\top(\tilde{X}\theta - y)$$

**Finding the minimum.** $\mathcal{L}$ is convex and differentiable, so the unique minimum is where the gradient vanishes:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{2}{m}\tilde{X}^\top(\tilde{X}\theta - y) = 0$$

$$\tilde{X}^\top \tilde{X}\,\theta = \tilde{X}^\top y$$

$$\boxed{\theta^* = (\tilde{X}^\top \tilde{X})^{-1}\tilde{X}^\top y}$$

This is the **Normal Equation** — a single matrix solve that yields the exact global minimum.

[`fit`](./01-closed-form-regression.py) implements this directly: it prepends the bias column to form $\tilde{X}$, then calls `np.linalg.lstsq` to solve the system. `lstsq` is preferred over `np.linalg.inv` because it uses an SVD-based solver that stays numerically stable when features are collinear — cases where $\tilde{X}^\top \tilde{X}$ is near-singular and direct inversion would amplify rounding errors. After `fit`, the model exposes `coef_` (shape `(n,)`) and `intercept_` (scalar).

**Prediction** is the linear map $\hat{y} = X\theta + b$. [`predict`](./01-closed-form-regression.py) computes exactly this.

**Goodness of fit.** A natural measure is R², the fraction of variance in $y$ explained by the model:

$$R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$$

$R^2 = 1$ means a perfect fit; $R^2 = 0$ means the model does no better than predicting the mean. [`score`](./01-closed-form-regression.py) returns this value.

---

## Limitations

The Normal Equation works beautifully for small datasets. Two problems emerge as scale increases.

**Cost.** Forming $\tilde{X}^\top \tilde{X}$ and factorising it costs $O(mn^2 + n^3)$ time and $O(n^2)$ memory. For $n > 10{,}000$ features this becomes prohibitive — you are factorising a matrix with $10^8$ entries.

**No path to generalisation.** Ridge regression, Lasso, and logistic regression do not share this clean closed form. Each would need its own bespoke derivation. What we want is a general-purpose optimiser we can reuse across all of them.

Both problems are addressed by replacing the matrix solve with an iterative approach. The algorithm behind that — gradient descent — is introduced in [Gradient Descent](../optimization/01-gradient-descent.md). Once you have that, [Gradient Descent Regression](./02-gradient-descent-regression.md) applies it here and shows what we gain and what we give up.
