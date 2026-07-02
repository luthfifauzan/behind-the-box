# Linear Regression — Closed-Form Solution

## Motivation

Linear regression is the foundation of supervised machine learning. Before neural networks, before ensembles, nearly every practitioner's first instinct is to ask: "can I model this as a linear relationship?"

Beyond its utility, it is worth studying carefully because it has something rare in machine learning: an *exact* algebraic solution. This means we can understand the model completely — not just run it. Its failure modes also directly motivate the techniques that follow: iterative optimisation, regularisation, and probabilistic modelling.

Some familiarity with linear algebra (matrix multiplication, transpose) and calculus (partial derivatives) is helpful but not required — each step below is worked through explicitly.

---

## Derivation

### Step 1 — What does the model look like?

We have $m$ training examples, each with $n$ features. Stack them into a matrix $X \in \mathbb{R}^{m \times n}$ (one row per example) and a vector of targets $y \in \mathbb{R}^m$.

Our model says: *prediction = weighted sum of features + a constant bias*. In vector form for a single example $x$:

$$\hat{y} = \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n + \theta_0$$

The bias $\theta_0$ is awkward to carry separately. The standard trick is to prepend a column of ones to $X$, so the bias becomes just another weight. Call the augmented matrix $\tilde{X} \in \mathbb{R}^{m \times (n+1)}$ and the full parameter vector $\theta \in \mathbb{R}^{n+1}$. Now all $m$ predictions at once are simply:

$$\hat{y} = \tilde{X}\theta$$

### Step 2 — What does "best fit" mean?

We need a number that measures how wrong our predictions are. The standard choice is **mean squared error (MSE)**: average the squared difference between each prediction and the true target.

$$\mathcal{L}(\theta) = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2 = \frac{1}{m}\|\tilde{X}\theta - y\|^2$$

Why square the errors rather than take the absolute value? Two reasons. First, squaring penalises large mistakes disproportionately — a prediction that is off by 10 hurts 100× as much as one off by 1, which is usually what we want. Second, the squared function is smooth and differentiable everywhere, which lets us find the minimum analytically.

If you plot $\mathcal{L}$ as a function of $\theta$, it looks like a bowl: it curves upward in every direction. That shape is called **convex**, and it guarantees one thing: there is exactly one minimum, and it sits at the very bottom of the bowl.

### Step 3 — Finding the bottom of the bowl

At the bottom of any smooth bowl, the slope is zero in every direction. So we differentiate $\mathcal{L}$ with respect to $\theta$ and set the result to zero.

Taking the gradient (the multi-dimensional analogue of a derivative):

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{2}{m}\tilde{X}^\top(\tilde{X}\theta - y) = 0$$

The $\frac{2}{m}$ is just a constant — it cannot force the expression to zero. What must be zero is the rest:

$$\tilde{X}^\top(\tilde{X}\theta - y) = 0$$

**Geometric meaning.** $\tilde{X}\theta - y$ is the vector of residuals — how far each prediction is from the truth. The equation says $\tilde{X}^\top$ times that vector equals zero, meaning the residuals are *orthogonal* (perpendicular) to every column of $\tilde{X}$. In other words, our predictions $\tilde{X}\theta$ are the closest point in the "reachable space" of the model to the true $y$. This is exactly a projection.

### Step 4 — Solving for $\theta$

Rearranging the equation from step 3:

$$\tilde{X}^\top \tilde{X}\,\theta = \tilde{X}^\top y$$

This is a standard linear system $A\theta = b$ where $A = \tilde{X}^\top \tilde{X}$ and $b = \tilde{X}^\top y$. If $A$ is invertible we can write the explicit solution:

$$\boxed{\theta^* = (\tilde{X}^\top \tilde{X})^{-1}\tilde{X}^\top y}$$

This is the **Normal Equation**. The name "normal" refers to the geometric fact from step 3: the residuals are *normal* (perpendicular) to the feature space.

In practice we never compute $(\tilde{X}^\top \tilde{X})^{-1}$ explicitly — matrix inversion is numerically brittle. Instead, `np.linalg.lstsq` solves the same system more stably using an SVD decomposition under the hood.

`fit` implements this: prepend the ones column to form $\tilde{X}$, call `lstsq`, then split $\theta^*$ into `intercept_` (the first element) and `coef_` (the rest).

### Step 5 — Making predictions

Once we have $\theta^*$, prediction on new data is just the linear map:

$$\hat{y} = X\hat\theta_{1:} + \hat\theta_0$$

where $\hat\theta_{1:}$ is stored as `coef_` and $\hat\theta_0$ as `intercept_`. `predict` computes exactly this.

### Step 6 — How good is the fit?

MSE tells us the absolute error but not whether the model is *useful*. A baseline that always predicts the mean of $y$ has some MSE of its own; a good model should beat it.

$R^2$ measures this relative improvement:

$$R^2 = 1 - \frac{\underbrace{\sum_i(y_i - \hat{y}_i)^2}_{\text{model's error}}}{\underbrace{\sum_i(y_i - \bar{y})^2}_{\text{baseline's error}}}$$

- $R^2 = 1$: your predictions are perfect.
- $R^2 = 0$: your model is no better than predicting the mean every time.
- $R^2 < 0$: your model is *worse* than that baseline (possible on test data).

`score` returns this value.

---

## Implementation API

```python
class LinearRegression:
    coef_: np.ndarray       # shape (n_features,), set after fit
    intercept_: float       # scalar, set after fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...
```

**Numerical note.** Prefer `np.linalg.lstsq` over computing $(X^\top X)^{-1}$ explicitly. `lstsq` uses an SVD-based solver that stays numerically stable when features are collinear — cases where $\tilde{X}^\top \tilde{X}$ is near-singular and direct inversion would amplify rounding errors.

---

## Limitations

The Normal Equation works beautifully for small datasets. Two problems emerge as scale increases.

**Cost.** Forming $\tilde{X}^\top \tilde{X}$ and factorising it costs $O(mn^2 + n^3)$ time and $O(n^2)$ memory. For $n > 10{,}000$ features this becomes prohibitive.

**No path to generalisation.** Ridge regression, Lasso, and logistic regression do not share this clean closed form. What we want is a general-purpose optimiser we can reuse across all of them — that is gradient descent, covered next in [Gradient Descent Regression](./02-gradient-descent-regression.md).
