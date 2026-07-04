# Linear Regression — Closed-Form Solution

## Motivation

Linear regression is the foundation of supervised machine learning. Before neural networks, before ensembles, nearly every practitioner's first instinct is to ask: "can I model this as a linear relationship?"

Beyond its utility, it is worth studying carefully because it has something rare in machine learning: an *exact* algebraic solution. This means we can understand the model completely — not just run it. Its failure modes also directly motivate the techniques that follow: iterative optimisation, regularisation, and probabilistic modelling.

Some familiarity with linear algebra (matrix multiplication, transpose) and calculus (partial derivatives) is helpful but not required — each step below is worked through explicitly.

---

## Running example

To make every step concrete, we will trace a single small dataset through the entire derivation.

**Problem:** predict a student's exam score from hours studied and hours slept.

| Student | Hours studied ($x_1$) | Hours slept ($x_2$) | Score ($y$) |
|---------|-----------------------|---------------------|-------------|
| A       | 1                     | 4                   | 40          |
| B       | 2                     | 6                   | 60          |
| C       | 3                     | 4                   | 60          |
| D       | 1                     | 6                   | 50          |

For this example the true relationship is $\text{score} = 10 + 10 \cdot x_1 + 5 \cdot x_2$. At the end of the derivation we will verify the algorithm recovers exactly $\theta_0 = 10$, $\theta_1 = 10$, $\theta_2 = 5$. Real data has noise and we would only get an approximation — but noise-free data keeps the arithmetic clean.

---

## Derivation

### Step 1 — What does the model look like?

We have $m$ training examples, each with $n$ features. Stack them into a matrix $X \in \mathbb{R}^{m \times n}$ (one row per example) and a vector of targets $y \in \mathbb{R}^m$.

Our model says: *prediction = weighted sum of features + a constant bias*:

$$\hat{y} = \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n + \theta_0 \tag{1}$$

The bias $\theta_0$ is awkward to carry separately. The standard trick: prepend a column of ones to $X$, so the bias becomes just another weight to be learned. Call the augmented matrix $\tilde{X} \in \mathbb{R}^{m \times (n+1)}$ and the full parameter vector $\theta \in \mathbb{R}^{n+1}$. All $m$ predictions at once are then:

$$\hat{y} = \tilde{X}\theta \tag{2}$$

**In our example.** We have $m = 4$ students, $n = 2$ features.

$$X = \begin{bmatrix} 1 & 4 \\ 2 & 6 \\ 3 & 4 \\ 1 & 6 \end{bmatrix}, \qquad y = \begin{bmatrix} 40 \\ 60 \\ 60 \\ 50 \end{bmatrix}$$

After prepending a ones column:

$$\tilde{X} = \begin{bmatrix} 1 & 1 & 4 \\ 1 & 2 & 6 \\ 1 & 3 & 4 \\ 1 & 1 & 6 \end{bmatrix}, \qquad \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \end{bmatrix}$$

Notice how each row of $\tilde{X}$ starts with a 1 — that is what absorbs the bias. The matrix product $\tilde{X}\theta$ expands to one equation per student:

$$\tilde{X}\theta = \begin{bmatrix} \theta_0 + \theta_1 \cdot 1 + \theta_2 \cdot 4 \\ \theta_0 + \theta_1 \cdot 2 + \theta_2 \cdot 6 \\ \theta_0 + \theta_1 \cdot 3 + \theta_2 \cdot 4 \\ \theta_0 + \theta_1 \cdot 1 + \theta_2 \cdot 6 \end{bmatrix}$$

If we already knew the answer $\theta = [10,\,10,\,5]^\top$, plugging in would give:

$$\tilde{X}\theta = \begin{bmatrix} 10 + 10 + 20 \\ 10 + 20 + 30 \\ 10 + 30 + 20 \\ 10 + 10 + 30 \end{bmatrix} = \begin{bmatrix} 40 \\ 60 \\ 60 \\ 50 \end{bmatrix} = y \checkmark$$

The goal of the algorithm is to find $\theta$ without being told the answer.

### Step 2 — What does "best fit" mean?

We need a number that measures how wrong our predictions are. The standard choice is **mean squared error (MSE)**: average the squared difference between each prediction and the true target.

$$\mathcal{L}(\theta) = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2 = \frac{1}{m}\|\tilde{X}\theta - y\|^2 \tag{3}$$

Why square the errors? Two reasons. First, squaring penalises large mistakes more — an error of 10 hurts 100× as much as an error of 1. Second, the squared function is smooth everywhere, which lets us find the minimum analytically.

If you plot $\mathcal{L}$ as a function of $\theta$, it looks like a bowl: it curves upward in every direction. That shape is called **convex**, and it guarantees exactly one minimum at the very bottom.

**In our example.** Start with a bad guess $\theta = [0,\,0,\,0]^\top$ (predict zero for everyone):

$$\mathcal{L}([0,0,0]) = \frac{(0-40)^2 + (0-60)^2 + (0-60)^2 + (0-50)^2}{4} = \frac{1600+3600+3600+2500}{4} = 2825$$

Now use the correct $\theta = [10,\,10,\,5]^\top$:

$$\mathcal{L}([10,10,5]) = \frac{(40-40)^2 + (60-60)^2 + (60-60)^2 + (50-50)^2}{4} = \frac{0+0+0+0}{4} = 0$$

MSE = 2825 means "very wrong"; MSE = 0 means "perfect". We want to minimise this number.

### Step 3 — Finding the bottom of the bowl

At the bottom of any smooth bowl, the slope is zero in every direction. So we differentiate $\mathcal{L}$ with respect to $\theta$ and set the result to zero.

**Scalar warm-up.** For a single number $\theta$, consider $f(\theta) = (a\theta - b)^2$. By the chain rule:

$$f'(\theta) = 2(a\theta - b) \cdot a = 2a(a\theta - b)$$

Setting $f'(\theta) = 0$ gives $a\theta = b$, so $\theta = b/a$. That is just "solve for $\theta$".

**Matrix version.** Our loss $\mathcal{L}(\theta) = \|\tilde{X}\theta - y\|^2$ is the same idea but $a$ is now the matrix $\tilde{X}$ and $\theta$ is a vector. Applying the chain rule to each component and collecting the results produces a transpose:

$$\nabla_\theta\,\mathcal{L} = 2\tilde{X}^\top(\tilde{X}\theta - y) \tag{4}$$

The transpose $\tilde{X}^\top$ plays the role that $a$ played in the scalar case. Setting this gradient to zero:

$$2\tilde{X}^\top(\tilde{X}\theta - y) = 0 \tag{5}$$

The constant $2$ cannot force the expression to zero, so we divide it away:

$$\tilde{X}^\top(\tilde{X}\theta - y) = 0 \tag{6}$$

**Geometric meaning.** $\tilde{X}\theta - y$ is the vector of residuals — how far each prediction is from the truth. The equation says $\tilde{X}^\top$ times that vector is zero: the residuals are *orthogonal* (perpendicular) to every column of $\tilde{X}$. Our predictions are the closest reachable point to the true $y$ — exactly a projection.

**In our example.** With the bad guess $\theta = [0,\,0,\,0]^\top$, the residuals are:

$$\tilde{X}\theta - y = \begin{bmatrix} 0-40 \\ 0-60 \\ 0-60 \\ 0-50 \end{bmatrix} = \begin{bmatrix} -40 \\ -60 \\ -60 \\ -50 \end{bmatrix}$$

Then $\tilde{X}^\top(\tilde{X}\theta - y)$:

$$\begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & 2 & 3 & 1 \\ 4 & 6 & 4 & 6 \end{bmatrix} \begin{bmatrix} -40 \\ -60 \\ -60 \\ -50 \end{bmatrix} = \begin{bmatrix} -210 \\ -390 \\ -1060 \end{bmatrix} \neq \mathbf{0}$$

Not zero — we are not at the minimum yet. With the true $\theta = [10,\,10,\,5]^\top$ the residuals are all zero, so $\tilde{X}^\top \mathbf{0} = \mathbf{0}$ — we are exactly at the bottom.

### Step 4 — Solving for $\theta$

Expanding (6):

$$\tilde{X}^\top\tilde{X}\,\theta - \tilde{X}^\top y = 0$$

$$\tilde{X}^\top \tilde{X}\,\theta = \tilde{X}^\top y \tag{7}$$

Let $A = \tilde{X}^\top \tilde{X}$ and $b = \tilde{X}^\top y$. We have a linear system $A\theta = b$ and want to isolate $\theta$.

**Scalar analogy for the inverse.** In the scalar world, if $6\theta = 30$ we divide both sides by 6 (equivalently, multiply by $1/6$) to get $\theta = 5$. For matrices, "dividing by $A$" means multiplying by $A^{-1}$, the **matrix inverse** of $A$.

The key property of the inverse is:

$$A^{-1}A = I$$

where $I$ is the **identity matrix** — the matrix version of the number 1. It has ones on the diagonal and zeros everywhere else, and multiplying anything by $I$ leaves it unchanged: $I\theta = \theta$.

Multiplying both sides of $A\theta = b$ on the left by $A^{-1}$:

$$A^{-1} A\,\theta = A^{-1} b$$

$$I\,\theta = A^{-1} b \qquad \leftarrow \text{because } A^{-1}A = I$$

$$\boxed{\theta^* = A^{-1}b = (\tilde{X}^\top \tilde{X})^{-1}\tilde{X}^\top y} \tag{8}$$

This is the **Normal Equation**. The name "normal" comes from the geometric fact captured in (6): the residuals are *normal* (perpendicular) to the feature space.

**In our example.** First compute $\tilde{X}^\top \tilde{X}$ and $\tilde{X}^\top y$:

$$\tilde{X}^\top \tilde{X} = \begin{bmatrix} 1&1&1&1 \\ 1&2&3&1 \\ 4&6&4&6 \end{bmatrix} \begin{bmatrix} 1&1&4 \\ 1&2&6 \\ 1&3&4 \\ 1&1&6 \end{bmatrix} = \begin{bmatrix} 4 & 7 & 20 \\ 7 & 15 & 34 \\ 20 & 34 & 104 \end{bmatrix}$$

$$\tilde{X}^\top y = \begin{bmatrix} 1&1&1&1 \\ 1&2&3&1 \\ 4&6&4&6 \end{bmatrix} \begin{bmatrix} 40 \\ 60 \\ 60 \\ 50 \end{bmatrix} = \begin{bmatrix} 210 \\ 390 \\ 1060 \end{bmatrix}$$

Inverting a $3 \times 3$ matrix by hand requires a lengthy formula. In practice, we never do this manually — we hand the system to numpy, which is exactly what `np.linalg.lstsq` does. It solves $A\theta = b$ more stably using an SVD decomposition under the hood. The result it returns is:

$$\theta^* = \begin{bmatrix} 10 \\ 10 \\ 5 \end{bmatrix}$$

Verification — plug back into $A\theta = b$:

$$\begin{bmatrix} 4 & 7 & 20 \\ 7 & 15 & 34 \\ 20 & 34 & 104 \end{bmatrix} \begin{bmatrix} 10 \\ 10 \\ 5 \end{bmatrix} = \begin{bmatrix} 40+70+100 \\ 70+150+170 \\ 200+340+520 \end{bmatrix} = \begin{bmatrix} 210 \\ 390 \\ 1060 \end{bmatrix} = \tilde{X}^\top y \checkmark$$

`fit` implements this: prepend the ones column to form $\tilde{X}$, call `lstsq`, then split $\theta^*$ into `intercept_` (the first element, $\theta_0$) and `coef_` (the rest, $\theta_1, \theta_2, \ldots$).

### Step 5 — Making predictions

Once we have $\theta^*$, prediction on new data is just the linear map:

$$\hat{y} = X\hat\theta_{1:} + \hat\theta_0$$

where $\hat\theta_{1:}$ is stored as `coef_` and $\hat\theta_0$ as `intercept_`. `predict` computes exactly this.

**In our example.** A new student studied for 4 hours and slept 7 hours. In matrix form, their feature vector (with the ones prepended) is $[1,\,4,\,7]$:

$$\hat{y} = [1,\;4,\;7] \cdot \begin{bmatrix} 10 \\ 10 \\ 5 \end{bmatrix} = 10 + 40 + 35 = 85$$

The model predicts a score of 85.

### Step 6 — How good is the fit?

MSE tells us the absolute error but not whether the model is *useful*. A baseline that always predicts $\bar{y}$ (the mean of the targets) has some MSE of its own; a good model should beat it.

$R^2$ measures this relative improvement:

$$R^2 = 1 - \frac{\underbrace{\sum_i(y_i - \hat{y}_i)^2}_{\text{model's error}}}{\underbrace{\sum_i(y_i - \bar{y})^2}_{\text{baseline's error}}} \tag{9}$$

- $R^2 = 1$: predictions are perfect.
- $R^2 = 0$: model is no better than always predicting the mean.
- $R^2 < 0$: model is *worse* than that baseline (possible on held-out test data).

**In our example.**

$$\bar{y} = \frac{40+60+60+50}{4} = 52.5$$

Since our predictions are exact (no noise), every residual is zero: $\text{SS}_\text{res} = 0$.

$$\text{SS}_\text{tot} = (40-52.5)^2 + (60-52.5)^2 + (60-52.5)^2 + (50-52.5)^2 = 156.25 + 56.25 + 56.25 + 6.25 = 275$$

$$R^2 = 1 - \frac{0}{275} = 1.0$$

A perfect score — as expected for noise-free data. On a real (noisy) dataset you would typically see $R^2$ between 0 and 1.

`score` returns this value. Because every regressor computes $R^2$ the same way, `score` lives in `RegressorMixin` ([base.py](../base.py)) and `LinearRegression` inherits it.

---

## Implementation API

```python
class LinearRegression:
    coef_: np.ndarray       # shape (n_features,), set after fit
    intercept_: float       # scalar, set after fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def score(self, X: np.ndarray, y: np.ndarray) -> float: ...  # inherited from RegressorMixin (base.py)
```

**Numerical note.** Prefer `np.linalg.lstsq` over computing $(X^\top X)^{-1}$ explicitly. `lstsq` uses an SVD-based solver that stays numerically stable when features are collinear — cases where $\tilde{X}^\top \tilde{X}$ is near-singular and direct inversion would amplify rounding errors.

---

## Limitations

The Normal Equation works beautifully for small datasets. Two problems emerge as scale increases.

**Cost.** Forming $\tilde{X}^\top \tilde{X}$ and factorising it costs $O(mn^2 + n^3)$ time and $O(n^2)$ memory. For $n > 10{,}000$ features this becomes prohibitive.

**No path to generalisation.** Ridge regression, Lasso, and logistic regression do not share this clean closed form. What we want is a general-purpose optimiser we can reuse across all of them — that is gradient descent, covered next in [Gradient Descent Regression](./02-gradient-descent-regression.md).
