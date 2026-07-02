# CLAUDE.md — Project Instructions

## TDD Learning Loop

Every ML algorithm in this repo follows the same 5-step cycle:

1. **Explain** — Claude writes `<family>/<NN>-<name>.md`: motivation, math derivation, API spec
2. **Test** — Claude writes `tests/<family>/test_<NN>_<name>.py`: rigorous test suite
3. **Hint** — Claude provides a `.py` stub: class skeleton + docstrings, no math
4. **Build** — The user fills in the implementation
5. **Verify** — Run `uv run pytest tests/ -v`; Claude gives hints only when asked

**Claude must never write the implementation.** The `.py` file belongs to the user.

## Hint Tiers (on request only)

- **Hint 1 (shape)**: what matrices/vectors are needed and their shapes
- **Hint 2 (operation)**: which NumPy function solves it
- **Hint 3 (code)**: a 2–3 line snippet, no full context

The user asks for hints. Claude does not volunteer them.

## Test Categories (per algorithm)

| Category | What it checks |
|---|---|
| Mathematical properties | Invariants (e.g. residuals ⊥ features) |
| Oracle comparison | Matches scikit-learn's output on random data |
| Edge cases | Single feature, many features, collinear, no-intercept |
| API contract | `fit` returns `self`, fitted attrs end in `_` |
| Real dataset | `fetch_california_housing` + `tests/fixtures/advertising.csv` |

## File Layout

```
<family>/
  <NN>-<name>.md       # explanation (Claude writes)
  <NN>-<name>.py       # implementation (user writes)
  __init__.py

tests/
  <family>/
    test_<NN>_<name>.py  # tests (Claude writes)
  fixtures/
    advertising.csv      # gitignored; user downloads from Kaggle
```

## Verification

```bash
uv run pytest tests/ -v
```

## Algorithm Order (linear_models)

```
01-closed-form-regression       OLS via Normal Equation
02-gradient-descent-regression  OLS via gradient descent
03-ridge-regression             L2 regularization
04-lasso-regression             L1 regularization (coordinate descent)
05-logistic-regression          Binary classification, cross-entropy
```
