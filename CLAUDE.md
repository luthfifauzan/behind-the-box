# CLAUDE.md — Project Instructions

## TDD Learning Loop

Every ML algorithm in this repo follows the same 5-step cycle:

1. **Explain** — Claude writes `behind_the_box/<family>/<NN>-<name>.md`: motivation, math derivation, API spec
2. **Test** — Claude writes `tests/<family>/test_<NN>_<name>.py`: subclasses the shared check suites in `tests/<family>/regressor_checks.py` + model-specific math-property tests
3. **Hint** — Claude provides a `.py` stub: class skeleton + docstrings, no math
4. **Build** — The user fills in the implementation
5. **Verify** — Run `uv run pytest tests/ -v`; Claude gives hints only when asked. Once green, `uv run python -m experiments.benchmark` compares the model against its sklearn baseline

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
behind_the_box/            # installable package (sklearn/sklearn style; ADR-0003)
  __init__.py              # aggregates family registries into MODELS
  base.py                  # shared mixins (RegressorMixin.score); keep minimal
  datasets.py              # dataset loaders (sklearn.datasets analog, fetch+cache); single source for tests + experiments
  <family>/
    __init__.py            # auto-discovers NN-*.py; no per-model registration needed
    <NN>-<name>.md         # explanation (Claude writes)
    <NN>-<name>.py         # implementation (user writes)

tests/
  conftest.py              # shared fixtures: rng, synthetic, california_housing, advertising
  <family>/
    regressor_checks.py    # reusable check suites, configured via subclass attrs
    test_<NN>_<name>.py    # tests (Claude writes)
  fixtures/
    advertising.csv        # gitignored; optional — auto-downloaded to ~/.cache/behind_the_box/ otherwise

experiments/               # model-selection harness (ADR-0004); Claude writes
  benchmark.py             # leaderboard CLI: models vs sklearn baselines
```

## Conventions

- **Class names**: sklearn parity where a counterpart exists (`LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`); otherwise descriptive (`GradientDescentRegressor`). One public model class per file, unique across the repo.
- **Imports**: `from behind_the_box.<family> import <Class>` — works everywhere; `uv sync` installs the package editable.
- **Shared code readability**: anything moved into `base.py` must be pointed to from each model's class docstring and linked from its `.md` — a reader of one pair is always told where inherited methods live.
- User-written math stays user-written: refactors may *move* the user's implementation code but never rewrite it.

## Verification

```bash
uv run pytest tests/ -v                                   # -m "not network" when offline
uv run python -m experiments.benchmark --dataset california --cv 5
```

## Algorithm Order (linear_models)

```
01-closed-form-regression       OLS via Normal Equation
02-gradient-descent-regression  OLS via gradient descent
03-ridge-regression             L2 regularization
04-lasso-regression             L1 regularization (coordinate descent)
05-logistic-regression          Binary classification, cross-entropy
```
