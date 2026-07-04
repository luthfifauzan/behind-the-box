"""Benchmark behind_the_box models against their sklearn baselines.

Fits every requested model (plus the sklearn counterpart) on a dataset,
then prints a leaderboard: fit time, train/test R², optional K-fold CV
score, and coefficient distance from sklearn.

Usage:
    uv run python -m experiments.benchmark
    uv run python -m experiments.benchmark --dataset california --cv 5
    uv run python -m experiments.benchmark --models 01-closed-form-regression

sklearn's model_selection utilities (train_test_split, KFold) are used as
tooling here — the learning target is the models, not the split logic.
"""

import argparse
import time
from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.model_selection import KFold, train_test_split

from behind_the_box import MODELS
from experiments.datasets import LOADERS

# sklearn counterpart per algorithm file stem — the "known good" baseline
# each model is compared against. Extend as new models land.
SKLEARN_BASELINES: dict[str, Callable] = {
    "01-closed-form-regression": SklearnLR,
}


Split = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _evaluate(factory: Callable, split: Split, cv: int, seed: int) -> dict:
    """Fit one model and collect leaderboard metrics."""
    X_train, X_test, y_train, y_test = split
    start = time.perf_counter()
    model = factory().fit(X_train, y_train)
    fit_ms = (time.perf_counter() - start) * 1000

    row = {
        "fit_ms": fit_ms,
        "train_r2": model.score(X_train, y_train),
        "test_r2": model.score(X_test, y_test),
        "coef": getattr(model, "coef_", None),
    }

    if cv:
        # Manual K-fold: our models are plain classes (no get_params/clone),
        # so sklearn's cross_val_score cannot drive them.
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        scores = [
            factory().fit(X[tr], y[tr]).score(X[te], y[te])
            for tr, te in KFold(n_splits=cv, shuffle=True, random_state=seed).split(X)
        ]
        row["cv_mean"], row["cv_std"] = np.mean(scores), np.std(scores)

    return row


def run(dataset: str, model_names: list[str], cv: int, seed: int) -> None:
    """Fit the requested models + sklearn baselines and print a leaderboard."""
    X, y, problem = LOADERS[dataset](seed=seed)
    split: Split = train_test_split(X, y, test_size=0.2, random_state=seed)
    print(
        f"dataset={dataset} ({X.shape[0]}x{X.shape[1]}, {problem}) "
        f"seed={seed} test_size=0.2" + (f" cv={cv}" if cv else ""),
    )

    candidates: dict[str, Callable] = {}
    for name in model_names:
        candidates[name] = MODELS[name]
        baseline = SKLEARN_BASELINES.get(name)
        if baseline is not None:
            candidates[f"sklearn:{baseline.__name__}"] = baseline

    rows = {
        name: _evaluate(factory, split, cv, seed)
        for name, factory in candidates.items()
    }

    # Coefficient distance vs the first sklearn baseline (if any).
    ref_coef = next(
        (r["coef"] for n, r in rows.items() if n.startswith("sklearn:")), None,
    )

    header = f"{'model':<32} {'fit_ms':>8} {'train_R²':>9} {'test_R²':>9}"
    if cv:
        header += f" {'CV_R²':>17}"
    header += f" {'max|Δcoef|':>11}"
    print(header)
    print("-" * len(header))

    for name, row in sorted(rows.items(), key=lambda kv: -kv[1]["test_r2"]):
        line = (
            f"{name:<32} {row['fit_ms']:>8.2f} "
            f"{row['train_r2']:>9.4f} {row['test_r2']:>9.4f}"
        )
        if cv:
            line += f" {row['cv_mean']:>8.4f} ± {row['cv_std']:<6.4f}"
        if ref_coef is not None and row["coef"] is not None:
            line += f" {np.max(np.abs(row['coef'] - ref_coef)):>11.2e}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", choices=sorted(LOADERS), default="synthetic",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS),
        default=sorted(MODELS),
        help="algorithm file stems (default: all implemented models)",
    )
    parser.add_argument("--cv", type=int, default=0, help="K-fold CV splits (0 = off)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.dataset, args.models, args.cv, args.seed)


if __name__ == "__main__":
    main()
