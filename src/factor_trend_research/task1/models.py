"""Small, prespecified model families; optional LightGBM never falls back."""
import itertools
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge


def model_grid(family, config):
    if family == "ridge":
        return [{"lambda": float(x)} for x in config["ridge_lambdas"]]
    if family == "logistic":
        return [{"C": float(x)} for x in config["logistic_C"]]
    if family == "lightgbm":
        grid = config["lightgbm_grid"]
        return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    raise ValueError(f"Unknown model family: {family}")


def estimator(family, parameters, training_rows):
    if family == "ridge":
        # sklearn minimizes SSE + alpha*||b||^2; n*lambda represents MSE+lambda penalty.
        return Ridge(alpha=training_rows*parameters["lambda"], solver="cholesky")
    if family == "logistic":
        # Default finite-C LBFGS is L2 in sklearn>=1.6; omitting the deprecated
        # penalty argument also supports the newer l1_ratio=0 default API.
        return LogisticRegression(C=parameters["C"], solver="lbfgs",
                                  max_iter=1000, tol=1e-6)
    if family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as error:
            raise ImportError("LightGBM was requested: install the project's lightgbm extra "
                              "and its platform runtime, or explicitly choose ridge,logistic. "
                              "No substitute model is used.") from error
        return LGBMClassifier(objective="binary", learning_rate=.05, reg_lambda=1.,
                              max_bin=63, n_jobs=1, random_state=20261109,
                              deterministic=True, force_col_wise=True, verbosity=-1,
                              **parameters)
    raise ValueError(f"Unknown model family: {family}")


def fit_model(family, parameters, x, target, tail_y):
    model = estimator(family, parameters, len(target))
    mask = np.isfinite(tail_y) if family != "ridge" else np.ones(len(target), bool)
    y = tail_y[mask].astype(int) if family != "ridge" else target
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x[mask], y)
    return model


def predict_score(model, family, x):
    if family == "ridge":
        return model.predict(x)
    positive = list(model.classes_).index(1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
        return model.predict_proba(x)[:, positive]
