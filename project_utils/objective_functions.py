import pandas as pd
import xgboost as xgb
from typing import Callable
from optuna.integration import XGBoostPruningCallback
import numpy as np


def get_objective_xgb_binary(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    scoring: callable,
) -> Callable:
    def objective(trial) -> float:
        xgb_param_grid = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 300, 500),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.05, 0.2),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 5),
            "scale_pos_weight": trial.suggest_float("xgb_scale_pos_weight", 1, 5),
            "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 5),
            "subsample": trial.suggest_float("xgb_subsample", 0.25, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("xgb_lambda", 1e-5, 1.0, log=True),
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "n_jobs": -1,
            "random_state": 42,
        }

        model = xgb.XGBClassifier(**xgb_param_grid)
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-logloss")
        model.set_params(callbacks=[pruning_callback], early_stopping_rounds=20)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        preds = model.predict(X_val)
        score = scoring(y_val, preds)
        return score

    return objective


def get_objective_xgb_multi_class(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    scoring: callable,
    weights: np.ndarray,
) -> Callable:
    def objective(trial) -> float:

        use_weights = trial.suggest_categorical("use_class_weights", [True, False])
        xgb_param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.25, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("lambda", 0, 1.0, log=True),
            "reg_alpha": trial.suggest_float("alpha", 0.0, 1.0, log=True),
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "objective": "multi:softmax",
            "n_jobs": -1,
            "random_state": 42,
        }

        model = xgb.XGBClassifier(**xgb_param_grid)
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-mlogloss")
        model.set_params(callbacks=[pruning_callback], early_stopping_rounds=40)
        fit_params = {
            "eval_set": [(X_val, y_val)],
        }
        if use_weights:
            fit_params["sample_weight"] = weights

        model.fit(X_train, y_train, **fit_params)
        preds = model.predict(X_val)
        score = scoring(y_val, preds, average="macro")
        return score

    return objective


def get_objective_xgb_regression(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    scoring: callable,
) -> Callable:
    def objective(trial) -> float:
        xgb_param_grid = {
            "n_estimators": trial.suggest_int("estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.3),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("xgb_lambda", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("xgb_alpha", 1e-8, 1.0, log=True),
            "eval_metric": "rmse",
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 2,
        }

        model = xgb.XGBRegressor(**xgb_param_grid)
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
        model.set_params(callbacks=[pruning_callback], early_stopping_rounds=40)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )
        preds = model.predict(X_val)
        score = scoring(y_val, preds, squared=False)
        return score

    return objective
