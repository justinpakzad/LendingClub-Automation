from joblib import load
from preprocessing import grade_mapping, subgrade_mapping
import os
import pandas as pd


def load_models() -> dict:
    """Load all necessary machine learning models/pipelines."""
    base_path = os.path.abspath(os.path.dirname(__file__))
    models_base_path = os.path.join(base_path, "saved_model_pipelines")

    return {
        "accepted_rejected_pipeline": load(
            os.path.join(models_base_path, "xgb_pipeline_accepted_rejected.joblib")
        ),
        "tfdif_vectorizer": load(os.path.join(models_base_path, "tfidf_model.joblib")),
        "minibatch_kmeans": load(
            os.path.join(models_base_path, "minibatch_kmeans_model.joblib")
        ),
        "grade_pipeline": load(
            os.path.join(models_base_path, "xgb_pipeline_grade.joblib")
        ),
        "subgrade_pipeline": load(
            os.path.join(models_base_path, "xgb_pipeline_subgrade.joblib")
        ),
        "int_rate_pipeline": load(
            os.path.join(models_base_path, "xgb_pipeline_int_rate.joblib")
        ),
    }


def predict_accepted_rejected(df_processed: pd.DataFrame, models: dict) -> bool:
    """Predicts the acceptance/rejection of a loan application"""
    accepted_rejected_pred = models["accepted_rejected_pipeline"].predict_proba(
        df_processed
    )[:, -1]
    is_accepted = accepted_rejected_pred[0] >= 0.6
    return is_accepted


def predict_grade(df_processed: pd.DataFrame, models: dict) -> str:
    """Predicts the grade of a loan"""
    grade_pred = models["grade_pipeline"].predict(df_processed)[0]
    grade_letter = grade_mapping(grade_pred)
    df_processed["grade"] = grade_letter
    return grade_letter


def predict_subgrade(df_processed: pd.DataFrame, models: dict) -> str:
    """Predicts the sub grade of a loan"""
    subgrade_pred = models["subgrade_pipeline"].predict(df_processed)[0]
    subgrade_letter = subgrade_mapping(subgrade_pred)
    df_processed["sub_grade"] = subgrade_letter
    return subgrade_letter


def predict_int_rate(df_processed: pd.DataFrame, models: dict) -> float:
    """Predicts the interest rate"""
    int_rate_pred = models["int_rate_pipeline"].predict(df_processed)[0]
    return round(int_rate_pred.item(), 2)
