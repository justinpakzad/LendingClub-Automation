import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
import string
import re
from sklearn.base import TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import spmatrix
from typing import Tuple, List, Optional
from scipy.stats import chi2_contingency, mannwhitneyu
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr


def compute_correlations_and_pvalues(
    df: pd.DataFrame, target: pd.Series, method="pearson"
) -> pd.DataFrame:
    """
    Computes correlation of features vs target variable.
    Returns correlation values as well as p-values.
    """
    features = df.select_dtypes(exclude=["object", "category"]).drop(target, axis=1)
    results = []
    clean_df = df.dropna()
    for feature in features.columns:
        if method == "pearson":
            corr, p_value = pearsonr(clean_df[feature], clean_df[target])
        elif method == "spearman":
            corr, p_value = spearmanr(clean_df[feature], clean_df[target])
        else:
            raise ValueError("Method must be either 'pearson' or 'spearman'.")
        results.append((feature, corr, p_value))
    results_df = pd.DataFrame(
        results, columns=["feature", "correlation", "p_value"]
    ).sort_values(by="correlation", ascending=False)

    return results_df


def get_numerical_categorical_columns(X: pd.DataFrame):
    """Extracts numerical and categorical column names"""
    numerical_cols = X.select_dtypes(
        exclude=["object", "datetime64", "category"]
    ).columns
    categorical_cols = X.select_dtypes(exclude=np.number).columns
    return numerical_cols, categorical_cols


def bin_by_quartiles(X: pd.DataFrame, column_to_bin: pd.Series) -> pd.Series:
    """Bins selected column by quartile"""
    bins = [
        X[column_to_bin].min(),
        X[column_to_bin].quantile(0.50),
        X[column_to_bin].quantile(0.75),
        X[column_to_bin].max(),
    ]
    bins = sorted(set(bins))
    labels = [f"{column_to_bin}_bin{i+1}" for i in range(len(bins) - 1)]

    binned_series = pd.cut(
        X[column_to_bin], bins=bins, labels=labels, include_lowest=True
    )
    return binned_series


def apply_minibatch_kmeans(
    tfidf_train: spmatrix, *tfidf_additional_data: spmatrix, n_clusters: int
) -> Tuple[ndarray, ...]:
    """Applies  MiniBatchKMeans clustering to the TF-IDF transformed data."""
    mini_kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, batch_size=10_000, n_init="auto"
    )
    train_clusters = mini_kmeans.fit_predict(tfidf_train)
    additional_data_clusters = [
        mini_kmeans.predict(data) for data in tfidf_additional_data
    ]
    return (train_clusters,) + tuple(additional_data_clusters), mini_kmeans


def apply_tfidf_vectorizer(
    X_train: List[str], *additional_data: List[str]
) -> List[spmatrix]:
    """
    Fit and transforms TF-IDF vectorizer to the training data
    and applies the transformation to additional columns.
    """
    tfidf = TfidfVectorizer(
        max_features=1000, min_df=1, max_df=0.8, stop_words="english"
    )
    train_fit_transformed = tfidf.fit_transform(X_train)
    additional_data_transformed = [tfidf.transform(data) for data in additional_data]

    return [train_fit_transformed] + additional_data_transformed, tfidf


def preprocess_text(text: str) -> str:
    """Preprocceses text for vectorization"""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    return text


def null_percentages(df: pd.DataFrame) -> pd.Series:
    """Computes percentages of null values for a dataframe"""
    return (df.isnull().sum() / df.shape[0]) * 100


def create_numerical_pipeline(imputer: TransformerMixin) -> Pipeline:
    """Creates a pipeline for processing numerical data."""
    return Pipeline([("imputer", imputer)])


def create_categorical_pipeline(
    imputer: TransformerMixin, encoder: TransformerMixin
) -> Pipeline:
    """Creates a pipeline for processing categorical data."""
    return Pipeline([("imputer", imputer), ("encoder", encoder)])


def create_preprocessor(
    numerical_cols: list[str],
    categorical_cols: list[str],
    imputer_numerical: TransformerMixin,
    imputer_categorical: TransformerMixin,
    encoder: TransformerMixin,
    scaler: Optional[TransformerMixin] = None,
) -> ColumnTransformer:
    """
    Creates a preprocessor for both numerical and
    categorical data, with an option to scale all features.
    """

    transformers = [
        (
            "numerical",
            create_numerical_pipeline(imputer_numerical),
            numerical_cols,
        ),
        (
            "categorical",
            create_categorical_pipeline(imputer_categorical, encoder),
            categorical_cols,
        ),
    ]
    if scaler:
        return Pipeline(
            [
                (
                    "preprocessing",
                    ColumnTransformer(transformers, remainder="passthrough"),
                ),
                ("scaler", scaler),
            ]
        )

    return ColumnTransformer(transformers, remainder="passthrough")


def create_pipeline(
    model: TransformerMixin, preprocessor: ColumnTransformer, feature_selection=None
) -> Pipeline:
    """
    Creates pipeline from given a model, preprocessor,
    and optional feature selection tool
    """
    steps = [("preprocessor", preprocessor)]
    if feature_selection:
        steps.append(("feature_selection", feature_selection))
    steps.append(("classifier", model))
    return Pipeline(steps)


def replace_categories_inplace(df: pd.DataFrame, col: str, mapping: dict) -> str:
    """
    In-place modification to replace instances of a specified column based
    on whether they contain certain keywords with their new values.
    """
    try:
        for keyword, new_val in mapping.items():
            df.loc[df[col].str.contains(keyword, case=False, na=False), col] = new_val
        return "Successfully modified"
    except Exception as e:
        print(f"Error updating DataFrame: {e}")


def extract_feature_importances(model_pipeline: Pipeline) -> pd.DataFrame:
    """
    Extracts feature importance from pipeline
    and returns dataframe.
    """
    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = model_pipeline.named_steps["classifier"].feature_importances_
    feature_importances = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)
    feature_importances["feature"] = (
        feature_importances["feature"]
        .str.replace("categorical|pass|numerical", " ", regex=True)
        .str.strip()
    )

    return feature_importances


def confusion_matrix_df(y_test: ndarray, y_preds: ndarray) -> pd.DataFrame:
    """Creates confusion matrix dataframe"""
    conf_matrix = pd.DataFrame(
        confusion_matrix(y_test, y_preds),
        columns=["Predicted Negative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"],
    )
    return conf_matrix


def mutual_information_scores(
    X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer
) -> pd.DataFrame:
    """
    Computes mutual information scores for training data and returns a DataFrame.
    """
    preprocessor.fit(X_train, y_train)
    X_preprocessed = preprocessor.transform(X_train)
    mi_scores = mutual_info_classif(X_preprocessed, y_train)
    mi_scores_df = pd.DataFrame(
        {"feature": preprocessor.get_feature_names_out(), "mi_score": mi_scores}
    )
    mi_scores_df["feature"] = (
        mi_scores_df["feature"]
        .str.replace("categorical|numerical|pass|_", " ", regex=True)
        .str.strip()
    )
    mi_scores_df = mi_scores_df.sort_values(by="mi_score", ascending=False).reset_index(
        drop=True
    )
    return mi_scores_df


def multiple_test_chi2(X: pd.DataFrame, y: pd.Series) -> dict[str, List[float]]:
    """Performs chi-square tests for a list of features against a target."""
    results = {}
    for feature in X.columns:
        contingency_table = pd.crosstab(X[feature], y)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        results[feature] = [p_value, chi2]
    return results


def mutiple_test_mann_whitney(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Performs Mann-Whitney U tests for a list of features against a target."""
    results = {}
    for feature in X.columns:
        non_null_indices = X[feature].notna() & y.notna()
        feature_data = X.loc[non_null_indices, feature]
        target_data = y[non_null_indices]
        group1 = feature_data[target_data == 0]
        group2 = feature_data[target_data == 1]
        u_stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
        results[feature] = [p_value, u_stat]
    return results


def evaluate_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    preprocessor_linear: TransformerMixin,
    preprocessor_tree_based: TransformerMixin,
) -> tuple[dict, dict]:
    """
    Evaluates linear and tree based models
    and returns metrics as well as feature importances.
    """
    model_results = {}
    feature_importances = {}
    for model_name, model in tqdm(models.items()):
        if model_name in ["LogisticRegression", "LinearSVM"]:
            pipeline = create_pipeline(model, preprocessor_linear, None)
        else:
            pipeline = create_pipeline(model, preprocessor_tree_based, None)

        pipeline.fit(X_train, y_train)
        y_preds = pipeline.predict(X_val)

        model_results[model_name] = [
            f1_score(y_val, y_preds, average="macro"),
            precision_score(y_val, y_preds, average="macro"),
            recall_score(y_val, y_preds, average="macro"),
        ]

        if model_name in ["XGBoost", "RandomForest"]:
            feature_importances[model_name] = extract_feature_importances(pipeline)

    return model_results, feature_importances
