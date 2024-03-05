import pandas as pd
import json
import string
import re
import os
from typing import Any, Dict


def preprocess_data(data: dict, models: dict, full_preprocess=False) -> pd.DataFrame:
    """
    Preprocess incoming JSON data and return a processed DataFrame.
    """
    df = pd.DataFrame([data])
    df["zip_code"] = df["zip_code"].apply(preprocess_zip_code)
    df["city"] = zip_to_city_mapping(df["zip_code"])
    df["emp_length"] = df["emp_length"].apply(convert_emp_length)
    df["average_fico_score"] = compute_average_fico_scores(
        df["fico_range_high"], df["fico_range_low"]
    )
    df["purpose"].fillna("other").apply(preprocess_text)
    purpose_tfidf = models["tfdif_vectorizer"].transform(df["purpose"])
    purpose_cluster = models["minibatch_kmeans"].predict(purpose_tfidf)
    df["purpose_cluster"] = purpose_cluster

    if full_preprocess:
        df = add_indicator_features(df)
        df["term"] = process_term(df["term"])
    return df


def preprocess_text(text: str) -> str:
    """Preprocceses text for vectorization"""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    return text


def add_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds binary indicator features for several 'months since' type columns."""
    mnths_since_events = [
        "mths_since_last_record",
        "mths_since_recent_bc_dlq",
        "mths_since_last_major_derog",
        "mths_since_recent_revol_delinq",
        "mths_since_last_delinq",
    ]
    for feature in mnths_since_events:
        df[f"{feature}_indicator"] = df[feature].apply(
            lambda x: 1 if pd.notnull(x) and x != 0 else 0
        )
    df = df.drop(columns=mnths_since_events)
    return df


def preprocess_zip_code(zip_code: str) -> str:
    """Cleans zip code by removing 'xx' and stripping whitespace."""
    zip_code_cleaned = (
        zip_code if isinstance(zip_code, int) else zip_code.replace("xx", "").strip()
    )
    return zip_code_cleaned


def process_term(term: pd.Series) -> pd.Series:
    """Extracts numeric value from term string."""
    cleaned_term = (
        term
        if isinstance(term, int)
        else term.str.replace("months", "").str.strip().astype(int)
    )
    return cleaned_term


def compute_average_fico_scores(fico_range_high: str, fico_range_low: str) -> float:
    """Computes the average FICO score from high and low range columns and adds it to the DataFrame."""
    high = pd.to_numeric(fico_range_high, errors="coerce")
    low = pd.to_numeric(fico_range_low, errors="coerce")
    return (high + low) / 2


def convert_emp_length(emp_length: str) -> int:
    """Converts employment length string to an integer value."""
    emp_length_mapping = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }
    return emp_length_mapping.get(emp_length, None)


def load_mapping(file_name: str) -> Dict[str, Any]:
    """Loads a mapping from a JSON file."""
    base_path = os.path.abspath(os.path.dirname(__file__))
    mapping_base_path = os.path.join(base_path, "mapping_data")
    with open(f"{mapping_base_path}/{file_name}", "r") as f:
        return json.load(f)


def grade_mapping(grade: int) -> str:
    """Maps grade output back to a letter"""
    grade_mapping = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
    return grade_mapping.get(grade, None)


def subgrade_mapping(subgrade: int) -> str:
    """Maps subgrade output back to letter-digit format"""
    subgrade_mapping = {
        i: f"{grade}{num}"
        for i, (grade, num) in enumerate((g, n) for g in "ABCDEFG" for n in range(1, 6))
    }
    return subgrade_mapping.get(subgrade, None)


def zip_to_city_mapping(zip_code: pd.Series) -> pd.Series:
    """Converts first three digits of zip codes to citys"""
    zip_code_to_city_mapping = load_mapping("zip_code_mapping.json")
    return zip_code.map(zip_code_to_city_mapping)
