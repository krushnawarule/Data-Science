"""Simple example of data filtering and cleaning with pandas."""

from __future__ import annotations

import pandas as pd


DATA_FILE = "student-scores.csv"


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    """Load CSV data into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning operations."""
    cleaned = df.copy()

    # Normalize column names
    cleaned.columns = cleaned.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # Remove duplicate rows
    cleaned = cleaned.drop_duplicates()

    # Convert score columns to numeric and fill missing values with median
    score_cols = [col for col in cleaned.columns if col.endswith("_score")]
    cleaned[score_cols] = cleaned[score_cols].apply(pd.to_numeric, errors="coerce")
    cleaned[score_cols] = cleaned[score_cols].fillna(cleaned[score_cols].median())

    # Fill missing values in other numeric columns with median
    numeric_cols = cleaned.select_dtypes(include="number").columns
    cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median())

    # Fill categorical missing values with mode
    categorical_cols = cleaned.select_dtypes(include=["object", "string", "category", "bool"]).columns
    for col in categorical_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0])

    return cleaned


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return students with high scores and good attendance."""
    filtered = df[
        (df["math_score"] >= 80)
        & (df["english_score"] >= 80)
        & (df["science_score"] >= 80)
        & (df["absence_days"] <= 5)
    ]

    return filtered.sort_values(["math_score", "science_score", "english_score"], ascending=False)


def main() -> None:
    df = load_data()
    cleaned = clean_data(df)
    filtered = filter_data(cleaned)

    print(f"Original rows: {len(df)}")
    print(f"Rows after cleaning: {len(cleaned)}")
    print(f"Rows after filtering: {len(filtered)}")
    print("\nTop filtered records:")
    print(filtered.head(10))


if __name__ == "__main__":
    main()
