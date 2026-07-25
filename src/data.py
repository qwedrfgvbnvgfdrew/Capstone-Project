"""
data.py
-------
Loads the xAPI-Edu-Data dataset and performs light, well-documented cleaning.

Dataset: xAPI-Edu-Data (Students' Academic Performance Dataset)
Source:  https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
License: CC BY-SA 4.0

This module intentionally does NOT do any encoding/scaling — that lives in
preprocessing.py so the exact same transformation can be reused at inference
time (see predict.py). Keeping loading and preprocessing separate also makes
it obvious where the leakage boundary is: raw data in, engineered features
out, nothing fit here.
"""

import os
import pandas as pd

# Columns that describe how a student behaves in the course (used for clustering)
BEHAVIOR_NUMERIC_FEATURES = [
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion",
]

# Categorical behavioral / demographic features used for clustering.
# NOTE on PlaceofBirth: the raw file also contains a `PlaceofBirth` column.
# It is near-duplicate information with `NationalITy` (in this dataset every
# student's stated nationality and place of birth co-occur almost one-to-one),
# so it is dropped at the feature-selection stage in preprocessing.py to avoid
# redundant, highly-correlated categorical features. This is documented in the
# Capstone Brief section on "highly correlated variables" and is NOT the same
# issue as the Class leakage risk described below.
CATEGORICAL_FEATURES = [
    "gender",
    "NationalITy",
    "StageID",
    "GradeID",
    "SectionID",
    "Topic",
    "Semester",
    "Relation",
    "StudentAbsenceDays",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
]

ALL_CLUSTERING_FEATURES = CATEGORICAL_FEATURES + BEHAVIOR_NUMERIC_FEATURES

# Class is EXCLUDED from clustering (see Capstone Brief, Section 4 & 5:
# "Potential leakage risks"). It represents academic performance (the outcome),
# not learning behavior, and is only used AFTER clustering to help interpret
# the discovered learner segments.
TARGET_FOR_INTERPRETATION_ONLY = "Class"

REQUIRED_RAW_COLUMNS = ALL_CLUSTERING_FEATURES + ["PlaceofBirth", TARGET_FOR_INTERPRETATION_ONLY]


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw xAPI-Edu-Data CSV exactly as distributed, with a schema check."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find the dataset at '{csv_path}'. "
            "See data/README.md for download instructions."
        )
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The dataset at '{csv_path}' is missing expected columns: {missing}. "
            "Confirm you downloaded the original xAPI-Edu-Data.csv (480 rows, 17 columns)."
        )
    return df


def basic_quality_checks(df: pd.DataFrame) -> dict:
    """Return a small dict of data-quality facts used in the EDA notebook / README."""
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_missing_values_total": int(df.isna().sum().sum()),
        "class_distribution": df["Class"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    df = load_raw_data("data/xAPI-Edu-Data.csv")
    print(basic_quality_checks(df))
