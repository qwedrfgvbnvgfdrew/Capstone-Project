"""
preprocessing.py
----------------
Builds the leakage-safe preprocessing pipeline and the train/test split used
throughout the project.

Split strategy
--------------
Each row is one independent student record (no repeated users, no time
ordering, no group structure to respect), so a simple random split is
appropriate. We hold out 20% of students as an "unseen" test set that is
NEVER used to fit the scaler, the one-hot encoder, or the clustering model.
The held-out set is used only in evaluate.py to check whether the trained
model generalizes to students it has never seen (Criterion 4 / Section 6 of
the Capstone Brief: "Data splitting / validation strategy").

Preprocessing steps
--------------------
1. Drop `PlaceofBirth` (redundant with `NationalITy`) and `Class` (excluded
   from clustering; kept aside only for post-hoc interpretation).
2. One-hot encode all categorical / demographic columns.
3. Standardize all numeric behavioral columns (K-Means and the other
   distance-based algorithms require comparable scales).
All fitting happens on the TRAIN split only; the TEST split is only ever
`.transform()`-ed, never `.fit()`-ed on.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.data import CATEGORICAL_FEATURES, BEHAVIOR_NUMERIC_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20


def split_data(df, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Random 80/20 split. Rows are independent students, so no grouping/time split is needed."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """Column transformer: one-hot encode categoricals, standardize numeric behavior features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), BEHAVIOR_NUMERIC_FEATURES),
        ],
        remainder="drop",  # explicitly drops PlaceofBirth and Class
    )
    return preprocessor


def get_feature_frame(df):
    """Return only the columns the preprocessor expects, in a stable order."""
    return df[CATEGORICAL_FEATURES + BEHAVIOR_NUMERIC_FEATURES].copy()
