"""
predict.py
----------
Inference interface: assign a brand-new student to one of the discovered
learner segments using the saved preprocessing pipeline + final K-Means model.

This satisfies:
  - Expected Deliverable: "A simple prediction function ... that assigns a new
    student to one of the discovered clusters."
  - Acceptance Criterion: "The solution can process previously unseen input."

Usage (after running `python -m src.train` at least once so models/ exists):

    from src.predict import predict_student_cluster

    student = {
        "gender": "F",
        "NationalITy": "Jordan",
        "StageID": "MiddleSchool",
        "GradeID": "G-08",
        "SectionID": "A",
        "Topic": "Math",
        "Semester": "F",
        "Relation": "Mum",
        "StudentAbsenceDays": "Under-7",
        "ParentAnsweringSurvey": "Yes",
        "ParentschoolSatisfaction": "Good",
        "raisedhands": 55,
        "VisITedResources": 60,
        "AnnouncementsView": 30,
        "Discussion": 40,
    }
    result = predict_student_cluster(student)
    print(result)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from src.data import CATEGORICAL_FEATURES, BEHAVIOR_NUMERIC_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

REQUIRED_FIELDS = CATEGORICAL_FEATURES + BEHAVIOR_NUMERIC_FEATURES

RECOMMENDATIONS = {
    "Highly Engaged Learners": "Offer advanced learning materials and leadership opportunities.",
    "Consistent Learners": "Maintain regular learning support and encourage continued engagement.",
    "Struggling but Active Learners": "Provide tutoring, personalized feedback, and additional practice resources.",
    "At-Risk Learners": "Send early-warning notifications, increase mentor communication, and recommend academic support services.",
}


class InvalidStudentInputError(ValueError):
    """Raised when a student record is missing fields or has an invalid type/value."""


def _validate_student(student: dict) -> None:
    if not isinstance(student, dict):
        raise InvalidStudentInputError(f"Expected a dict of student features, got {type(student)}.")

    missing = [f for f in REQUIRED_FIELDS if f not in student]
    if missing:
        raise InvalidStudentInputError(f"Missing required field(s): {missing}")

    for col in BEHAVIOR_NUMERIC_FEATURES:
        value = student[col]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidStudentInputError(
                f"Field '{col}' must be numeric (0-100 scale in the original dataset), got {value!r} ({type(value)})."
            )
        if value < 0 or value > 100:
            raise InvalidStudentInputError(
                f"Field '{col}' = {value} is outside the expected 0-100 range used by this dataset."
            )

    for col in CATEGORICAL_FEATURES:
        value = student[col]
        if not isinstance(value, str) or not value.strip():
            raise InvalidStudentInputError(f"Field '{col}' must be a non-empty string, got {value!r}.")


_preprocessor = None
_model = None
_metadata = None


def _load_artifacts():
    """Lazy-load artifacts once per process."""
    global _preprocessor, _model, _metadata
    if _preprocessor is None:
        preproc_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        model_path = os.path.join(MODELS_DIR, "final_kmeans_model.joblib")
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        for p in (preproc_path, model_path, meta_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Could not find '{p}'. Run `python -m src.train` first to generate the saved model artifacts."
                )
        _preprocessor = joblib.load(preproc_path)
        _model = joblib.load(model_path)
        with open(meta_path) as f:
            _metadata = json.load(f)
    return _preprocessor, _model, _metadata


def predict_student_cluster(student: dict) -> dict:
    """
    Assign a single new student to a discovered learner segment.

    Parameters
    ----------
    student : dict
        Must contain every field listed in REQUIRED_FIELDS (see src/data.py).
        Behavioral fields (raisedhands, VisITedResources, AnnouncementsView,
        Discussion) must be numeric in the 0-100 range used by the original
        dataset. Categorical fields must be non-empty strings using the same
        category spellings as the training data (e.g. StudentAbsenceDays is
        "Under-7" or "Above-7").

    Returns
    -------
    dict with keys: cluster_id, cluster_name, recommendation, confidence_note
    """
    _validate_student(student)
    preprocessor, model, metadata = _load_artifacts()
    cluster_name_map = {int(k): v for k, v in metadata["cluster_name_map"].items()}

    row = pd.DataFrame([{k: student[k] for k in REQUIRED_FIELDS}])
    X = preprocessor.transform(row)
    cluster_id = int(model.predict(X)[0])
    cluster_name = cluster_name_map.get(cluster_id, f"Cluster {cluster_id}")

    # Confidence note: how much closer is the assigned centroid than the runner-up?
    distances = model.transform(X)[0]
    sorted_dist = np.sort(distances)
    gap = float(sorted_dist[1] - sorted_dist[0])
    if gap < 0.5:
        confidence_note = (
            "Low confidence: this student's behavior sits nearly equidistant between two "
            "learner profiles. Treat the assignment as provisional."
        )
    elif gap < 1.5:
        confidence_note = "Moderate confidence in this cluster assignment."
    else:
        confidence_note = "High confidence: this student is clearly closest to one learner profile."

    return {
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "recommendation": RECOMMENDATIONS.get(cluster_name, "Review student profile manually."),
        "confidence_note": confidence_note,
    }


def predict_students_batch(students: list) -> pd.DataFrame:
    """Convenience wrapper: run predict_student_cluster over a list of student dicts."""
    rows = []
    for i, s in enumerate(students):
        try:
            result = predict_student_cluster(s)
            result["row_index"] = i
            result["error"] = None
        except InvalidStudentInputError as e:
            result = {"row_index": i, "cluster_id": None, "cluster_name": None,
                      "recommendation": None, "confidence_note": None, "error": str(e)}
        rows.append(result)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    example_student = {
        "gender": "F",
        "NationalITy": "Jordan",
        "StageID": "MiddleSchool",
        "GradeID": "G-08",
        "SectionID": "A",
        "Topic": "Math",
        "Semester": "F",
        "Relation": "Mum",
        "StudentAbsenceDays": "Under-7",
        "ParentAnsweringSurvey": "Yes",
        "ParentschoolSatisfaction": "Good",
        "raisedhands": 55,
        "VisITedResources": 60,
        "AnnouncementsView": 30,
        "Discussion": 40,
    }
    print(predict_student_cluster(example_student))

    invalid_student = dict(example_student)
    invalid_student["raisedhands"] = "a lot"  # wrong type on purpose
    try:
        predict_student_cluster(invalid_student)
    except InvalidStudentInputError as e:
        print(f"\nCorrectly rejected invalid input: {e}")
