# EDU-01 — Student Learning Segmentation

**Student:** Nilufar Azimjonova
**Track:** Track 2 — Field-Based Scenario (EdTech, Scenario EDU-01: Student Learning Segmentation)
**Project brief:** the official scenario brief is included in this repository as
[`FIELD-BASED_CAPSTONE_SCENARIO.pdf`](FIELD-BASED_CAPSTONE_SCENARIO.pdf); every
design decision below traces back to that document.

**Repository URL:** https://github.com/qwedrfgvbnvgfdrew/Capstone-Project.git

---

## 1. Problem Statement

An online education platform gives every student the same generic support,
regardless of how differently they actually behave in the course — some
students engage constantly (raising hands, visiting resources, joining
discussions), others barely engage at all. Without a way to group students by
behavior, the platform's educators cannot target interventions, and student
disengagement goes unnoticed until it shows up in final grades. The goal of
this project is to **discover meaningful, interpretable learner groups from
behavioral data alone**, so a student-success team can act on engagement
patterns before they become poor outcomes.

## 2. Selected Project Track

**Track 2 — Field-Based Scenario**, EdTech domain, Scenario EDU-01 (Student
Learning Segmentation), as defined in `FIELD-BASED_CAPSTONE_SCENARIO.pdf`.

## 3. Dataset Source

- **xAPI-Edu-Data** (Students' Academic Performance Dataset), 480 students, 17 columns.
- Source: https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
- License: CC BY-SA 4.0.
- Full data documentation, including the exact columns used and why
  `PlaceofBirth` and `Class` are excluded from clustering: see
  [`data/README.md`](data/README.md).

## 4. ML Task Type

**Unsupervised learning — clustering.** There is no ground-truth label for
"learner type"; the model discovers natural groups of students from 4
numeric behavioral features (raised hands, resources visited, announcements
viewed, discussion posts) and 11 categorical/demographic features, using
K-Means as the final algorithm. The `Class` academic-performance label exists
in the data but is deliberately **excluded from training** and used only
afterward, to sanity-check whether the discovered behavioral clusters relate
to real academic outcomes.

- **Input at inference time:** one student's behavioral + demographic fields
  (see Section 10, "Example Input and Output," for the exact schema).
- **Output:** a cluster ID (0–3), a human-readable learner-profile name
  (e.g. "At-Risk Learners"), a recommended intervention, and a confidence note.

## 5. Project Pipeline / System Architecture

```
data/xAPI-Edu-Data.csv
        │
        ▼
 src/data.py            → load + schema validation + data-quality checks
        │
        ▼
 src/preprocessing.py   → 80/20 train/test split (random, fixed seed)
                           ColumnTransformer: OneHotEncoder (categorical)
                                              + StandardScaler (behavioral)
                           fit ONLY on train
        │
        ▼
 src/train.py            → naive baseline (random labels)
                           → simple baseline (K-Means, k=2)
                           → K-Means grid k=2..8 (Elbow + Silhouette), logged to MLflow
                           → Agglomerative (Ward) and DBSCAN comparisons, logged to MLflow
                           → 5-seed stability check (Adjusted Rand Index)
                           → final model selection (k=4, justified in reports/results.md)
                           → saves models/*.joblib + models/model_metadata.json
                           → saves reports/figures/*.png
        │
        ▼
 src/evaluate.py         → loads saved artifacts
                           → evaluates on the held-out 20% test split ("unseen students")
                           → error analysis: borderline / poorly-fit students
                           → saves reports/test_evaluation.json, reports/borderline_students_test.csv
        │
        ▼
 src/predict.py          → predict_student_cluster(dict) -> cluster + name + recommendation
                           → input validation (missing fields, wrong types, out-of-range values)
        │
        ▼
 demo.ipynb (Colab)      → clones/installs, runs the pipeline end-to-end,
                           demonstrates valid AND invalid inference inputs
```

Directory layout:

```
capstone-project/
├── README.md                      <- this file
├── requirements.txt
├── .gitignore
├── FIELD-BASED_CAPSTONE_SCENARIO.pdf   <- official scenario brief
├── demo.ipynb                      <- Colab-first, reproducible, end-to-end demo
├── data/
│   ├── xAPI-Edu-Data.csv
│   └── README.md                   <- dataset documentation
├── notebooks/
│   ├── 01_eda.ipynb                <- exploratory data analysis
│   └── 02_experiments.ipynb        <- full modeling/experiment walkthrough
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── preprocessor.joblib
│   ├── final_kmeans_model.joblib
│   ├── pca_projector.joblib
│   ├── model_metadata.json
│   └── README.md
├── reports/
│   ├── results.md                  <- full write-up of every experiment and metric
│   ├── experiment_results.csv
│   ├── test_evaluation.json
│   ├── borderline_students_test.csv
│   └── figures/
│       ├── elbow_plot.png
│       ├── silhouette_vs_k.png
│       └── pca_clusters_train.png
├── mlflow.db                       <- (generated locally by src/train.py; not committed)
├── mlruns/                         <- (generated locally by src/train.py; not committed)
└── submission/
    └── Submission_Details.docx     <- LMS submission file
```

## 6. Models / Approaches Tested

| Model | Role |
|---|---|
| Random cluster assignment | Naive baseline — confirms real structure exists (near-zero silhouette) |
| K-Means, k=2 | Simple baseline |
| K-Means, k=2..8 | Main experiment grid (Elbow Method + Silhouette Score) |
| Agglomerative Clustering (Ward linkage), k=4 | Comparison approach |
| DBSCAN, eps grid 0.5–3.0 | Comparison approach |
| **K-Means, k=4 (final)** | **Selected final model** |

All experiments are logged to MLflow (SQLite backend at `mlflow.db`, model
artifacts under `mlruns/`) whenever `python -m src.train` is run. These two
are **not** committed to the repository — every run creates a fresh,
timestamped model snapshot under `mlruns/`, so committing it would mean an
ever-growing pile of near-duplicate folders. Instead:
- **Evidence of every experiment** (parameters + metrics for every model
  tried) is committed in the human-readable
  [`reports/experiment_results.csv`](reports/experiment_results.csv).
- To inspect the live MLflow UI yourself, just re-run `python -m src.train`
  locally, then: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## 7. Final Model and Justification

**Final model: K-Means, k=4, random_state=42.**

This is not the numerically "best" clustering by silhouette score alone —
that would be k=2 (silhouette 0.200 vs. 0.136 at k=4). k=4 was deliberately
selected because:

1. The scenario brief's functional requirements and expected deliverables
   explicitly call for **four named, actionable learner profiles** (Highly
   Engaged / Consistent / Struggling but Active / At-Risk Learners) that
   educators can act on — a 2-cluster split is too coarse to route four
   different intervention strategies.
2. K-Means is the only one of the three algorithms compared that supports
   `.predict()` on brand-new, unseen students without retraining — essential
   for the "assign a new student" requirement in the brief.
3. It is highly stable: mean pairwise Adjusted Rand Index of **0.926** across
   5 different random seeds.
4. Agglomerative Clustering at k=4 scored lower (silhouette 0.109 vs. 0.136).
   DBSCAN could not find a usable global clustering at any tested density
   threshold — it either found 0 clusters or marked the large majority of
   students as noise (see `reports/results.md`, Section 3, for the exact numbers).

Full quantitative justification, including the honest disclosure that k=2 has
the best pure silhouette score, is in [`reports/results.md`](reports/results.md).

## 8. Evaluation Metrics and Results

| Metric | Train (n=384) | Test / unseen (n=96) |
|---|---|---|
| Silhouette Score | 0.1358 | 0.1118 |
| Calinski-Harabasz Index | 63.10 | 14.33 |
| Davies-Bouldin Index | 2.269 | 2.395 |

- **Silhouette Score** measures how well-separated and internally cohesive
  clusters are (higher is better); it is the primary metric because it works
  without ground-truth labels, exactly as the clustering task requires.
- **Calinski-Harabasz** and **Davies-Bouldin** are reported as secondary,
  corroborating internal-validation metrics.
- **Stability**: mean pairwise Adjusted Rand Index across 5 seeds = **0.926**.
- **Post-hoc validation against `Class`** (never used in training): the
  At-Risk Learners cluster contains 0% high-performing and the highest share
  of low-performing students on the unseen test split; Highly Engaged and
  Consistent Learners contain 0% low-performing students. Full crosstab in
  `reports/results.md`, Section 6.

Full breakdown, all experiment numbers, and the complete cluster-profile
table are in **[`reports/results.md`](reports/results.md)** — read this file
for the complete evaluation write-up.

## 9. Installation Instructions

```bash
git clone <PASTE-YOUR-GITHUB-REPO-URL-HERE>
cd capstone-project
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## 10. Training Instructions

Everything below is also demonstrated, cell by cell, in `demo.ipynb` and
`notebooks/02_experiments.ipynb` — you do not have to use the command line if
you prefer Colab.

```bash
# From the repository root:
python -m src.train      # fits preprocessing + all clustering models,
                          # logs every experiment to MLflow, saves the
                          # final model + figures + metadata
python -m src.evaluate   # evaluates the saved final model on the held-out
                          # test split and runs the error analysis
```

Both scripts print their results to the console and also save them to
`reports/` and `models/` so nothing is lost if you close the terminal.

To inspect the logged MLflow experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 11. Demo and Inference Run Instructions (Colab-first)

Open **`demo.ipynb`** in Google Colab (or Jupyter). It is self-contained: it
installs dependencies, loads the dataset and saved model directly from this
repository, and demonstrates:

1. Loading the saved preprocessing pipeline and final K-Means model.
2. Assigning a **new, previously unseen** student to a learner segment.
3. Rejecting an **invalid** student record (missing field, wrong type,
   out-of-range value) with a clear error message instead of crashing.
4. Visualizing the four discovered learner segments in PCA space.

No application or API layer is used for this project — the Colab notebook is
the demo/inference interface, as permitted by the scenario brief.

## 12. Example Input and Output

**Input** (a dict passed to `src.predict.predict_student_cluster`):

```python
{
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
```

**Output:**

```python
{
    "cluster_id": 2,
    "cluster_name": "Consistent Learners",
    "recommendation": "Maintain regular learning support and encourage continued engagement.",
    "confidence_note": "Moderate confidence in this cluster assignment."
}
```

**Invalid input example** (`raisedhands` as a string instead of a number):

```python
predict_student_cluster({..., "raisedhands": "a lot"})
# Raises: InvalidStudentInputError: Field 'raisedhands' must be numeric
# (0-100 scale in the original dataset), got 'a lot' (<class 'str'>).
```

## 13. Known Limitations

- **k=4 is a business-interpretability choice, not the statistically
  tightest clustering** — silhouette score is actually highest at k=2 on this
  dataset. This is disclosed, not hidden (see `reports/results.md`).
- **Small dataset**: 480 students from one platform; clusters may not
  generalize to other institutions or LMS platforms.
- **DBSCAN did not find a usable global clustering** of this feature space at
  any tested density threshold — most points were labeled noise. This
  suggests the behavioral feature space does not have naturally
  density-separated clusters (K-Means/Agglomerative's centroid-based
  assumption fits this data better than density-based clustering).
- **Borderline students exist**: on the held-out test set, 9.4% of students
  have a negative per-sample silhouette (poorly fit to their assigned
  cluster), and 10 students are essentially equidistant between two clusters.
  `predict_student_cluster` surfaces this via a `confidence_note` rather than
  presenting every assignment as equally certain.
- **Hard cluster assignment**: K-Means gives every student exactly one label,
  even though real learning behavior is a spectrum. A soft-clustering method
  (e.g. Gaussian Mixture Models) could represent partial membership, and is a
  natural direction for future work.

## 14. Responsible AI Considerations

- **Bias / fairness:** demographic fields (`gender`, `NationalITy`) are
  included as clustering inputs because they describe the student, but they
  are never used, on their own, to imply ability, intelligence, or future
  potential. The discovered clusters are **behavioral profiles**, not
  judgments about a student's worth or ceiling — this is stated explicitly so
  the tool is not misapplied. Cluster composition by gender/nationality
  should be reviewed periodically by educators before clusters are used
  operationally, to check no demographic group is disproportionately
  concentrated in the "At-Risk" segment for reasons unrelated to genuine
  behavior (e.g., systemic access differences).
- **Privacy:** the dataset is fully anonymized, publicly available under
  CC BY-SA 4.0, and contains no personally identifiable information. No new
  personal data is collected by this project.
- **Appropriate use:** cluster assignments are meant to **support** an
  educator's judgment (e.g., flag a student for outreach), never to
  **replace** it or to make automated decisions about grading, admission, or
  discipline. The `Class` academic-performance label is used only to
  interpret clusters after the fact — it is never fed into the model, and the
  model's output should never be reframed as a prediction of a student's
  grade.
- **Misuse risk:** this is a prototype trained on 480 students from one
  platform; deploying it operationally on a different, larger, or more
  diverse student population without re-validating cluster quality and
  fairness would be inappropriate.

## 15. Colab Setup Troubleshooting

`demo.ipynb` supports two ways to get the project into Colab:

- **Option A (recommended):** run the first code cell as-is — it prompts you
  to upload `capstone-project.zip` directly and extracts it in the Colab
  session. No GitHub required.
- **Option B:** if you have a working GitHub repo, comment in the `git clone`
  line in the second code cell instead.

**Common cause of errors:** uploading files to GitHub one at a time (or
dragging loose files instead of the whole folder) silently drops the
`src/`, `data/`, `models/`, and `reports/` subfolder structure, and can even
cause same-named files (e.g. multiple `requirements.txt`) to overwrite each
other. If that happens, `demo.ipynb`'s Step 2 ("Auto-repair the folder
layout") detects flattened files and reconstructs the correct structure
automatically — you do not need to manually fix your repo for the demo to
run. To fix the GitHub repo itself for submission, delete its contents and
re-upload by **dragging the entire extracted `capstone-project` folder**
(not individual files) onto GitHub's "Add file → Upload files" drop zone, or
push it with `git` from a local clone.

## 16. Reproducibility Notes

- All random seeds are fixed (`random_state=42` for the train/test split and
  the final model; stability is additionally checked across seeds
  `[0, 1, 2, 3, 42]`).
- The train/test split is a **plain random 80/20 split**: each row is one
  independent student record with no time ordering or repeated-user
  structure to respect, so a random split is the logically correct and
  simplest strategy here.
- No leakage: the `Class` outcome column and the redundant `PlaceofBirth`
  column are dropped by the `ColumnTransformer` itself (`remainder="drop"`)
  before any model ever sees the data; the preprocessing pipeline (encoder +
  scaler) is fit **only** on the train split and only ever `.transform()`-ed
  on the test split.
- Every script (`src/train.py`, `src/evaluate.py`, `src/predict.py`) can be
  re-run from a clean clone of this repository with no hidden local state —
  the only inputs are `data/xAPI-Edu-Data.csv` (included) and the Python
  packages in `requirements.txt`.
