# Results & Interpretation — EDU-01 Student Learning Segmentation

All numbers on this page come directly from running `python -m src.train`
followed by `python -m src.evaluate` (see `models/model_metadata.json`,
`reports/experiment_results.csv`, and `reports/test_evaluation.json` for the
raw machine-readable versions of everything below).

Data split: 480 students → **384 train / 96 test** (80/20 random split,
`random_state=42`). The preprocessing pipeline (one-hot encoding + scaling)
and every clustering model were fit **only** on the 384-student train split;
the 96-student test split was held out and only ever `.transform()`-ed and
`.predict()`-ed, never used to fit anything.

## 1. Baselines

| Model | k | Silhouette (train) |
|---|---|---|
| Naive baseline — random cluster assignment | 4 | **-0.018** |
| Simple baseline — K-Means, minimal k | 2 | **0.200** |

The random baseline's silhouette score near zero confirms there is genuine,
non-trivial cluster structure in the behavioral data — a random assignment
finds none, while K-Means immediately finds a modest but real structure.

## 2. Model search: K-Means, k = 2 to 8

| k | Silhouette | Inertia | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|---|
| 2 | 0.2001 | 3082.5 | 110.78 | 1.812 |
| 3 | 0.1545 | 2789.9 | 81.03  | 2.220 |
| **4** | **0.1358** | 2654.2 | 63.10 | 2.269 |
| 5 | 0.1060 | 2533.4 | 53.97 | 2.565 |
| 6 | 0.1062 | 2443.7 | 47.42 | 2.374 |
| 7 | 0.1010 | 2388.1 | 41.79 | 2.333 |
| 8 | 0.0890 | 2329.6 | 37.97 | 2.610 |

See `reports/figures/elbow_plot.png` and `reports/figures/silhouette_vs_k.png`.

**Honest disclosure:** silhouette score decreases monotonically as k
increases on this dataset — the numerically best K-Means solution by
silhouette alone is **k=2** (0.2001), not k=4. This is reported transparently
rather than adjusted after the fact.

### Why k=4 was still selected as the final model

k=4 (silhouette 0.1358) is a **32.2% relative drop** from the k=2 optimum.
That drop is deliberately accepted because:

- The Capstone Brief's functional requirements and expected deliverables
  explicitly call for **four named, actionable learner profiles** (Highly
  Engaged / Consistent / Struggling but Active / At-Risk), not the minimum
  mathematically-tightest split.
- The brief's own criterion for cluster usefulness (Section 11, Q4) is
  interpretability and actionable interventions, not silhouette maximization
  alone: *"a cluster will be considered useful if ... the cluster can be
  clearly interpreted [and] educators can identify practical interventions."*
  A 2-cluster solution (e.g., roughly "engaged" vs. "disengaged") is
  mathematically cleaner but too coarse to route four different intervention
  strategies.
- The four discovered clusters (see Section 4 below) do map onto
  behaviorally and educationally distinct, actionable profiles, and their
  relationship to actual `Class` outcomes (never used in training) supports
  that they're capturing real signal, not noise.

This trade-off is a limitation, not a hidden flaw — see Section 6.

## 3. Comparison models (at k=4)

| Model | Silhouette (train) | Notes |
|---|---|---|
| K-Means (final) | 0.1358 | Selected — supports `.predict()` on new students |
| Agglomerative Clustering (Ward linkage) | 0.1085 | Slightly worse separation than K-Means; no native `.predict()` for new points |
| DBSCAN (best config found, eps=2.0, min_samples=5) | 0.378* | *Computed only on the ~15% of points DBSCAN didn't mark as noise — **326 of 384 train points (84.9%) were labeled noise** at this setting. At a looser eps=2.5, noise dropped to 106/384 (27.6%) but silhouette fell to 0.184 with only 3 clusters found. DBSCAN did not find a usable global clustering of this dataset at any tested eps (grid: 0.5–3.0); its high silhouette figure is misleading on its own because it excludes the vast majority of students. |

**Conclusion:** K-Means is both the best-separated *usable* clustering (once
DBSCAN's noise problem is accounted for) and the only one of the three that
supports assigning new, unseen students without retraining — this is why it
is the model saved to `models/final_kmeans_model.joblib`.

## 4. Cluster stability

The final K-Means (k=4) was re-fit with 5 different random seeds
(`[0, 1, 2, 3, 42]`). The mean pairwise Adjusted Rand Index (ARI) across all
seed pairs was **0.926**, indicating the four learner groups are highly
reproducible and not an artifact of a particular random initialization.

## 5. Final model performance: train vs. unseen test set

| Metric | Train (n=384) | Test / unseen (n=96) |
|---|---|---|
| Silhouette | 0.1358 | **0.1118** |
| Calinski-Harabasz | 63.10 | 14.33 |
| Davies-Bouldin | 2.269 | 2.395 |

Silhouette drops modestly (0.136 → 0.112) on students the model never saw
during fitting, which is expected and acceptable — the cluster structure
generalizes reasonably rather than collapsing on new data. The drop in
Calinski-Harabasz is expected too, since that metric scales with sample size
(96 vs 384 points) rather than only cluster quality.

## 6. Cluster profiles (learner segments)

Mean behavioral feature values per cluster, computed on the **train** split
(raw 0–100 activity-count scale):

| Cluster | raisedhands | VisitedResources | AnnouncementsView | Discussion |
|---|---|---|---|---|
| **Highly Engaged Learners** | 73.6 | 78.7 | 67.0 | 76.6 |
| **Consistent Learners** | 68.8 | 74.4 | 50.7 | 30.5 |
| **Struggling but Active Learners** | 53.2 | 79.2 | 34.4 | 30.8 |
| **At-Risk Learners** | 18.2 | 19.1 | 17.1 | 35.2 |

Post-hoc interpretation against `Class` (academic performance — **never**
used to fit the clusters), on the test / unseen split:

| Cluster | % Low class | % Middle class | % High class |
|---|---|---|---|
| At-Risk Learners | 62.5% | 37.5% | 0.0% |
| Consistent Learners | 0.0% | 63.2% | 36.8% |
| Highly Engaged Learners | 0.0% | 52.2% | 47.8% |
| Struggling but Active Learners | 7.1% | 64.3% | 28.6% |

This cross-tabulation is a strong qualitative validation signal: **At-Risk
Learners contain 0% high-performing students and the highest share of
low-performing students**, while Highly Engaged and Consistent Learners
contain 0% low-performing students. The clusters, discovered purely from
behavior, line up sensibly with actual academic outcomes without ever having
seen those outcomes during training.

**Note on "Struggling but Active Learners":** this group visits course
resources about as often as the Highly Engaged group (79.2 vs. 78.7) but
participates far less in live/interactive channels — raising hands,
announcements, and discussion are all noticeably lower. The name reflects
that this group is *actively consuming material* but not *actively
participating*, which is a distinct and actionable behavioral pattern (their
class-performance mix is, accordingly, the second-most positive of the four
groups — closer to Consistent Learners than to At-Risk).

## 7. Error analysis (on the unseen test set, n=96)

- **9 of 96 students (9.4%)** have a negative per-sample silhouette value,
  meaning they are on average closer to a *different* cluster's centroid
  than to their own — these are the weakest-fit assignments.
- **10 students** (bottom 10% by centroid-distance gap) are borderline: the
  distance to their assigned cluster's centroid and to the second-closest
  cluster's centroid differ by less than ~0.1 standard units. The full list
  is in `reports/borderline_students_test.csv`. For example, one student
  assigned to "At-Risk Learners" (raisedhands=20, VisitedResources=22,
  AnnouncementsView=53, Discussion=13) has almost identical distances to the
  At-Risk and a neighboring cluster's centroid — practically, this student's
  low participation is offset by unusually high announcement-viewing, making
  a single hard label an oversimplification of their behavior.
- These borderline cases are a natural and expected consequence of using a
  hard-assignment algorithm (K-Means) on real behavioral data that doesn't
  fall into perfectly separated groups; the `confidence_note` field returned
  by `src/predict.py::predict_student_cluster` flags exactly this situation
  for any new student with a small centroid gap, so downstream users (e.g. an
  educator dashboard) know to treat a borderline assignment as provisional
  rather than authoritative.

## 8. Limitations and assumptions (summary — see README for the full list)

- Silhouette score alone favors k=2; k=4 is a deliberate, documented
  interpretability trade-off, not the statistically tightest split.
- DBSCAN could not find a usable clustering of the full dataset at any tested
  eps — this dataset's behavioral feature space does not have DBSCAN-style
  density-separated clusters, at least not on the categorical+numeric feature
  encoding used here.
- 480 students from one platform is a small sample; discovered clusters may
  not transfer to a different institution's engagement patterns.
- Demographic features (gender, nationality) are included as clustering
  inputs but are never used to imply ability; clusters are behavioral
  profiles, and `Class` is only used post-hoc for interpretation, never for
  training or cluster assignment.
