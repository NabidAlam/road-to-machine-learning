"""Module 08 solutions. Run as main to verify against tasks.run_checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tasks import run_checks


def exercise_1_kmeans_labels(X, k=3):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    return model.fit_predict(X)


def exercise_2_n_clusters(labels):
    return int(len(np.unique(labels)))


def exercise_3_silhouette(X, labels):
    return float(silhouette_score(X, labels))


def exercise_4_best_k(X, ks=(2, 3, 4)):
    best_k = None
    best_s = -2.0
    for k in ks:
        labels = exercise_1_kmeans_labels(X, k=k)
        s = silhouette_score(X, labels)
        if s > best_s:
            best_s = s
            best_k = k
    return int(best_k)


def exercise_5_inertia(X, k=3):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X)
    return float(model.inertia_)


def exercise_6_label_length(X, k=3):
    return int(len(exercise_1_kmeans_labels(X, k=k)))


if __name__ == "__main__":
    run_checks(globals())
