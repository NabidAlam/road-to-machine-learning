"""Module 08 exercise checks. Tiny synthetic data only. No downloads."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def make_toy(seed=7):
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.45, random_state=seed)
    return X


def exercise_1_kmeans_labels(X, k=3):
    """Fit KMeans(k). Return labels."""
    raise NotImplementedError


def exercise_2_n_clusters(labels):
    """Return number of unique cluster ids."""
    raise NotImplementedError


def exercise_3_silhouette(X, labels):
    """Return silhouette score."""
    raise NotImplementedError


def exercise_4_best_k(X, ks=(2, 3, 4)):
    """Return k in ks with the highest silhouette."""
    raise NotImplementedError


def exercise_5_inertia(X, k=3):
    """Return KMeans inertia for k clusters."""
    raise NotImplementedError


def exercise_6_label_length(X, k=3):
    """Return length of the label vector (must equal n_samples)."""
    raise NotImplementedError


def run_checks(ns):
    X = make_toy()
    labels = ns["exercise_1_kmeans_labels"](X, k=3)
    assert labels.shape[0] == X.shape[0]
    assert ns["exercise_2_n_clusters"](labels) == 3
    sil = ns["exercise_3_silhouette"](X, labels)
    assert sil > 0.4
    assert ns["exercise_4_best_k"](X, ks=(2, 3, 4)) == 3
    inertia = ns["exercise_5_inertia"](X, k=3)
    assert inertia > 0
    assert ns["exercise_6_label_length"](X, k=3) == 120
    print("module08 exercises OK")


if __name__ == "__main__":
    print("Implement solutions in solutions.py then run that file.")
