import os
import tempfile

import joblib
import numpy as np
from sklearn.datasets import make_blobs

from entrenamiento.fenotipado import FenotipadoKMeans


def test_kmeans_assignments_length():
    X, _ = make_blobs(n_samples=150, centers=3, random_state=0)
    model = FenotipadoKMeans(n_clusters=3)
    labels = model.fit_predict(X)
    assert len(labels) == X.shape[0]


def test_silhouette_positive():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=0)
    model = FenotipadoKMeans(n_clusters=3)
    model.fit(X)
    score = model.silhouette(X)
    assert score > 0


def test_serializable():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=1)
    model = FenotipadoKMeans(n_clusters=3)
    model.fit(X)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
    tmp.close()
    try:
        joblib.dump(model, tmp.name)
        loaded = joblib.load(tmp.name)
        labels = loaded.predict(X)
        assert len(labels) == X.shape[0]
    finally:
        os.unlink(tmp.name)
