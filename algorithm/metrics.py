"""Evaluation metrics. scikit-learn is used here only for evaluation (K-means, NMI)."""

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score


def relative_reconstruction_error(V_clean, W, H):
    """RRE = ||V_clean - WH||_F / ||V_clean||_F."""
    return np.linalg.norm(V_clean - W @ H) / np.linalg.norm(V_clean)


def assign_cluster_label(X, Y, seed):
    """K-means on the rows of X; each cluster takes its majority true label."""
    kmeans = KMeans(n_clusters=len(set(Y)), n_init=10, random_state=seed).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
    return Y_pred


def clustering_scores(H, Y, seed):
    """Accuracy and NMI of K-means clusters computed on the columns of H."""
    Y_pred = assign_cluster_label(H.T, Y, seed)
    return accuracy_score(Y, Y_pred), normalized_mutual_info_score(Y, Y_pred)
