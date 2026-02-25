"""
Method 2: Geometric Outlier Detection
Embed windows into low-dimensional space and detect manifold deviations.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ..core.base import BaseEventDetector
from ..core.features import compute_color_histogram
from ..core.video_processor import VideoWindow

logger = logging.getLogger(__name__)


class GeometricOutlierMethod(BaseEventDetector):
    """
    Geometric outlier-based event discovery.

    Key idea: Events correspond to deviations from low-dimensional manifold.

    Steps:
    1. Embed windows: W_i -> z_i in R^d
    2. Detect outliers via distance, local density, or trajectory curvature
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        outlier_method: str = "knn",
        k_neighbors: int = 10,
        top_k: int = 10,
        histogram_bins: int = 32,
    ):
        super().__init__(top_k=top_k, diversity_weight=0.0)
        self.embedding_dim = embedding_dim
        self.outlier_method = outlier_method
        self.k_neighbors = k_neighbors
        self.histogram_bins = histogram_bins
        self.pca: Optional[PCA] = None

    def _score_windows(self, windows: list[VideoWindow]) -> np.ndarray:
        """Embed windows and compute outlier scores."""
        embeddings = self.embed_windows(windows)
        logger.info("Embedded windows into %d-D space", embeddings.shape[1])
        return self.compute_outlier_scores(embeddings)

    def _select(self, windows: list[VideoWindow], scores: np.ndarray) -> list[VideoWindow]:
        """Select top-k by score (no diversity penalty for geometric method)."""
        top_indices = np.argsort(scores)[-self.top_k:][::-1]
        return [windows[i] for i in top_indices]

    def embed_windows(self, windows: list[VideoWindow]) -> np.ndarray:
        """
        Embed windows into low-dimensional space using PCA on color histograms.

        Returns:
            Embeddings array of shape (N, embedding_dim)
        """
        features = []
        for window in windows:
            histograms = [
                compute_color_histogram(frame, bins=self.histogram_bins)
                for frame in window.frames
            ]
            mean_hist = np.mean(histograms, axis=0)
            features.append(mean_hist)

        features_array = np.stack(features)

        n_components = min(
            self.embedding_dim, features_array.shape[0], features_array.shape[1]
        )
        self.pca = PCA(n_components=n_components)
        return self.pca.fit_transform(features_array)

    def compute_outlier_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute outlier scores. Higher = more outlier-like."""
        if self.outlier_method == "distance":
            return self._distance_outlier(embeddings)
        elif self.outlier_method == "knn":
            return self._knn_outlier(embeddings)
        elif self.outlier_method == "curvature":
            return self._curvature_outlier(embeddings)
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")

    def _distance_outlier(self, embeddings: np.ndarray) -> np.ndarray:
        """Distance from mean embedding."""
        mean = np.mean(embeddings, axis=0)
        return np.linalg.norm(embeddings - mean, axis=1)

    def _knn_outlier(self, embeddings: np.ndarray) -> np.ndarray:
        """k-NN density estimation. Higher avg distance = more outlier-like."""
        k = min(self.k_neighbors, len(embeddings) - 1)
        if k < 1:
            return np.zeros(len(embeddings))

        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(embeddings)

        distances, _ = nbrs.kneighbors(embeddings)
        return np.mean(distances[:, 1:], axis=1)

    def _curvature_outlier(self, embeddings: np.ndarray) -> np.ndarray:
        """Trajectory curvature: angle between consecutive embedding vectors."""
        scores = np.zeros(len(embeddings))

        for i in range(1, len(embeddings) - 1):
            v1 = embeddings[i] - embeddings[i - 1]
            v2 = embeddings[i + 1] - embeddings[i]

            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_product < 1e-9:
                continue

            cos_angle = np.dot(v1, v2) / norm_product
            scores[i] = np.arccos(np.clip(cos_angle, -1, 1))

        if len(embeddings) > 2:
            scores[0] = scores[1]
            scores[-1] = scores[-2]

        return scores


def visualize_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    outlier_scores: np.ndarray,
    save_path: str = None,
):
    """Visualize 2D projection of embedding space."""
    import matplotlib.pyplot as plt

    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    scatter = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=labels, cmap="coolwarm", s=50, alpha=0.6,
    )
    ax.set_title("Ground Truth Labels", fontsize=14, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.colorbar(scatter, ax=ax, label="Event (1) / Normal (0)")

    ax = axes[1]
    scatter = ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=outlier_scores, cmap="viridis", s=50, alpha=0.6,
    )
    ax.set_title("Outlier Scores", fontsize=14, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.colorbar(scatter, ax=ax, label="Outlier Score")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved embedding visualization to %s", save_path)

    plt.show()
