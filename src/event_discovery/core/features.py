"""
Shared feature computation utilities.

Provides common feature extraction functions used across multiple
detection methods, eliminating code duplication.
"""


import cv2
import numpy as np


def compute_color_histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Compute normalized RGB color histogram for a single frame.

    Args:
        frame: BGR image array of shape (H, W, 3)
        bins: Number of histogram bins per channel

    Returns:
        Normalized histogram of shape (3 * bins,)
    """
    hist_r = np.histogram(frame[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(frame[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(frame[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float64)
    return hist / (hist.sum() + 1e-6)


def compute_edge_density_variance(frames: np.ndarray, low: int = 100, high: int = 200) -> float:
    """
    Compute variance of edge density across frames.

    Used as a proxy for interaction/object-activity changes.

    Args:
        frames: Array of shape (T, H, W, 3) in BGR format
        low: Canny low threshold
        high: Canny high threshold

    Returns:
        Standard deviation of per-frame edge pixel counts
    """
    edge_counts = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        edge_counts.append(np.sum(edges > 0))
    return float(np.std(edge_counts))


def compute_pixel_variance(frames: np.ndarray) -> float:
    """
    Compute mean per-frame pixel variance.

    Used as a proxy for visual uncertainty/information content.

    Args:
        frames: Array of shape (T, H, W, 3)

    Returns:
        Mean variance across frames
    """
    variances = [float(np.var(frame)) for frame in frames]
    return float(np.mean(variances))


def compute_pixel_entropy(frames: np.ndarray, bins: int = 256) -> float:
    """
    Compute mean per-frame pixel entropy.

    A more principled uncertainty measure than pixel variance.

    Args:
        frames: Array of shape (T, H, W, 3)
        bins: Number of histogram bins for entropy computation

    Returns:
        Mean entropy across frames
    """
    entropies = []
    for frame in frames:
        hist, _ = np.histogram(frame.flatten(), bins=bins, range=(0, 256))
        hist = hist / (hist.sum() + 1e-9)
        entropy = -np.sum(hist * np.log(hist + 1e-9))
        entropies.append(entropy)
    return float(np.mean(entropies))


def normalize_features_batch(
    raw_features: list[dict[str, float]],
) -> list[dict[str, float]]:
    """
    Z-score normalize features across a batch.

    Args:
        raw_features: List of feature dicts, each mapping name -> raw value

    Returns:
        List of feature dicts with z-score normalized values
    """
    if not raw_features:
        return raw_features

    keys = raw_features[0].keys()
    normalized = [{} for _ in raw_features]

    for key in keys:
        values = np.array([f[key] for f in raw_features])
        mean_val = np.mean(values)
        std_val = np.std(values)
        for i, val in enumerate(values):
            normalized[i][key] = (
                (val - mean_val) / (std_val + 1e-6) if std_val > 1e-9 else 0.0
            )

    return normalized


def temporal_similarity(start_time_1: float, start_time_2: float, sigma: float = 10.0) -> float:
    """
    Compute temporal similarity between two windows.

    sim(w1, w2) = exp(-|t1 - t2| / sigma)

    Args:
        start_time_1: Start time of first window
        start_time_2: Start time of second window
        sigma: Decay constant in seconds

    Returns:
        Similarity score in [0, 1]
    """
    time_diff = abs(start_time_1 - start_time_2)
    return float(np.exp(-time_diff / sigma))


def greedy_diverse_select(
    candidates: list,
    scores: np.ndarray,
    top_k: int,
    diversity_weight: float = 0.5,
    sigma: float = 10.0,
) -> list:
    """
    Greedy selection with diversity penalty.

    Solves: max sum S(W_i) - lambda * sum sim(W_i, W_j)

    Args:
        candidates: List of VideoWindow objects
        scores: Array of scores for each candidate
        top_k: Number of items to select
        diversity_weight: Weight of the diversity penalty
        sigma: Temporal similarity decay constant

    Returns:
        List of selected candidates
    """
    if len(candidates) <= top_k:
        return list(candidates)

    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(top_k):
        if not remaining:
            break

        best_score = -np.inf
        best_idx = None

        for idx in remaining:
            score = scores[idx]

            if selected_indices:
                max_sim = max(
                    temporal_similarity(
                        candidates[idx].start_time,
                        candidates[sel_idx].start_time,
                        sigma=sigma,
                    )
                    for sel_idx in selected_indices
                )
                score -= diversity_weight * max_sim

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]
