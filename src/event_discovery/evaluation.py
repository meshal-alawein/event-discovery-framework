"""
Evaluation utilities: metrics computation and comparison.
"""

import json
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from .core.video_processor import VideoProcessor, VideoWindow
from .methods.hierarchical_energy import HierarchicalEnergyMethod, EnergyConfig
from .methods.geometric_outlier import GeometricOutlierMethod
from .methods.optimization_sparse import PureOptimizationMethod, OptimizationConfig

logger = logging.getLogger(__name__)


def load_ground_truth(annotation_path: str) -> List[dict]:
    """Load ground truth annotations."""
    with open(annotation_path, "r") as f:
        data = json.load(f)
    return data["events"]


def temporal_iou(gt_event: dict, detected: VideoWindow) -> float:
    """Compute intersection over union for temporal windows."""
    start = max(gt_event["start_time"], detected.start_time)
    end = min(gt_event["end_time"], detected.end_time)

    if start >= end:
        return 0.0

    intersection = end - start
    union = (
        (gt_event["end_time"] - gt_event["start_time"])
        + (detected.end_time - detected.start_time)
        - intersection
    )

    return intersection / union if union > 0 else 0.0


def compute_metrics(
    detected: List[VideoWindow],
    ground_truth: List[dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 score using temporal IoU matching.
    """
    tp = 0
    matched_gt = set()

    for det in detected:
        best_iou = 0
        best_gt_idx = None

        for i, gt in enumerate(ground_truth):
            if i not in matched_gt:
                iou = temporal_iou(gt, det)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx is not None:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(detected) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def baseline_uniform_sampling(
    video_path: str, sample_rate: int = 10, top_k: int = 10
) -> List[VideoWindow]:
    """Baseline: Uniform sampling. Sample every N-th window."""
    processor = VideoProcessor()
    windows = processor.chunk_video(video_path)
    sampled = [windows[i] for i in range(0, len(windows), sample_rate)]
    return sampled[:top_k]


def baseline_rule_based(
    video_path: str, top_k: int = 10
) -> List[VideoWindow]:
    """Baseline: Rule-based heuristics using optical flow magnitude."""
    processor = VideoProcessor()
    windows = processor.chunk_video(video_path)

    scores = []
    for window in windows:
        flow = processor.compute_optical_flow(window)
        flow_magnitude = np.sqrt(flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2)
        score = float(np.mean(flow_magnitude))
        scores.append(score)

    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [windows[i] for i in top_indices]


def run_comparison(
    video_path: str,
    annotation_path: str,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Run all methods and generate comparison table."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)

    ground_truth = load_ground_truth(annotation_path)
    logger.info("Loaded %d ground truth events", len(ground_truth))

    results = []

    # Method 1: Hierarchical Energy
    logger.info("Running Method 1: Hierarchical Energy")
    start = time.time()
    method1 = HierarchicalEnergyMethod(EnergyConfig(top_k=10))
    detected1 = method1.process_video(video_path)
    elapsed1 = time.time() - start
    metrics1 = compute_metrics(detected1, ground_truth)
    results.append({"method": "Hierarchical Energy", **metrics1, "time_sec": elapsed1})

    # Method 2: Geometric Outlier
    logger.info("Running Method 2: Geometric Outlier")
    start = time.time()
    method2 = GeometricOutlierMethod(embedding_dim=32, outlier_method="knn", top_k=10)
    detected2 = method2.process_video(video_path)
    elapsed2 = time.time() - start
    metrics2 = compute_metrics(detected2, ground_truth)
    results.append({"method": "Geometric Outlier", **metrics2, "time_sec": elapsed2})

    # Method 3: Pure Optimization
    logger.info("Running Method 3: Pure Optimization")
    start = time.time()
    method3 = PureOptimizationMethod(OptimizationConfig(top_k=10))
    detected3 = method3.process_video(video_path)
    elapsed3 = time.time() - start
    metrics3 = compute_metrics(detected3, ground_truth)
    results.append({"method": "Pure Optimization", **metrics3, "time_sec": elapsed3})

    # Baseline: Uniform Sampling
    logger.info("Running Baseline: Uniform Sampling")
    start = time.time()
    detected_uniform = baseline_uniform_sampling(video_path)
    elapsed_uniform = time.time() - start
    metrics_uniform = compute_metrics(detected_uniform, ground_truth)
    results.append(
        {"method": "Uniform Sampling", **metrics_uniform, "time_sec": elapsed_uniform}
    )

    # Baseline: Rule-Based
    logger.info("Running Baseline: Rule-Based")
    start = time.time()
    detected_rules = baseline_rule_based(video_path)
    elapsed_rules = time.time() - start
    metrics_rules = compute_metrics(detected_rules, ground_truth)
    results.append(
        {"method": "Rule-Based", **metrics_rules, "time_sec": elapsed_rules}
    )

    df = pd.DataFrame(results)
    csv_path = f"{output_dir}/tables/comparison_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    return df


def generate_latex_table(df: pd.DataFrame, output_path: str):
    """Generate LaTeX table from results DataFrame."""
    latex_table = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Comparison of Event Discovery Methods}\n"
        "\\label{tab:comparison}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Method & Precision & Recall & F1 & Time (s) \\\\\n"
        "\\midrule\n"
    )

    for _, row in df.iterrows():
        latex_table += (
            f"{row['method']} & {row['precision']:.3f} & {row['recall']:.3f} "
            f"& {row['f1']:.3f} & {row['time_sec']:.1f} \\\\\n"
        )

    latex_table += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex_table)

    logger.info("LaTeX table saved to %s", output_path)
