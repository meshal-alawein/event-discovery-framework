# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-25

### Added

- **Core framework**: `BaseEventDetector` abstract class implementing Template Method pattern
  — all detectors share chunk → score → select pipeline
- **Hierarchical Energy method** (`HierarchicalEnergyMethod`): Physics-inspired event
  detection with multi-scale filtering based on renormalization group theory
  - `EnergyConfig` dataclass for configuring weights and thresholds
  - Adaptive thresholding: `tau_l = mean(E) + sigma_l * std(E)`
  - Greedy diverse selection with temporal similarity penalty
- **Geometric Outlier method** (`GeometricOutlierMethod`): Manifold-based detection
  - PCA embedding of color histogram features
  - Three outlier modes: `distance`, `knn`, `curvature`
  - Embedding visualization utility (`visualize_embedding_space`)
- **Pure Optimization method** (`PureOptimizationMethod`): Direct sparse selection
  without hierarchical filtering, for ablation comparison
  - `OptimizationConfig` with weight normalization
- **Dense VLM baseline** (`DenseVLMMethod`): GPT-4V oracle baseline
  - OpenAI API integration with structured prompting
  - Cost estimation utility (`estimate_cost`)
- **Shared feature utilities** (`core/features.py`):
  - `compute_color_histogram` — normalized RGB histogram
  - `compute_edge_density_variance` — Canny edge proxy for interaction
  - `compute_pixel_variance` — uncertainty proxy
  - `compute_pixel_entropy` — principled uncertainty measure
  - `normalize_features_batch` — z-score normalization across batch
  - `temporal_similarity` — exponential decay similarity
  - `greedy_diverse_select` — submodular maximization with 1-1/e guarantee
- **Video processing pipeline** (`core/video_processor.py`):
  - `VideoWindow` dataclass: temporal window with frames and metadata
  - `VideoProcessor`: sliding window chunking with configurable size/stride
  - Farneback dense optical flow computation
  - Window downsampling for embedding
  - `visualize_detections`: annotated output video generation
- **CLI** (`cli.py`): Click-based command-line interface
  - `detect`: process a video file with configurable method and top-k
  - `compare`: run all methods with ground truth evaluation
  - `estimate-cost`: estimate VLM API cost for given video duration
- **Evaluation** (`evaluation.py`):
  - `compute_metrics`: precision/recall/F1 with temporal IoU matching
  - `temporal_iou`: intersection-over-union for temporal windows
  - `load_ground_truth`: robust JSON annotation loader
  - `run_comparison`: automated multi-method benchmark
  - `generate_latex_table`: publication-ready table generation
  - Baseline implementations: `baseline_uniform_sampling`, `baseline_rule_based`
- **Test suite** (32 tests across 3 modules):
  - `test_hierarchical_energy.py`: EnergyConfig, HierarchicalEnergyMethod, features
  - `test_evaluation.py`: metrics computation, IoU, annotation loading
  - `test_video_processor.py`: VideoWindow, VideoProcessor, optical flow
- **Research paper** (LaTeX, NeurIPS format):
  - Complete 6-section paper with introduction, related work, methods,
    experiments, results, and conclusion
  - 30+ references in BibTeX
  - 6 publication-quality PDF figures
- **Interactive demo** (`notebooks/01_demo_quick.ipynb`): Google Colab compatible
- **Figure generation** (`scripts/generate_paper_figures.py`): Reproducible figures
- **Documentation**:
  - Professional README with Quick Start, API reference, math framework
  - `IMPLEMENTATION_GUIDE.md`: Detailed step-by-step guide
  - `CONTRIBUTING.md`: Contribution workflow and standards
  - `SECURITY.md`: Vulnerability reporting policy
  - `AGENTS.md`: AI agent governance rules
  - GitHub issue templates (bug report, feature request)
  - Dependabot configuration for security updates

### Performance

- 20–100× faster than dense VLM processing
- 90%+ recall on important events with 98.5% compute reduction (nuScenes)
- Greedy diverse selection: 1-1/e approximation guarantee (O(n·k) time)

### Infrastructure

- `pyproject.toml`: setuptools build with optional dependency groups
  (`viz`, `vlm`, `notebook`, `dev`, `all`)
- `.github/workflows/ci.yml`: Full CI pipeline (lint → type-check → tests)
- `.pre-commit-config.yaml`: Pre-commit hooks (ruff, mypy)
- `dependabot.yml`: GitHub Actions and pip dependency updates

[0.1.0]: https://github.com/meshal-alawein/event-discovery-framework/releases/tag/v0.1.0
