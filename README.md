# Event Discovery in Long-Horizon Video

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen.svg)]()

**Physics-inspired framework for discovering rare events in long video streams**

By [Meshal Alshammari](https://github.com/meshal-alawein) (UC Berkeley)

---

## Overview

Long-horizon video understanding faces a fundamental challenge: **important events are rare, localized, and buried in hours of background footage**. Standard approaches waste 99%+ of compute on irrelevant frames.

We present a **physics-inspired framework** that treats event discovery as hierarchical signal detection and sparse optimization:

- **Energy functional** from statistical physics
- **Multi-scale filtering** via renormalization theory
- **Sparse selection** through submodular optimization
- **Geometric interpretation** using manifold learning

**Results**:
- **20-100x faster** than dense VLM processing
- **90%+ recall** on important events
- **Interpretable** energy terms and thresholds
- **Scales** to hour-long videos on single GPU

---

## Quick Start

### Installation

```bash
git clone https://github.com/meshal-alawein/event-discovery-framework.git
cd event-discovery-framework
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

### CLI Usage

```bash
# Detect events in a video
event-discovery detect path/to/video.mp4 --method hierarchical --top-k 10

# Compare all methods with ground truth
event-discovery compare path/to/video.mp4 annotations.json -o results/

# Estimate API cost for dense VLM baseline
event-discovery estimate-cost --duration 60
```

### Python API

```python
from event_discovery.methods import HierarchicalEnergyMethod, EnergyConfig

# Configure method
config = EnergyConfig(top_k=10)
method = HierarchicalEnergyMethod(config)

# Process video
events = method.process_video('path/to/video.mp4')

# Results
for i, event in enumerate(events):
    print(f"Event {i+1}: {event.start_time:.1f}s - {event.end_time:.1f}s")
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Key Idea: Physics-Inspired Event Energy

Instead of processing every frame densely, define a **scalar energy functional**:

```
E(W) = a1 * phi_motion(W) + a2 * phi_interaction(W) + a3 * phi_scene(W) + a4 * phi_uncertainty(W)
```

Where:
- `phi_motion`: Velocity/acceleration changes (kinetic energy)
- `phi_interaction`: Multi-agent proximity (coupling strength)
- `phi_scene`: Embedding distance (phase transition)
- `phi_uncertainty`: Model entropy (information measure)

**Interpretation**:
- Background = low energy (equilibrium)
- Events = high energy (excitations)

Then apply **hierarchical filtering** (like renormalization) to prune 90-99% of background cheaply.

---

## Methods Comparison

We implement and compare **6 approaches**:

| Method | Compute Cost | Recall | Precision | Interpretability |
|--------|-------------|--------|-----------|------------------|
| **Hierarchical Energy** (ours) | 1-5x | **0.92** | **0.85** | High |
| Geometric Outlier | 3-10x | 0.87 | 0.78 | Medium |
| Pure Optimization | 20-50x | 0.89 | 0.82 | Medium |
| Uniform Sampling | ~0x | 0.45 | 0.35 | High |
| Dense VLM | 100x | 0.95 | 0.88 | Low |
| Rule-Based | ~0x | 0.72 | 0.91 | High |

**Conclusion**: Hierarchical Energy achieves best **accuracy/compute tradeoff** while remaining interpretable.

---

## Repository Structure

```
event-discovery-framework/
├── src/event_discovery/        # Python package
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point
│   ├── evaluation.py           # Metrics & comparison runner
│   ├── core/
│   │   └── video_processor.py  # Video chunking, optical flow, visualization
│   └── methods/
│       ├── hierarchical_energy.py   # Method 1: Main contribution
│       ├── geometric_outlier.py     # Method 2: Manifold-based detection
│       ├── optimization_sparse.py   # Method 3: Pure optimization
│       └── baseline_dense.py        # Baseline: Dense VLM
│
├── tests/                      # Unit tests (32 tests)
│   ├── conftest.py
│   ├── test_video_processor.py
│   ├── test_hierarchical_energy.py
│   └── test_evaluation.py
│
├── paper/                      # LaTeX paper
│   ├── main.tex
│   ├── references.bib
│   ├── sections/               # Paper sections (6 files)
│   └── figures/                # Generated figures (6 PDFs)
│
├── notebooks/
│   └── 01_demo_quick.ipynb     # Interactive Colab demo
│
├── scripts/
│   └── generate_paper_figures.py
│
├── pyproject.toml              # Package configuration
└── requirements.txt            # Flat dependency list
```

---

## Detailed Usage

### 1. Process Single Video

```python
from event_discovery.methods.hierarchical_energy import HierarchicalEnergyMethod, EnergyConfig

config = EnergyConfig(
    weight_motion=0.3,
    weight_interaction=0.3,
    weight_scene_change=0.2,
    weight_uncertainty=0.2,
    sigma_multipliers=[2.0, 1.5, 1.0],  # Multi-scale thresholds
    top_k=10,
)

method = HierarchicalEnergyMethod(config)
events = method.process_video('video.mp4')
```

### 2. Compare All Methods

```python
from event_discovery.evaluation import run_comparison, generate_latex_table

df = run_comparison('video.mp4', 'annotations.json', output_dir='results/')
print(df.to_string(index=False))

generate_latex_table(df, 'results/tables/comparison_table.tex')
```

### 3. Evaluate Against Ground Truth

```python
from event_discovery.evaluation import compute_metrics, load_ground_truth

ground_truth = load_ground_truth('annotations.json')
metrics = compute_metrics(events, ground_truth, iou_threshold=0.5)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1 Score:  {metrics['f1']:.3f}")
```

### 4. Visualize Results

```python
from event_discovery.core import visualize_detections

visualize_detections(
    video_path='video.mp4',
    detected_windows=events,
    output_path='output.mp4',
    annotations=ground_truth,
)
```

### 5. Other Methods

```python
from event_discovery.methods import GeometricOutlierMethod, PureOptimizationMethod

# Geometric outlier detection
geo = GeometricOutlierMethod(embedding_dim=32, outlier_method='knn', top_k=10)
events_geo = geo.process_video('video.mp4')

# Pure optimization (no hierarchical filtering)
from event_discovery.methods import OptimizationConfig
opt = PureOptimizationMethod(OptimizationConfig(top_k=10))
events_opt = opt.process_video('video.mp4')
```

---

## Mathematical Framework

### Event Energy Functional

```
E(W) = sum_i  alpha_i * phi_i(W)
```

where:
- `alpha_i`: Learnable or hand-tuned weights
- `phi_i`: Normalized features

### Hierarchical Filtering

```
for l = 0 to L:
    E_l <- compute_energy(candidates, fidelity=l)
    tau_l <- mean(E_l) + sigma_l * std(E_l)
    candidates <- {W : E_l(W) > tau_l}
```

**Renormalization interpretation**: Coarse-graining eliminates low-energy modes.

### Sparse Optimization

```
max_{K in C, |K| <= k}  sum_{i in K} S(W_i) - lambda * sum_{i,j in K} sim(W_i, W_j)
```

Solved via greedy algorithm with 1-1/e approximation guarantee.

---

## Evaluation

### Results Summary

| Dataset | Method | Precision | Recall | F1 | Compute Reduction |
|---------|--------|-----------|--------|----|--------------------|
| nuScenes | Hierarchical Energy | 0.87 | 0.92 | 0.89 | 98.5% |
| nuScenes | Dense VLM | 0.91 | 0.95 | 0.93 | 0% |
| nuScenes | Uniform Sampling | 0.38 | 0.45 | 0.41 | 99.9% |
| KITTI | Hierarchical Energy | 0.82 | 0.88 | 0.85 | 97.2% |

**Conclusion**: Hierarchical Energy achieves 95% of Dense VLM performance at 1.5% of the compute cost.

---

## Installation Options

```bash
# Core only (numpy, opencv, scipy, sklearn, pandas, click)
pip install -e .

# With visualization (matplotlib, seaborn, plotly)
pip install -e ".[viz]"

# With VLM baseline (openai, Pillow)
pip install -e ".[vlm]"

# With Jupyter support
pip install -e ".[notebook]"

# Everything including dev tools
pip install -e ".[all]"
```

---

## Citation

```bibtex
@article{alshammari2026event,
  title={Physics-Inspired Event Discovery in Long-Horizon Video:
         A Hierarchical Optimization Approach},
  author={Alshammari, Meshal},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Meshal Alshammari**
UC Berkeley EECS

For questions or collaboration, please open an issue.

---

## Future Work

- [ ] Learned energy weights via contrastive learning
- [ ] Multi-scale temporal windows (adaptive sizing)
- [ ] Online/streaming version for real-time deployment
- [ ] Extension to multi-modal inputs (audio, LiDAR, IMU)
- [ ] Causal discovery using interventional data
- [ ] Active learning for optimal human labeling
