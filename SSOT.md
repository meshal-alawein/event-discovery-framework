---
type: normative
authority: canonical
audience: [agents, contributors, maintainers]
last-verified: 2026-02-25
---

# SSOT — Event Discovery Framework Authority

> **Status: Normative.** Do not modify without maintainer review.

This repository is the **single source of truth** for the
`event-discovery-framework` Python package.

## Canonical Location

| Artifact | Location |
|----------|----------|
| Source code | `src/event_discovery/` |
| Tests | `tests/` |
| Package config | `pyproject.toml` |
| Documentation | `README.md`, `IMPLEMENTATION_GUIDE.md` |
| Research paper | `paper/` |
| Changelog | `CHANGELOG.md` |

## Package Identity

| Field | Value |
|-------|-------|
| Package name | `event-discovery` |
| Python import | `import event_discovery` |
| CLI command | `event-discovery` |
| GitHub | `https://github.com/meshal-alawein/event-discovery-framework` |
| License | MIT |
| Author | Meshal Alshammari (UC Berkeley EECS) |

## Version Policy

- Version lives in `pyproject.toml` under `[project] version`
- Version is also echoed in `src/event_discovery/__init__.py` as `__version__`
- Releases are tagged `vX.Y.Z` on `main`
- CHANGELOG.md entry required for every release

## Architecture Invariants

The following architectural decisions are canonical:

1. **Single entry point**: All methods inherit `BaseEventDetector` from `core/base.py`
2. **Template Method pattern**: `process_video` → `_score_windows` → `_select`
3. **Shared features**: Feature extraction lives in `core/features.py`, not in method files
4. **Config dataclasses**: Every method has a corresponding `<Method>Config` dataclass
5. **No circular imports**: `core/` never imports from `methods/`
6. **Lazy optional imports**: VLM dependencies imported inside functions, not at module level

## Public API Surface

The following symbols constitute the public API (breaking changes require version bump):

```python
# Methods
from event_discovery.methods import (
    HierarchicalEnergyMethod, EnergyConfig,
    GeometricOutlierMethod,
    PureOptimizationMethod, OptimizationConfig,
    DenseVLMMethod,
)

# Core
from event_discovery.core import VideoWindow, VideoProcessor
from event_discovery.core.features import (
    compute_color_histogram, compute_edge_density_variance,
    compute_pixel_variance, compute_pixel_entropy,
    temporal_similarity, greedy_diverse_select,
)

# Evaluation
from event_discovery.evaluation import (
    compute_metrics, load_ground_truth, run_comparison,
    temporal_iou, generate_latex_table,
)
```

## Governance Documents

| Document | Purpose |
|----------|---------|
| [AGENTS.md](AGENTS.md) | Root governance rules for agents and contributors |
| [SSOT.md](SSOT.md) | This file — scope and canonical authority |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Human contributor guide |
| [SECURITY.md](SECURITY.md) | Vulnerability reporting |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
