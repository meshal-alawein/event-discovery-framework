---
type: normative
authority: canonical
audience: [agents, contributors, maintainers]
last-verified: 2026-02-25
---

# AGENTS — Event Discovery Framework Governance

> **Status: Normative.** Do not modify without maintainer review.

This repository is governed by clear engineering and documentation standards
aligned with the **Morphism Categorical Governance Framework** principles.

## Governance Source

| Authority | Location |
|-----------|----------|
| Root governance | [AGENTS.md](AGENTS.md) (this file) |
| Contributing guide | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Security policy | [SECURITY.md](SECURITY.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |

## Repository Scope

`event-discovery-framework` is a **Python research library** for physics-inspired
event discovery in long-horizon video. It is a standalone repository under the
`meshal-alawein` GitHub organization.

## Directory Layout

| Directory | Purpose | Governance Level |
|-----------|---------|-----------------|
| `src/event_discovery/` | Core Python package | **Primary** — all changes require tests |
| `src/event_discovery/core/` | Shared infrastructure (base, features, video_processor) | **Stable** — changes need ADR comment |
| `src/event_discovery/methods/` | Detection algorithms | **Active** — test coverage required |
| `tests/` | Test suite | **Required** — never delete without replacement |
| `paper/` | LaTeX research paper | **Additive only** — do not modify compiled figures |
| `notebooks/` | Interactive demos | **Illustrative** — must stay runnable |
| `scripts/` | Automation scripts | **Tooling** — document any changes |
| `.github/` | CI/CD workflows and templates | **Infrastructure** |
| `docs/` | Internal documentation | **Supplementary** |

## Invariants (Must Always Hold)

1. **Tests pass**: `pytest tests/ -v` must exit 0 on `main`
2. **Lint clean**: `ruff check src/ tests/` must exit 0
3. **Type safe**: `mypy src/event_discovery/` must report no errors on `main`
4. **Imports work**: `python -c "import event_discovery"` must succeed after install
5. **README accurate**: README code examples must match actual API signatures
6. **No secrets**: API keys or credentials must never appear in source

## Agent Rules

When this repository is modified by an AI agent or automated tool:

- **Read** `AGENTS.md`, `CONTRIBUTING.md`, and `SSOT.md` before making changes
- **Never** modify `paper/figures/*.pdf` (generated artifacts)
- **Never** skip the test suite — run `pytest tests/ -v` before committing
- **Always** update `CHANGELOG.md` when changing public API or behavior
- **Always** keep docstrings and type hints accurate
- **Prefer** small, focused commits with conventional commit messages
- **Check** that new methods inherit from `BaseEventDetector` in `core/base.py`

## Naming Conventions

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Config dataclasses**: `<MethodName>Config`
- **Detectors**: `<MethodName>Method(BaseEventDetector)`

## Commit Message Format

```
type(scope): short description

feat(methods): add transformer-based window encoder
fix(evaluation): correct IoU boundary condition
docs(readme): update installation instructions
test(hierarchical): add edge case for empty window list
refactor(core): extract feature normalization to shared util
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`

## Code Review Requirements

- All PRs require at least one approval before merge
- CI must pass (tests + lint + type-check)
- New methods must include unit tests covering: init, score_windows, edge cases
- Changes to `EnergyConfig` or `BaseEventDetector` require explicit approval

## Dependency Policy

- **Core deps**: Keep minimal (numpy, opencv, scipy, sklearn, pandas, click, tqdm)
- **Optional deps**: VLM, visualization, notebook extras defined in `pyproject.toml`
- **Dev deps**: pytest, ruff, mypy — no production code may import dev deps
- **Version pins**: Minimum versions only (no upper bounds unless proven necessary)

---

*Aligned with Morphism Systems governance principles. See https://morphism.systems for framework reference.*
