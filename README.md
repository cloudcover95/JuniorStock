# JuniorStock

**Sovereign Edge-Native Multi-Agent Quantitative Trading Orchestrator (V6.3)**

Production-grade SDK with deterministic CI/CD, hardened packaging, and full test matrix.

## V6.3 Production Hardening
- GitHub Actions CI (Python 3.9 deterministic runner)
- Makefile for consistent developer experience
- Hardened pyproject.toml with hatchling + proper extras
- PyTest matrix for K_α / manifold integrity

## Quick Start (Production)

```bash
make install
make test
make lint
```

All work consolidated on `main`.