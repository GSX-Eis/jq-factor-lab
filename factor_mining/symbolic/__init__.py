"""Symbolic factor mining package.

This package implements the v1 mining surface that generates standalone
FactorBase-compatible factor files.  The search backend is intentionally small
in v1 so that future DSO-style backends can replace it without changing the
data, objective, artifact, or compiler contracts.
"""

from .pipeline import run_symbolic_mining_pipeline

__all__ = ["run_symbolic_mining_pipeline"]

