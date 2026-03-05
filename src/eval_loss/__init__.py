# eval/__init__.py
"""
Evaluation package (no-angles pipeline) for Lantern-style models.

Design:
- Keep imports lazy so `import eval` is lightweight (no matplotlib load).
- Expose a tiny, stable public API via __all__:
    - run_comprehensive_eval_noangles  : main entry point
    - compute_energy_conservation, corr_matrix, cfd_offdiag_fro : core metrics
"""

from typing import Any

__all__ = [
    "run_comprehensive_eval_noangles",
    "compute_energy_conservation",
    "corr_matrix",
    "cfd_offdiag_fro",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    # Lazy accessors to avoid importing heavy deps until needed
    if name == "run_comprehensive_eval_noangles":
        from .eval_runner import run_comprehensive_eval_noangles
        return run_comprehensive_eval_noangles

    if name in ("compute_energy_conservation", "corr_matrix", "cfd_offdiag_fro"):
        from .metrics import compute_energy_conservation, corr_matrix, cfd_offdiag_fro
        return {
            "compute_energy_conservation": compute_energy_conservation,
            "corr_matrix": corr_matrix,
            "cfd_offdiag_fro": cfd_offdiag_fro,
        }[name]

    raise AttributeError(f"module 'eval' has no attribute {name!r}")
