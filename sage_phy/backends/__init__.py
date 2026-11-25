"""Simulation backends.

analytic  closed-form physics, no GPU. For testing the pipeline end to end and
          as the reference in gate G4. Its numbers are never reported as results.
isaac     Isaac Sim 5.0 / PhysX. Import only inside Isaac Sim's Python.
"""


def get_backend(name: str, cfg: dict, **kw):
    if name == "analytic":
        from .analytic import AnalyticBackend
        return AnalyticBackend(cfg)
    if name == "isaac":
        from .isaac import IsaacBackend
        return IsaacBackend(cfg, **kw)
    raise ValueError(f"unknown backend {name!r}")
