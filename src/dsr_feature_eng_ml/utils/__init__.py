"""Utils module for dsr_feature_eng_ml.

Avoid importing optional system-inspection dependencies until the exported
helpers are actually accessed.
"""

__all__ = ["validate_n_jobs", "check_memory_risk"]


def __getattr__(name: str):
    if name in {"validate_n_jobs", "check_memory_risk"}:
        from dsr_feature_eng_ml.utils.memory import check_memory_risk, validate_n_jobs

        exports = {
            "validate_n_jobs": validate_n_jobs,
            "check_memory_risk": check_memory_risk,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
