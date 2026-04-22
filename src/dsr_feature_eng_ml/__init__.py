"""Package entry point for dsr_feature_eng_ml.

Expose lightweight package APIs without importing optional visualization
dependencies until they are actually needed.
"""

from dsr_feature_eng_ml.enums import (
    ModelEnumSortOrder,
    ModelTypeData,
    ModelTypeDataRecType,
    PlotFileName,
    TaskTypeSortOrder,
)

__all__ = [
    "validate_n_jobs",
    "check_memory_risk",
    "PlotFileName",
    "ModelTypeData",
    "ModelEnumSortOrder",
    "TaskTypeSortOrder",
    "ModelTypeDataRecType",
    "AuditPDFRenderer",
]


def __getattr__(name: str):
    if name == "AuditPDFRenderer":
        from dsr_feature_eng_ml.evaluation.audit_pdf_renderer import AuditPDFRenderer

        return AuditPDFRenderer
    if name in {"validate_n_jobs", "check_memory_risk"}:
        from dsr_feature_eng_ml.utils.memory import check_memory_risk, validate_n_jobs

        exports = {
            "validate_n_jobs": validate_n_jobs,
            "check_memory_risk": check_memory_risk,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
