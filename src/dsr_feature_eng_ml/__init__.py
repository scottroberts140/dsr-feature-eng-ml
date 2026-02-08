"""Package entry point for dsr_feature_eng_ml.

Ensure the preferences singleton initializes immediately on package import.
"""

from dsr_feature_eng_ml.utils.memory import validate_n_jobs, check_memory_risk
from dsr_feature_eng_ml.enums import (
    ModelTypeTuningMultiplier,
    PlotFileName,
    ModelTypeTaskType,
    ModelTypeData,
    ModelEnumSortOrder,
    TaskTypeSortOrder,
    ModelTypeDataRecType,
)
from dsr_feature_eng_ml.evaluation.audit_pdf_renderer import AuditPDFRenderer

__all__ = [
    "validate_n_jobs",
    "check_memory_risk",
    "ModelTypeTuningMultiplier",
    "PlotFileName",
    "ModelTypeTaskType",
    "ModelTypeData",
    "ModelEnumSortOrder",
    "TaskTypeSortOrder",
    "ModelTypeDataRecType",
    "AuditPDFRenderer",
]
