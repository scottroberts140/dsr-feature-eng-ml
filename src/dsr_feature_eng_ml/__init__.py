"""Package entry point for dsr_feature_eng_ml.

Ensure the preferences singleton initializes immediately on package import.
"""

from dsr_feature_eng_ml.enums import (
    ModelEnumSortOrder,
    ModelTypeData,
    ModelTypeDataRecType,
    PlotFileName,
    TaskTypeSortOrder,
)
from dsr_feature_eng_ml.evaluation.audit_pdf_renderer import AuditPDFRenderer
from dsr_feature_eng_ml.utils.memory import check_memory_risk, validate_n_jobs

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
