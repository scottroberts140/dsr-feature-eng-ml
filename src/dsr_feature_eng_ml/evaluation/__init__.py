"""Evaluation and audit public exports.

Keep the package surface lightweight by lazily importing plotting-heavy
modules only when specific evaluation symbols are requested.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dsr_feature_eng_ml.evaluation.model_audit_summary import ModelAuditSummary
    from dsr_feature_eng_ml.evaluation.schema import (
        DataSplits,
        ModelAuditorConfig,
        ModelConfiguration,
        ModelConfigurationStats,
        ModelFeatureImportance,
    )

    ModelSplitStats = ModelConfigurationStats.ModelSplitStats
    SplitType = ModelConfigurationStats.ModelSplitStats.SplitType

__all__ = [
    "DataSplits",
    "ModelConfiguration",
    "ModelFeatureImportance",
    "ModelAuditorConfig",
    "ModelConfigurationStats",
    "ModelSplitStats",
    "SplitType",
    "ModelAuditSummary",
]


def __getattr__(name: str):
    if name == "ModelAuditSummary":
        from dsr_feature_eng_ml.evaluation.model_audit_summary import ModelAuditSummary

        return ModelAuditSummary

    schema_exports = {
        "DataSplits",
        "ModelConfiguration",
        "ModelFeatureImportance",
        "ModelAuditorConfig",
        "ModelConfigurationStats",
    }
    if name in schema_exports:
        from dsr_feature_eng_ml.evaluation import schema

        return getattr(schema, name)

    if name == "ModelSplitStats":
        from dsr_feature_eng_ml.evaluation.schema import ModelConfigurationStats

        return ModelConfigurationStats.ModelSplitStats

    if name == "SplitType":
        from dsr_feature_eng_ml.evaluation.schema import ModelConfigurationStats

        return ModelConfigurationStats.ModelSplitStats.SplitType

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
