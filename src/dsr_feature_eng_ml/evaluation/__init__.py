from typing import TypeAlias

from dsr_feature_eng_ml.evaluation.schema import (
    DataSplits,
    ModelConfiguration,
    ModelFeatureImportance,
    ModelAuditorConfig,
    ModelConfigurationStats,
)
from dsr_feature_eng_ml.evaluation.model_audit_summary import ModelAuditSummary

ModelSplitStats: TypeAlias = ModelConfigurationStats.ModelSplitStats
SplitType: TypeAlias = ModelConfigurationStats.ModelSplitStats.SplitType

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
