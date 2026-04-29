"""Memory-aware helpers for tuning and resource checks."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pandas as pd
import psutil
from dsr_utils.formatting import (
    BoolFormat,
    BoolRepresentation,
    DataFormat,
    DataScale,
    IntegerFormat,
    ValueDescFormat,
    format_label_value_pairs,
)

if TYPE_CHECKING:
    from dsr_feature_eng_ml.models.model_specification import ModelSpecification

logger = logging.getLogger(__name__)


def validate_n_jobs(value: int) -> int:
    """
    Normalize n_jobs to a safe, available CPU count.

    Parameters
    ----------
    value : int
        The requested number of parallel jobs. A value of -1 indicates
        that all available cores should be used.

    Returns
    -------
    int
        A validated job count, bounded between 1 and the system's
        available CPU count.
    """
    cpu_count = os.cpu_count() or 1

    if value == -1:
        return cpu_count

    # Ensure we return at least 1 and never more than the physical max
    return max(1, min(value, cpu_count))


def check_memory_risk(
    df: pd.DataFrame, model: "ModelSpecification", n_jobs: int = -1
) -> tuple[bool, float, float, float]:
    """
    Estimate system memory risk for model tuning and display a summary.

    Calculates the potential peak memory usage by factoring in the dataset size,
    model complexity (multiplier), and the degree of parallelism.

    Parameters
    ----------
    df : pd.DataFrame
        The training dataset.
    model : ModelSpecification
        The model instance containing tuning parameters and complexity multipliers.
    n_jobs : int, default -1
        The number of parallel workers requested.

    Returns
    -------
    risk : bool
        True if estimated peak exceeds 85% of available system memory.
    estimated_peak_gb : float
        The calculated maximum memory usage in bytes.
    available_gb : float
        The current available system memory in bytes.
    model_multiplier : float
        The complexity multiplier associated with the ModelType.
    """
    from dsr_feature_eng_ml.prefs_instance import prefs

    # 1. Setup environment
    available_gb = psutil.virtual_memory().available
    dataset_gb = df.memory_usage(deep=True).sum()

    # 2. Get the complexity multiplier
    model_multiplier = model.model_type.tuning_multiplier
    concurrent_workers = validate_n_jobs(n_jobs)

    # 3. Factor in Parallelism and candidate storage
    num_candidates = model.model_dials.num_candidates
    # Storage overhead accounts for serialized models and CV results
    storage_overhead = (
        (model.total_fits * 0.15) + (num_candidates * 0.1)
    ) * DataScale.GB.get_size()

    # 4. Calculate Peak Estimated Need
    # Heuristic: Base Data + (Data * Complexity * Workers) + Overhead
    processing_spike = dataset_gb * model_multiplier * concurrent_workers
    estimated_peak_gb = dataset_gb + processing_spike + storage_overhead

    risk = estimated_peak_gb > (available_gb * 0.85)

    # 5. Formatting and Reporting
    risk_format = BoolFormat(representation=BoolRepresentation.YES_NO)
    model_multiplier_format = ValueDescFormat(
        precision=1, description="x", description_leading_space=False
    )
    int_format = IntegerFormat()

    dataset_fmt = DataFormat(
        data_scale=DataScale.AUTO, precision=2, include_space_before_scale=True
    )

    stats = [
        ("Available", prefs.gb_format.format_value(available_gb)),
        ("Dataset", dataset_fmt.format_value(dataset_gb)),  # AUTO picks MB/GB
        ("Model Multiplier", model_multiplier_format.format_value(model_multiplier)),
        ("n_jobs", int_format.format_value(n_jobs)),
        ("Concurrent Workers", int_format.format_value(concurrent_workers)),
        ("Total Fits", int_format.format_value(model.total_fits)),
        ("Candidates", int_format.format_value(num_candidates)),
        ("Storage Overhead", prefs.gb_format.format_value(storage_overhead)),
        ("Estimated Peak", prefs.gb_format.format_value(estimated_peak_gb)),
        ("Risk", risk_format.format_value(risk)),
    ]

    logger.info("\n--- Memory Risk Audit ---")
    logger.info(format_label_value_pairs(stats))

    return risk, estimated_peak_gb, available_gb, model_multiplier
