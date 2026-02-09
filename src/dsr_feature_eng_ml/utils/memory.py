"""Memory-aware helpers for tuning and resource checks."""

from __future__ import annotations
import psutil
import pandas as pd
from typing import Tuple, TYPE_CHECKING
from dsr_utils.formatting import DataScale

if TYPE_CHECKING:
    from dsr_feature_eng_ml.models.model_specification import ModelSpecification


def validate_n_jobs(value: int) -> int:
    """Normalize n_jobs to a safe CPU count.

    Args:
        value: Requested number of jobs; -1 means all cores.

    Returns:
        A bounded job count between 1 and the available CPU count.
    """
    import os

    n_jobs = 1
    cpu_count = os.cpu_count() or 1

    if value == -1:
        n_jobs = cpu_count
    else:
        n_jobs = min(value, cpu_count)

    return n_jobs


def check_memory_risk(
    df: pd.DataFrame, model: ModelSpecification, n_jobs: int = -1
) -> Tuple[bool, float, float, float]:
    """Estimate memory risk for model tuning and print a summary.

    Args:
        df: Dataset used for training/tuning.
        model: Model specification with tuning parameters.
        n_jobs: Requested parallel workers (-1 for all cores).

    Returns:
        Tuple of (risk, estimated_peak_gb, available_gb, model_multiplier).
    """
    from dsr_utils.formatting import (
        BoolRepresentation,
        IntegerFormat,
        ValueDescFormat,
        BoolFormat,
        format_label_value_pairs,
    )
    from dsr_feature_eng_ml.preferences import prefs

    # 1. Setup environment
    available_gb = psutil.virtual_memory().available
    dataset_gb = df.memory_usage(deep=True).sum()

    # 2. Get the comlexity multiplier
    model_multiplier = model.model_type.tuning_multiplier
    concurrent_workers = validate_n_jobs(n_jobs)

    # 3. Factor in Parallelism
    num_candidates = model.model_dials.num_candidates
    storage_overhead = (
        (model.total_fits * 0.15) + (num_candidates * 0.1)
    ) * DataScale.GB.get_size()

    # 4. Calculate Peak Estimated Need
    # Peak = Base Data + (Data * Complexity * Parallelism factor)
    # We use sqrt(jobs) because not all workers peak at the exact same millisecond
    processing_spike = dataset_gb * model_multiplier * concurrent_workers
    estimated_peak_gb = dataset_gb + processing_spike + storage_overhead
    risk = estimated_peak_gb > (available_gb * 0.85)
    risk_format = BoolFormat(representation=BoolRepresentation.YES_NO)
    model_multiplier_format = ValueDescFormat(
        precision=1, description="x", description_leading_space=False
    )
    int_format = IntegerFormat()
    stats = [
        ("Available", prefs.gb_format.format_value(available_gb)),
        ("Dataset", prefs.gb_format.format_value(dataset_gb)),
        ("Model multiplier", model_multiplier_format.format_value(model_multiplier)),
        ("n_jobs", int_format.format_value(n_jobs)),
        ("Concurrent workers", int_format.format_value(concurrent_workers)),
        ("Total fits", int_format.format_value(model.total_fits)),
        ("Candidates", int_format.format_value(num_candidates)),
        ("Storage overhead", prefs.gb_format.format_value(storage_overhead)),
        ("Estimated Peak", prefs.gb_format.format_value(estimated_peak_gb)),
        ("Risk", risk_format.format_value(risk)),
    ]
    print(format_label_value_pairs(stats))
    return risk, estimated_peak_gb, available_gb, model_multiplier
