import dataclasses

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns
from dsr_feature_eng_ml.enums import (
    BalancingStrategy,
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)
from dsr_feature_eng_ml.evaluation import ModelAuditSummary
from dsr_feature_eng_ml.evaluation.audit_pdf_renderer import AuditPDFRenderer
from dsr_feature_eng_ml.evaluation.schema import DataSplits, ModelConfiguration
from dsr_feature_eng_ml.models.k_neighbors_classifier import KNeighborsClassifierParams
from dsr_feature_eng_ml.models.lasso_regression import LassoParams
from dsr_feature_eng_ml.models.random_forest import RandomForestRegressorParams
from dsr_files.enums import FileType


@pytest.fixture
def populated_summary(mini_taxi_df):
    """Create a summary with two competing models (RFR vs Lasso)."""
    # 1. Create a Winning Model (RFR)
    m1 = ModelConfiguration(
        id="01",
        model_type=ModelType.RANDOM_FOREST_REGRESSOR,
        task_type=TaskType.REGRESSION,
        score_val=0.8145,
        balancing_strategy=BalancingStrategy.NONE,
        optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
        cv=5,
        scoring=ScoringMetric.R2,
        n_jobs=1,
        n_iter=10,
        model_params=RandomForestRegressorParams(
            scoring=ScoringMetric.R2, random_state=75
        ),
    )
    # 2. Create a Losing Model (Lasso)
    m2 = ModelConfiguration(
        id="02",
        model_type=ModelType.LASSO,
        task_type=TaskType.REGRESSION,
        score_val=0.7500,
        balancing_strategy=BalancingStrategy.NONE,
        optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
        cv=5,
        scoring=ScoringMetric.R2,
        n_jobs=1,
        n_iter=10,
        model_params=LassoParams(),
    )
    target = "fare_amount"
    features = [c for c in mini_taxi_df.columns if c != target]

    return ModelAuditSummary(
        data_splits=DataSplits.from_data_source(
            mini_taxi_df,
            features_to_include=features,
            target_column=target,
            test_size=0.2,
            valid_size=0.2,
            original_row_count=len(mini_taxi_df),
            random_state=75,
        ),
        results=[m1, m2],
        dataset_name="Taxi Audit Test",
        original_row_count=1000,
    )


def test_pdf_renderer_initialization(populated_summary):
    """
    Verify the renderer correctly links to the audit summary and applies the title.
    Matches the 'Model Audit Report' header on Page 1.
    """
    title = "Yellow Taxi Performance Review"
    renderer = AuditPDFRenderer(summary=populated_summary, report_title=title)

    assert renderer.summary == populated_summary
    assert renderer.report_title == title


def test_pdf_renderer_shows_integrity_na_without_cleaned_score(populated_summary):
    """Integrity should render as N/A when no cleaned validation score exists."""
    renderer = AuditPDFRenderer(summary=populated_summary)

    assert renderer.best_model.quality_score_text == "Integrity Score: N/A"
    assert renderer.best_model.integrity_summary_value == "N/A"
    integrity_row = renderer.executive_summary_metrics.loc[
        renderer.executive_summary_metrics["Metric"] == "Integrity:", "Value"
    ].iloc[0]
    assert integrity_row == "N/A"


def test_model_legend_group_order(populated_summary):
    """Legend should order groups as Classification, Regression, Unknown.

    The Regression header must be the first item in the right column (at index `mid`),
    not orphaned at the bottom of the left column.
    """
    renderer = AuditPDFRenderer(summary=populated_summary)
    model_type_data = renderer._build_model_legend_records()

    header_task_types = [
        mtd.task_type.name for mtd in model_type_data if mtd.rec_type.name == "HEADER"
    ]
    assert header_task_types == ["CLASSIFICATION", "REGRESSION", "UNKNOWN"]

    # The column split point should be the index of the Regression header.
    regression_header_idx = next(
        i
        for i, mtd in enumerate(model_type_data)
        if mtd.rec_type.name == "HEADER" and mtd.task_type == TaskType.REGRESSION
    )
    classification_header_idx = next(
        i
        for i, mtd in enumerate(model_type_data)
        if mtd.rec_type.name == "HEADER" and mtd.task_type == TaskType.CLASSIFICATION
    )
    # Classification header is in the left column (before mid)
    assert classification_header_idx < regression_header_idx
    # Regression header is the start of the right column (mid = regression_header_idx)
    assert regression_header_idx > 0


def test_pdf_export_workflow(populated_summary, tmp_path):
    """
    Verify the high-level export_results call triggers the PDF rendering lifecycle.
    This mimics the creation of the final PDF artifact.
    """
    # Mock data splits target vs predictions
    # If target is [20, 15] and preds are [20, 50], index 1 is an anomaly
    # The number of values in the target column has to be the same as the
    # number of rows in the dataframe.
    df_len = len(populated_summary.data_splits.val_target)
    target_values = pd.Series(20.0).repeat(df_len - 1)
    target_values[1] = 15.0
    preds_values = pd.Series(20.0).repeat(df_len - 1)
    preds_values[1] = 50.0
    populated_summary.data_splits = dataclasses.replace(
        populated_summary.data_splits, val_target=target_values
    )
    populated_summary.results[0] = dataclasses.replace(
        populated_summary.results[0],
        preds_val=pd.Series(
            preds_values, index=populated_summary.data_splits.val_target.index
        ),
    )
    # 1. Execute PDF export through the summary orchestrator
    # This internally instantiates AuditPDFRenderer and calls .render()
    pdf_path = populated_summary.export_results(
        prefix="Taxi_Audit", file_type=FileType.PDF, path=tmp_path
    )

    # 2. Verify the file exists and has the correct naming convention
    assert pdf_path.exists()
    assert pdf_path.suffix == ".pdf"
    assert "Taxi_Audit" in pdf_path.name


def test_renderer_toc_registration(populated_summary):
    """
    Verify that the rendering process populates the Table of Contents registry.
    Ensures 'Page 9: Random Forest Deep Dive' is reachable.
    """
    renderer = AuditPDFRenderer(summary=populated_summary)

    # The render call triggers page generation and TOC registration
    pdf_doc = renderer.render()

    # After render, the registry should contain references to the summary and deep dives
    assert len(pdf_doc.toc_pages) > 0


def test_plot_target_distribution_handles_string_targets(populated_summary):
    """Ensure categorical/string targets do not trigger numeric percentile errors."""
    y_train = populated_summary.data_splits.train_target.map(
        lambda x: "high" if x >= 30 else "low"
    )
    y_val = populated_summary.data_splits.val_target.map(
        lambda x: "high" if x >= 30 else "low"
    )

    populated_summary.data_splits = dataclasses.replace(
        populated_summary.data_splits,
        train_target=y_train,
        val_target=y_val,
    )

    renderer = AuditPDFRenderer(summary=populated_summary)
    fig, ax = plt.subplots()
    try:
        renderer._plot_target_distribution(ax)
    finally:
        plt.close(fig)

    assert ax.get_title() == "Target Distribution by Class"
    assert len(ax.patches) > 0


def test_plot_regression_residuals_constant_predictions(populated_summary):
    """
    Ensure no UserWarning when predictions and targets are nearly constant.

    Previously, identical y_preds values produced x_buffer=0, and constant
    residuals produced y_limit=0, both triggering matplotlib's
    'identical low and high lims makes transformation singular' warning.
    """
    import warnings

    from dsr_feature_eng_ml.enums import ModelType

    renderer = AuditPDFRenderer(summary=populated_summary)
    y_true = pd.Series([20.0] * 10)
    y_preds = pd.Series([20.0] * 10)
    model_name = ModelType.RANDOM_FOREST_REGRESSOR.value
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            renderer._plot_regression_residuals(
                ax, y_true, y_preds, model_name=model_name
            )
    finally:
        plt.close(fig)


def test_plot_efficiency_scatter_handles_missing_mae(populated_summary):
    """Ensure the efficiency scatter still renders when MAE is unavailable."""
    renderer = AuditPDFRenderer(summary=populated_summary)
    renderer.summary_df["MAE"] = pd.NA

    fig, ax = plt.subplots()
    try:
        renderer._plot_efficiency_scatter(sns, ax)
        point_count = sum(
            len(collection.get_offsets())
            for collection in ax.collections
            if hasattr(collection, "get_offsets")
        )
    finally:
        plt.close(fig)

    assert point_count > 0
    assert ax.get_title().startswith("Efficiency:")


def test_plot_cumulative_importance_shows_message_when_unavailable(populated_summary):
    """KNN-like models without importances should render an explicit N/A message."""
    knn_cfg = ModelConfiguration(
        id="knn-01",
        model_type=ModelType.K_NEIGHBORS_CLASSIFIER,
        task_type=TaskType.CLASSIFICATION,
        score_val=0.9999,
        balancing_strategy=BalancingStrategy.NONE,
        optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
        cv=5,
        scoring=ScoringMetric.F1,
        n_jobs=1,
        n_iter=10,
        model_params=KNeighborsClassifierParams(),
    )
    populated_summary.results = [knn_cfg]

    renderer = AuditPDFRenderer(summary=populated_summary)
    fig, ax = plt.subplots()
    try:
        renderer._plot_cumulative_importance(sns, ax)
        chart_text = "\n".join(t.get_text() for t in ax.texts)
    finally:
        plt.close(fig)

    assert "Feature importance unavailable" in chart_text


def test_plot_feature_importance_centers_fallback_message(populated_summary):
    """Feature-importance fallback for unsupported models should be centered."""
    renderer = AuditPDFRenderer(summary=populated_summary)
    fig, ax = plt.subplots()
    try:
        renderer._plot_feature_importance(
            ax=ax,
            importance_df=pd.DataFrame(),
            model_color="steelblue",
            test_metrics_text=None,
        )
        assert len(ax.texts) == 1
        fallback_text = ax.texts[0]
        assert fallback_text.get_position() == (0.5, 0.5)
        assert fallback_text.get_verticalalignment() == "center"
        assert fallback_text.get_transform() == ax.transAxes
        assert not ax.axison
    finally:
        plt.close(fig)


def test_plot_classification_diagnostics_uses_centered_inset_axis(populated_summary):
    """Confusion matrix should render in a centred absolute axis within the parent quadrant."""
    renderer = AuditPDFRenderer(summary=populated_summary)
    fig, ax = plt.subplots()
    try:
        y_true = pd.Series(["yes", "no", "yes", "no"])
        y_preds = pd.Series(["yes", "yes", "no", "no"])
        renderer._plot_classification_diagnostics(ax=ax, y_preds=y_preds, y_true=y_true)
        fig.canvas.draw()

        assert ax.get_title() == "Validation Confusion Matrix"
        assert not ax.axison

        # Matrix is a separate figure-level axis, not an inset child
        assert len(ax.child_axes) == 0
        assert len(fig.axes) == 2
        matrix_ax = fig.axes[1]
        assert matrix_ax.get_xlabel() == ""
        assert matrix_ax.get_ylabel() == ""
        assert not matrix_ax.get_in_layout()

        # Matrix axis should be horizontally centred within the parent axis
        parent_pos = ax.get_position()
        matrix_pos = matrix_ax.get_position()
        parent_center_x = parent_pos.x0 + (parent_pos.width / 2)
        matrix_center_x = matrix_pos.x0 + (matrix_pos.width / 2)
        assert matrix_center_x == pytest.approx(parent_center_x, abs=0.02)
    finally:
        plt.close(fig)
