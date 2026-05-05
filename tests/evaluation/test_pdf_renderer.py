import dataclasses
from unittest.mock import MagicMock

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
from dsr_feature_eng_ml.evaluation.schema import (
    DataSplits,
    FeatureMetadata,
    ModelConfiguration,
)
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


def test_pdf_export_accepts_dict_model_params(populated_summary, tmp_path):
    """PDF export should render hyperparameter tables when params are plain dicts."""
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
        model_params={
            "n_estimators": 50,
            "optimization_strategy": OptimizationStrategy.RANDOM_SEARCH,
        },
        preds_val=pd.Series(
            preds_values, index=populated_summary.data_splits.val_target.index
        ),
    )

    pdf_path = populated_summary.export_results(
        prefix="Taxi_Audit_Dict_Params", file_type=FileType.PDF, path=tmp_path
    )

    assert pdf_path.exists()
    assert pdf_path.suffix == ".pdf"


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


def test_plot_feature_importance_caps_pdf_display_count(populated_summary):
    """Feature-importance subplots should cap bar count for PDF readability."""
    populated_summary.top_n_importance = 25
    populated_summary.pdf_feature_importance_chart_limit = 8
    renderer = AuditPDFRenderer(summary=populated_summary)

    importance_df = pd.DataFrame(
        {
            "id": [f"feature_{i}" for i in range(20)],
            "importance": [float(20 - i) for i in range(20)],
        }
    )

    fig, ax = plt.subplots()
    try:
        renderer._plot_feature_importance(
            ax=ax,
            importance_df=importance_df,
            model_color="#336699",
            test_metrics_text=None,
        )
    finally:
        plt.close(fig)

    assert ax.get_title() == "Top 8 Predictors"
    assert len(ax.patches) == 8
    assert any(
        "Showing 8 of requested top 20 for PDF readability" in text.get_text()
        for text in ax.texts
    )


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


def test_select_anomaly_features_uses_importance_when_capped(populated_summary):
    """When capped, anomaly columns should prioritize model feature importance."""
    populated_summary.anomaly_table_max_columns = 2
    renderer = AuditPDFRenderer(summary=populated_summary)

    renderer.best_model.model = MagicMock()
    renderer.best_model.model.feature_analysis = MagicMock()
    renderer.best_model.model.feature_analysis.feature_importances = pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3", "f4"],
            "importance": [0.9, 0.8, 0.7, 0.6],
        }
    )

    selected = renderer._select_anomaly_features(["f3", "f4", "f1", "f2"])
    assert selected == ["f1", "f2"]


def test_select_anomaly_features_falls_back_to_input_order(populated_summary):
    """When importances are unavailable, capped anomaly columns keep input order."""
    populated_summary.anomaly_table_max_columns = 2
    renderer = AuditPDFRenderer(summary=populated_summary)

    renderer.best_model.model = MagicMock()
    renderer.best_model.model.feature_analysis = MagicMock()
    renderer.best_model.model.feature_analysis.feature_importances = pd.DataFrame()

    selected = renderer._select_anomaly_features(["f3", "f4", "f1", "f2"])
    assert selected == ["f3", "f4"]


def test_get_anomaly_cap_note_present_when_capped(populated_summary):
    """Subtitle note should appear when anomaly feature columns are capped."""
    renderer = AuditPDFRenderer(summary=populated_summary)
    note = renderer._get_anomaly_cap_note(raw_count=12, selected_count=8)
    assert "Showing 8 of 12 anomaly context columns" in note


def test_get_anomaly_cap_note_empty_when_not_capped(populated_summary):
    """No subtitle note should be emitted when no anomaly column cap applies."""
    renderer = AuditPDFRenderer(summary=populated_summary)
    assert renderer._get_anomaly_cap_note(raw_count=8, selected_count=8) == ""


def test_get_anomaly_cap_note_suppressed_when_notes_disabled(populated_summary):
    """Cap note should be hidden when anomaly note rendering is disabled."""
    populated_summary.anomaly_table_show_notes = False
    renderer = AuditPDFRenderer(summary=populated_summary)

    note = renderer._get_anomaly_cap_note(raw_count=12, selected_count=8)
    assert note == ""


def test_get_anomaly_column_reduction_note_when_compressed(populated_summary):
    """Recommend anomaly cap when dynamic columns are visibly compressed."""
    populated_summary.anomaly_table_max_columns = None
    renderer = AuditPDFRenderer(summary=populated_summary)

    note = renderer._get_anomaly_column_reduction_note(
        dynamic_col_count=12,
        avg_dynamic_width=0.04,
        avg_base_width=0.12,
    )

    assert "anomaly_table_max_columns" in note


def test_get_anomaly_column_reduction_note_shown_with_cap_when_compressed(
    populated_summary,
):
    """When compressed, suggest lowering an existing cap."""
    populated_summary.anomaly_table_max_columns = 8
    renderer = AuditPDFRenderer(summary=populated_summary)

    note = renderer._get_anomaly_column_reduction_note(
        dynamic_col_count=12,
        avg_dynamic_width=0.04,
        avg_base_width=0.12,
    )

    assert "consider lowering" in note
    assert "anomaly_table_max_columns is 8" in note


def test_get_anomaly_column_reduction_note_not_shown_when_readable(populated_summary):
    """Do not show recommendation when dynamic columns remain readable."""
    populated_summary.anomaly_table_max_columns = None
    renderer = AuditPDFRenderer(summary=populated_summary)

    note = renderer._get_anomaly_column_reduction_note(
        dynamic_col_count=8,
        avg_dynamic_width=0.08,
        avg_base_width=0.12,
    )

    assert note == ""


def test_get_anomaly_column_reduction_note_shown_when_base_columns_compressed(
    populated_summary,
):
    """Recommend adjustment when key base columns are too narrow."""
    populated_summary.anomaly_table_max_columns = 9
    renderer = AuditPDFRenderer(summary=populated_summary)

    note = renderer._get_anomaly_column_reduction_note(
        dynamic_col_count=9,
        avg_dynamic_width=0.08,
        avg_base_width=0.07,
    )

    assert "Columns are compressed for fit" in note
    assert "anomaly_table_max_columns is 9" in note


def test_get_anomaly_column_reduction_note_suppressed_when_notes_disabled(
    populated_summary,
):
    """Compression advisory should be hidden when anomaly note rendering is disabled."""
    populated_summary.anomaly_table_show_notes = False
    populated_summary.anomaly_table_max_columns = 9
    renderer = AuditPDFRenderer(summary=populated_summary)

    note = renderer._get_anomaly_column_reduction_note(
        dynamic_col_count=9,
        avg_dynamic_width=0.04,
        avg_base_width=0.07,
    )

    assert note == ""


def test_build_anomaly_dynamic_headers_includes_ohe_suffix(populated_summary):
    """OHE anomaly columns should include encoded suffixes in headers."""
    renderer = AuditPDFRenderer(summary=populated_summary)
    renderer.summary.features = {
        "base_feature": FeatureMetadata(
            name="base_feature",
            id="F01",
            position=0,
            short_name="Base",
        )
    }
    base_header = "Base"

    headers = renderer._build_anomaly_dynamic_headers(
        raw_features=["base_feature_2", "base_feature_71"],
        resolved_features=["base_feature", "base_feature"],
    )

    assert headers == [f"{base_header} [2]", f"{base_header} [71]"]


def test_build_anomaly_dynamic_headers_keeps_duplicate_suffixes_unique(
    populated_summary,
):
    """If identical suffix labels collide, preserve uniqueness with a counter."""
    renderer = AuditPDFRenderer(summary=populated_summary)

    headers = renderer._build_anomaly_dynamic_headers(
        raw_features=["pickup_hour_sin", "pickup_hour_cos"],
        resolved_features=["pickup_hour", "pickup_hour"],
    )

    assert headers == ["pickup_hour", "pickup_hour (2)"]


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
