"""PDF rendering utilities for model audit reports."""

from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backend_bases import RendererBase
import seaborn as sns
from typing import Any, List, Tuple, Optional, TYPE_CHECKING, cast
import pandas as pd
import numpy as np
import os
from datetime import datetime
import time
import dataclasses
from enum import Enum
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dsr_feature_eng_ml.preferences import prefs
from dsr_feature_eng_ml.evaluation import ModelAuditSummary
from dsr_feature_eng_ml.evaluation.schema import ModelConfiguration, DataSplits
from dsr_feature_eng_ml.enums import (
    TaskType,
    ModelTypeData,
    ModelTypeDataRecType,
    ModelEnumSortOrder,
)
from dsr_utils.formatting import (
    TextAlignment,
    NumericScale,
    DataScale,
    BoolRepresentation,
    FormatConfig,
    CurrencyFormat,
    PercentageFormat,
    IntegerFormat,
    FloatFormat,
    ValueDescFormat,
    DateTimeFormat,
    DataFormat,
    EnumFormat,
    StringFormat,
    BoolFormat,
    format_text,
)
from dsr_utils.strings import apply_tracking
from dsr_files.pdf_handler import (
    PDFDocument,
    PageConfiguration,
    PageSize,
    PageOrientation,
    PageColors,
)
from dsr_utils.tables import (
    TableEdgeColor,
    TableEdgeLinewidth,
    TableColumnStyle,
    TableColumn,
    Table,
    TablePage,
    TableLayout,
    render_table,
    render_table_from_page_layout,
)
from dsr_utils.matplotlib import get_artist_bbox


class AuditPDFRenderer:
    """Render a multi-page PDF report for a `ModelAuditSummary`."""

    class Model:
        """Convenience wrapper for computed model summary fields."""

        def __init__(self, model: ModelConfiguration, data_splits: DataSplits):
            """Precompute key model indicators used by the report."""
            self.model = model
            self.model_quality = prefs.get_model_quality(model.quality_score)
            quality_score_format = ValueDescFormat(
                precision=2,
                description=self.model_quality.text,
                description_leading_space=True,
                description_decorator="()",
            )
            self.quality_score_text = f"Data Quality Score: {quality_score_format.format_value(model.quality_score)}"
            self.is_efficient = (
                model.efficiency(data_splits=data_splits)
                >= prefs.model_efficiency_threshold
            )
            self.is_accurate = model.val_score > prefs.model_accuracy_limit
            self.is_acceptable = model.val_score > prefs.model_acceptable_limit
            self.is_stable = model.val_score > prefs.model_stability_limit
            self.recommendation = prefs.get_model_recommendation(
                is_accurate=self.is_accurate,
                is_stable=self.is_stable,
                is_efficient=self.is_efficient,
                is_acceptable=self.is_acceptable,
            )

    def __init__(
        self,
        summary: ModelAuditSummary,
        report_title: str = "Model Audit Report",
    ):
        """Initialize the PDF renderer for a specific audit summary.

        Args:
            summary: Completed audit summary with results.
            report_title: Title shown throughout the report.
        """
        self.summary = summary
        self.report_title = report_title

        if len(summary.results) == 0:
            raise IndexError("Audit Summary does not contain any results.")

        self.results = summary.results
        _best_model = summary.best_overall_model

        if _best_model is None:
            raise ValueError("Best model could not be determined.")

        self.best_model = AuditPDFRenderer.Model(
            model=_best_model, data_splits=summary.data_splits
        )
        self.total_cpu_time = sum(res.total_duration for res in self.results)
        self.max_ram_observed = max(res.actual_peak_gb for res in summary.results)
        cpu_count = os.cpu_count()
        os_cores_format = ValueDescFormat(
            precision=0,
            description="Cores Detected",
            description_leading_space=True,
            description_decorator="",
        )
        peak_memory_demand_format = DataFormat(data_scale=DataScale.GB)
        formatted_peak_memory_demand = peak_memory_demand_format.format_value(
            self.max_ram_observed
        )
        data_volume_format = ValueDescFormat(
            precision=0, description="rows", description_leading_space=True
        )

        self.hardware_stats_text = (
            f"Audit Hardware Context: {os_cores_format.format_value(cpu_count)} | "
            f"Peak Memory Demand: {formatted_peak_memory_demand} | "
            f"Data Volume: {data_volume_format.format_value(summary.processed_row_count)} | "
            "Status: Hardware Safety Limits Respected"
        )
        # summary_df contains model results
        # importance_dict contains corresponding feature importance data; key is summary_df['ID']
        self.summary_df, self.importance_dict = self._get_audit_data()

        # Sort results in descending order
        self.summary_df.sort_values(by="Val Score", ascending=False, inplace=True)

        # Best model metrics
        model = self.best_model.model
        processed_row_format = IntegerFormat(
            precision=2, numeric_scale=NumericScale.AUTO
        )
        self.formatted_processed_row_count = processed_row_format.format_value(
            self.summary.processed_row_count
        )

        efficiency_format = ValueDescFormat(
            precision=1,
            numeric_scale=NumericScale.K,
            description="rows/sec",
            description_leading_space=True,
        )

        sampling_factor_format = PercentageFormat(precision=1)
        formatted_sampling_factor = sampling_factor_format.format_value(
            model.sampling_factor
        )

        winning_model_format = EnumFormat()
        model_scoring_format = EnumFormat()
        model_quality_format = ValueDescFormat(
            precision=2, description="/100 Score", description_leading_space=False
        )
        audit_duration_format = DateTimeFormat(
            use_duration_format=True, alignment=TextAlignment.LEFT
        )
        resource_os_cores_format = ValueDescFormat.from_format(os_cores_format)
        resource_os_cores_format.description = "Cores"
        self._best_model_metrics = pd.DataFrame(
            [
                [
                    "Winning Model:",
                    f"{winning_model_format.format_value(model.model_type)}",
                ],
                [
                    f"Validation Score ({model_scoring_format.format_value(model.scoring)}):",
                    f"{prefs.score_format.format_value(model.val_score)}",
                ],
                [
                    f"Test Score ({model_scoring_format.format_value(model.scoring)}):",
                    f"{prefs.score_format.format_value(model.test_score) if model.has_test_set_evaluation_scores else '-'}",
                ],
                [
                    "Throughput:",
                    f"{efficiency_format.format_value(model.efficiency(self.summary.data_splits))}",
                ],
                [
                    "Integrity:",
                    f"{model_quality_format.format_value(model.quality_score)}",
                ],
                ["Audit Scale:", f"{self.formatted_processed_row_count} processed"],
                [
                    "Audit Duration:",
                    f"{audit_duration_format.format_value(self.total_cpu_time)}",
                ],
                [
                    "Resources:",
                    f"{resource_os_cores_format.format_value(cpu_count)} | Peak Mem: {formatted_peak_memory_demand}",
                ],
                ["Methodology:", f"{formatted_sampling_factor} used for training"],
            ],
            columns=["Metric", "Value"],
        )
        self.executive_summary_metrics = self._best_model_metrics[:5]
        self.strategic_recommendation_metrics = self._best_model_metrics
        pc = PageConfiguration(
            page_size=PageSize.LETTER,
            orientation=PageOrientation.LANDSCAPE,
            colors=PageColors(page_num=prefs.color_neutral, title=prefs.color_title),
            margins=PageSize.LETTER.margins,
            header_func=self._draw_page_header,
            footer_func=self._draw_page_footer,
        )
        self.pdf_doc = PDFDocument(
            doc_title=report_title, page_configuration=pc, page_count_before_toc=2
        )

    def render(self) -> PDFDocument:
        """Orchestrates the full PDF document creation."""
        self._render_title_page()
        self._render_executive_summary()
        self._render_anomaly_page()
        self._render_model_legend()
        self._render_audit_results()
        self._render_cv_vs_final()
        self._render_features_page()

        for config in self.results:
            self._render_model_deep_dive(config=config)

        self._render_detailed_audit_stats()
        self._render_recommendation_page()
        self.pdf_doc.render_table_of_contents()
        return self.pdf_doc

    def _draw_page_header(
        self,
        pdf_page: PDFDocument.Page,
        page_name: str,
        print_page_name: bool = True,
    ) -> None:
        """Render the standard page header for a report page."""
        fig = pdf_page.fig
        # print(page_name)
        pc = self.pdf_doc._page_configuration
        # Report Title
        fig.text(
            0.5,
            0.96,
            self.report_title.upper(),
            fontsize=16,
            color=prefs.color_title,
            weight="bold",
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

        # Dataset name
        fig.text(
            pc.left_margin,
            0.93,
            f"Dataset: {self.summary.dataset_name}",
            color=prefs.color_neutral,
            fontsize=9,
            style="italic",
            ha="left",
            transform=fig.transFigure,
        )

        # Page Name
        if print_page_name:
            fig.text(
                0.5,
                0.93,
                page_name.upper(),
                fontsize=9,
                color=prefs.color_title,
                ha="center",
            )

        # Horizontal Rule (Separates Header from Content)
        line = Line2D(
            [pc.left_margin, pc.right_margin],
            [0.92, 0.92],
            transform=fig.transFigure,
            color=prefs.color_neutral,
            lw=0.5,
            alpha=0.3,
        )
        fig.add_artist(line)

    def _draw_page_footer(self, pdf_page: PDFDocument.Page) -> None:
        """Render the standard footer with hardware context and quality tag."""
        fig = pdf_page.fig
        fig.text(
            0.05,
            0.02,
            self.hardware_stats_text,
            ha="left",
            fontsize=8,
            color=prefs.color_neutral,
            style="italic",
            alpha=0.8,
            transform=fig.transFigure,
        )

        # Quality Score on the right (Semantic Success/Danger)
        model_quality = self.best_model.model_quality
        fig.text(
            0.95,
            0.02,
            self.best_model.quality_score_text,
            ha="right",
            fontsize=8,
            color=model_quality.color,
            style="italic",
            weight=model_quality.text_weight,
            transform=fig.transFigure,
        )

    def _render_title_page(self) -> None:
        """Render the cover/title page for the report."""
        if not self.results:
            return

        pdf_page = self.pdf_doc.create_new_page(
            page_name="Title",
            include_header=False,
            include_footer=False,
            include_in_page_numbering=False,
            print_page_name=False,
            include_in_index=False,
        )
        fig = pdf_page.fig

        # Layout
        fig.text(
            0.5,
            0.65,
            apply_tracking("PERFORMANCE AUDIT REPORT"),
            fontsize=14,
            color=prefs.color_title,
            alpha=0.7,
            ha="center",
        )
        fig.text(
            0.5,
            0.60,
            self.report_title.upper(),
            fontsize=28,
            weight="black",
            color=prefs.color_title,
            ha="center",
        )
        fig.text(
            0.5,
            0.54,
            f"Subject Dataset: {self.summary.dataset_name}",
            fontsize=12,
            ha="center",
        )

        rec_color = self.best_model.recommendation.color

        # Badge
        fig.text(
            0.5,
            0.42,
            self.best_model.recommendation.action,
            fontsize=18,
            weight="bold",
            color="white",
            ha="center",
            bbox=dict(
                facecolor=rec_color,
                edgecolor=rec_color,
                boxstyle="round,pad=0.8",
                lw=0,
            ),
        )

        # Quality Score Highlight
        fig.text(
            0.5,
            0.32,
            self.best_model.quality_score_text.upper(),
            fontsize=12,
            weight="bold",
            color=prefs.color_neutral,
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

        # Footer
        processed_rows_format = ValueDescFormat(
            precision=0, description="rows", description_leading_space=True
        )

        footer_text = (
            f"Processed {processed_rows_format.format_value(self.summary.processed_row_count)} | "
            f"Hardware Safety Limits: Respected"
        )
        fig.text(
            0.5,
            0.10,
            footer_text,
            fontsize=9,
            color=prefs.color_neutral,
            alpha=0.6,
            ha="center",
            va="center",
            transform=fig.transFigure,
        )
        timestamp_format = DateTimeFormat(
            date_format="%Y-%m-%d", time_format="%H:%M:%S"
        )
        fig.text(
            0.5,
            0.08,
            f"Generated on {timestamp_format.format_value(datetime.now())}",
            fontsize=8,
            color=prefs.color_neutral,
            alpha=0.4,
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

    def _get_risk_profile(
        self,
    ) -> str:
        """Generate a human-readable risk profile summary."""
        risks = []
        model = self.best_model.model

        # 1. Distribution Shape Risk
        if model.train_skew > 2.0:
            risks.append("Extreme Skew: Model may over-predict.")
        if model.train_kurtosis > 10.0:
            risks.append(
                "Fat-Tails: Outliers exert high leverage on performance stats."
            )

        # 2. Generic Pattern Detection
        # We ignore the synthetic columns (Actual/Predicted) for pattern searching
        # We look only at the 'dynamic_features' (the culprits)
        potential_concentrated_risks = []
        if self.summary.anomaly_data is not None:
            risk_percentage_format = PercentageFormat(precision=0)
            for feat in self.summary.anomaly_dynamic_features:
                # Get the values for this feature in our anomalies
                outlier_values = self.summary.anomaly_data[feat]

                # Check if they all share the same value (Mode frequency)
                mode_val = outlier_values.mode()
                if not mode_val.empty:
                    frequency = (outlier_values == mode_val[0]).sum() / len(
                        outlier_values
                    )

                    # If anomalies share the same value for a high-kurtosis feature
                    if frequency >= self.summary.anomaly_risk_concentration_threshold:
                        # Format the value for the report
                        feat_fmt = self.summary.features[feat].formatter
                        val_str = feat_fmt.format_value(mode_val[0])
                        msg = f"CONCENTRATED RISK: {risk_percentage_format.format_value(frequency)} of top errors share {feat}='{val_str}'."
                        potential_concentrated_risks.append((frequency, msg))

        # Sort potential risks by frequency (highest first), and take the top 5
        potential_concentrated_risks.sort(key=lambda x: x[0], reverse=True)
        risks.extend(r[1] for r in potential_concentrated_risks[:5])

        return (
            "\n".join(risks)
            if risks
            else "Nominal: No significant distribution risks detected."
        )

    def _render_metric_table(
        self,
        pdf_page: PDFDocument.Page,
        top_y: float,
        df: pd.DataFrame,
    ) -> TableLayout:
        """Render a two-column metric/value table and return its layout."""
        table_columns: dict[str, TableColumn] = {
            "Metric": TableColumn(
                detail_style=TableColumnStyle(
                    alpha=0.8,
                    ha="right",
                ),
                first_row_style=TableColumnStyle(
                    edge_color=TableEdgeColor(),
                    alpha=0.8,
                    ha="right",
                ),
                rpad=15.0,
            ),
            "Value": TableColumn(
                detail_style=TableColumnStyle(fontweight="bold", ha="left"),
                first_row_style=TableColumnStyle(
                    fontweight="bold",
                    edge_color=TableEdgeColor(),
                    ha="left",
                ),
                lpad=15.0,
            ),
        }
        pc = self.pdf_doc.page_configuration
        table_max_height = top_y - pc.bottom_margin
        table = Table(
            data=df,
            max_table_height=table_max_height,
            mid_x=0.5,
            top_y=top_y,
            fontsize=11,
            columns=table_columns,
            cell_edge_linewidth=TableEdgeLinewidth(bottom=0.8),
            table_edge_linewidth=TableEdgeLinewidth.all_edges(linewidth=0.8),
            table_edge_padding=(12.0, 12.0, 6.0, 6.0),
            table_edge_color=TableEdgeColor.closed(color=prefs.color_title),
            include_headers=False,
            detail_tpad=12.0,
            detail_bpad=12.0,
        )

        return render_table(
            pdf_page=pdf_page,
            table=table,
        )

    def _render_executive_summary(self) -> None:
        """Render the executive summary page with headline and metrics."""
        if len(self.results) == 0:
            return

        pdf_page = self.pdf_doc.create_new_page(page_name="Executive Summary")
        fig = pdf_page.fig
        rec = self.best_model.recommendation
        headline_top_y = 0.85
        risk_assessment_header_top_y = 0.75

        # Main Action Text (The "Headline")
        fig.text(
            0.5,
            headline_top_y,
            rec.action,
            fontsize=22,
            weight="bold",
            color=rec.color,
            ha="center",
            va="center",
            transform=fig.transFigure,
            bbox=dict(
                facecolor="white",
                edgecolor=rec.color,
                boxstyle="round,pad=0.6",  # Increased pad for "Executive" feel
                lw=2.5,  # Slightly thicker border
            ),
        )

        # Risk Assessment Line
        fig.text(
            0.5,
            risk_assessment_header_top_y,
            "*** AUDIT RISK PROFILE ***",
            fontsize=11,
            color=prefs.color_neutral,
            weight="black",  # Extra bold to differentiate from sub-text
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

        risk_text = self._get_risk_profile()
        risk_text_left_x = 0.27
        risk_text_top_y = 0.72
        risk_text_artist = fig.text(
            risk_text_left_x,
            risk_text_top_y,
            risk_text.upper(),
            fontsize=10,
            color=prefs.color_neutral,
            weight="black",  # Extra bold to differentiate from sub-text
            ha="left",
            va="top",
            linespacing=1.8,
            transform=fig.transFigure,
        )

        renderer = fig.canvas.get_renderer()  # type: ignore

        if renderer is None:
            fig.draw_without_rendering()
            renderer = fig.canvas.get_renderer()  # type: ignore

        risk_text_bbox = get_artist_bbox(
            obj=risk_text_artist, transform_to=fig, renderer=renderer
        )
        risk_text_bot_y = risk_text_top_y - risk_text_bbox.height
        _ = self._render_metric_table(
            pdf_page=pdf_page,
            top_y=risk_text_bot_y - 0.05,
            df=self.executive_summary_metrics,
        )

    def _render_features_page(self) -> None:
        """Render the feature list page with metadata table."""
        header_top_y = 0.88
        pdf_page = self.pdf_doc.create_new_page(page_name="Feature List")
        pdf_page.continuation_text = "(cont.)"
        pdf_page.continuation_page_top_y = header_top_y
        fig = pdf_page.fig

        # Header
        fig.text(
            0.5,
            header_top_y,
            "Features",
            fontsize=16,
            weight="bold",
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

        column_header_edge_color = TableEdgeColor(
            left=prefs.color_neutral, right=prefs.color_neutral
        )
        column_headers: list[str] = [
            "ID",
            "Feature",
            "Pos",
            "Short Name",
            "Parent Name",
            "Used in Fit",
            "Description",
        ]
        header_style = TableColumnStyle(
            fontweight="bold",
            ha="center",
            va="center",
            edge_color=column_header_edge_color,
            face_color="black",
            text_color="white",
        )
        detail_style_left = TableColumnStyle(
            ha="left",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            text_color=prefs.color_neutral,
        )
        even_row_style_left = TableColumnStyle(
            ha="left",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            face_color=prefs.color_light_gray,
            text_color=prefs.color_neutral,
        )
        detail_style_right = TableColumnStyle(
            ha="right",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            text_color=prefs.color_neutral,
        )
        even_row_style_right = TableColumnStyle(
            ha="right",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            face_color=prefs.color_light_gray,
            text_color=prefs.color_neutral,
        )
        detail_style_center = TableColumnStyle(
            ha="center",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            text_color=prefs.color_neutral,
        )
        even_row_style_center = TableColumnStyle(
            ha="center",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            face_color=prefs.color_light_gray,
            text_color=prefs.color_neutral,
        )

        table_columns: dict[str, TableColumn] = {}
        for col in column_headers:
            if col == "Pos":
                detail_style = detail_style_right
                even_row_style = even_row_style_right
            elif col == "Used in Fit":
                detail_style = detail_style_center
                even_row_style = even_row_style_center
            else:
                detail_style = detail_style_left
                even_row_style = even_row_style_left

            table_columns[col] = TableColumn(
                header_style=header_style,
                detail_style=detail_style,
                even_row_style=even_row_style,
                lpad=12.0,
                rpad=12.0,
            )

        table_data: list[list[str]] = []
        position_format = IntegerFormat()
        parent_name_format = StringFormat(fallback="")
        used_in_fit_format = BoolFormat(representation=BoolRepresentation.YES_NO)

        for feature_metadata in self.summary.features.values():
            table_data.append(
                [
                    feature_metadata.id,
                    feature_metadata.name,
                    position_format.format_value(feature_metadata.position),
                    feature_metadata.short_name,
                    parent_name_format.format_value(feature_metadata.parent_name),
                    used_in_fit_format.format_value(feature_metadata.is_used_in_fit),
                    feature_metadata.description,
                ]
            )

        table_df = pd.DataFrame(table_data, columns=list(table_columns.keys()))
        pc = self.pdf_doc.page_configuration
        table_top_y = header_top_y - 0.05
        table_max_height = table_top_y - pc.bottom_margin
        table = Table(
            data=table_df,
            max_table_height=table_max_height,
            mid_x=0.5,
            top_y=table_top_y,
            fontsize=9,
            columns=table_columns,
            cell_edge_linewidth=TableEdgeLinewidth.all_edges(0.4),
            table_edge_linewidth=TableEdgeLinewidth.all_edges(0.0),
            table_edge_color=TableEdgeColor.closed(color=prefs.color_title),
            header_tpad=8.0,
            header_bpad=8.0,
            detail_tpad=6.0,
            detail_bpad=6.0,
        )

        _ = render_table(pdf_page=pdf_page, table=table)

    def _render_anomaly_page(self) -> None:
        """Render the anomaly summary page, if anomaly data exists."""
        anomaly_data = (
            self.summary.anomaly_data
            if self.summary.anomaly_data is not None
            else pd.DataFrame()
        )
        anomaly_dynamic_features = (
            self.summary.anomaly_dynamic_features
            if self.summary.anomaly_dynamic_features is not None
            else []
        )

        pdf_page = self.pdf_doc.create_new_page(
            page_name="Data Anomaly Log", print_page_name=False
        )
        fig = pdf_page.fig

        ax_table = fig.add_subplot(1, 1, 1)
        ax_table.axis("off")
        model = self.best_model.model
        header_top_y = 0.88
        subheader_top_y = 0.85
        model_type_format = EnumFormat()
        kurtosis_format = FloatFormat(precision=2)

        # Header
        fig.text(
            0.5,
            header_top_y,
            f"Data Anomaly Log: {model_type_format.format_value(model.model_type)}",
            fontsize=16,
            weight="bold",
            ha="center",
            va="center",
            transform=fig.transFigure,
        )
        fig.text(
            0.5,
            subheader_top_y,
            f"Top {self.summary.top_n_anomalies} Primary Contributors to a Dataset Kurtosis of {kurtosis_format.format_value(model.val_kurtosis)} (out of {self.formatted_processed_row_count})",
            fontsize=10,
            color="gray",
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

        # Create Table Data
        # Resolve Dynamic Features to Display Names
        # We use a dict.fromkeys() trick to keep the order and remove duplicates
        # (e.g., if two dynamic features map to the same display feature, we show the display feature only once)
        resolved_features = list(
            dict.fromkeys(
                [
                    self.summary.anomaly_display_map.get(f, f)
                    for f in anomaly_dynamic_features
                ]
            )
        )
        feature_short_names = list(
            dict.fromkeys(
                [self.summary.features[f].short_name for f in resolved_features]
            )
        )

        AUDIT_ANOMALY_ACTUAL_COL = "_audit_Actual"
        AUDIT_ANOMALY_PREDICTED_COL = "_audit_Predicted"
        AUDIT_ANOMALY_ABS_ERROR_COL = "_audit_Abs_Error"
        AUDIT_ANOMALY_ACTUAL_COL_HEADER = "Actual"
        AUDIT_ANOMALY_PREDICTED_COL_HEADER = "Predicted"
        AUDIT_ANOMALY_ABS_ERROR_COL_HEADER = "Abs Error"

        # Add the "Smoking Gun" columns back in
        columns_to_show = [
            AUDIT_ANOMALY_ACTUAL_COL,
            AUDIT_ANOMALY_PREDICTED_COL,
            AUDIT_ANOMALY_ABS_ERROR_COL,
        ] + resolved_features
        column_headers = [
            AUDIT_ANOMALY_ACTUAL_COL_HEADER,
            AUDIT_ANOMALY_PREDICTED_COL_HEADER,
            AUDIT_ANOMALY_ABS_ERROR_COL_HEADER,
        ] + feature_short_names
        table_data: list[list[str]] = []
        column_formats: dict[str, FormatConfig] = {}
        table_columns: dict[str, TableColumn] = {}
        column_header_edge_color = TableEdgeColor(
            left=prefs.color_neutral, right=prefs.color_neutral
        )
        col_index = 0
        for feat in columns_to_show:
            if feat == AUDIT_ANOMALY_ACTUAL_COL:
                fmt = self.summary.actual_value_fmt
            elif feat == AUDIT_ANOMALY_PREDICTED_COL:
                fmt = self.summary.predicted_value_fmt
            elif feat == AUDIT_ANOMALY_ABS_ERROR_COL:
                fmt = self.summary.abs_error_fmt
            else:
                fmt = self.summary.features[feat].formatter

            column_formats[feat] = fmt
            text_alignment = fmt.matplot_alignment()
            header_style = TableColumnStyle(
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                edge_color=column_header_edge_color,
                face_color="black",
                text_color="white",
            )
            detail_style = TableColumnStyle(
                ha=text_alignment,
                va="center",
                fontsize=10,
                fontfamily="monospace",
                edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                face_color=prefs.color_paper,
                text_color=prefs.color_neutral,
            )
            even_row_style = TableColumnStyle(
                ha=text_alignment,
                va="center",
                fontsize=10,
                fontfamily="monospace",
                edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                face_color=prefs.color_light_gray,
                text_color=prefs.color_neutral,
            )
            table_columns[column_headers[col_index]] = TableColumn(
                detail_style=detail_style,
                header_style=header_style,
                even_row_style=even_row_style,
                lpad=20.0,
                rpad=20.0,
                has_consistent_width=True,
                has_consistent_height=True,
            )
            col_index += 1

        for _, row in anomaly_data.head(self.summary.top_n_anomalies).iterrows():
            formatted_row: list[str] = []
            for feat in columns_to_show:
                val = row[feat]
                fmt = column_formats[feat]
                formatted_val = fmt.format_value(val=val)
                formatted_row.append(formatted_val)
            table_data.append(formatted_row)

        table_df = pd.DataFrame(table_data, columns=column_headers)
        pc = self.pdf_doc.page_configuration
        table_top_y = subheader_top_y - 0.05
        table_max_height = table_top_y - pc.bottom_margin
        table = Table(
            data=table_df,
            max_table_height=table_max_height,
            mid_x=0.5,
            top_y=table_top_y,
            fontsize=10,
            columns=table_columns,
            cell_edge_linewidth=TableEdgeLinewidth.all_edges(0.4),
            table_edge_linewidth=TableEdgeLinewidth.all_edges(0.0),
            table_edge_color=TableEdgeColor.closed(color=prefs.color_title),
            header_tpad=8.0,
            header_bpad=8.0,
            detail_tpad=6.0,
            detail_bpad=6.0,
        )

        _ = render_table(pdf_page=pdf_page, table=table)

    def _render_model_legend(self) -> None:
        """Render the model type legend/glossary page."""
        pdf_page = self.pdf_doc.create_new_page(
            page_name="Legend / Glossary", print_page_name=False
        )
        fig = pdf_page.fig

        # Header
        fig.text(
            0.5, 0.88, "Model Audit Legend", fontsize=16, weight="bold", ha="center"
        )

        # Prepare data
        model_type_data = ModelTypeData.get_list(
            sort_order=ModelEnumSortOrder.TASK_TYPE_NAME, include_task_type_headers=True
        )
        legend_len = len(model_type_data)
        mid = legend_len // 2 + legend_len % 2

        # Layout constants
        line_height = 0.04
        start_y = 0.80
        left_x = 0.15
        right_x = 0.55
        bar_width = 0.015
        text_offset = 0.025

        for i, mtd in enumerate(model_type_data):
            # Determine column and position
            is_left = i < mid
            col_x = left_x if is_left else right_x
            row_idx = i if is_left else i - mid
            current_y = start_y - (row_idx * line_height)

            if mtd.rec_type == ModelTypeDataRecType.DATA:
                # Get the color assigned to this specific model
                color = self.summary.solid_color_palette[mtd.value]

                # Add the Color Bar (Rectangle)
                # We add the patch to the figure directly using fig.add_artist
                rect = Rectangle(
                    (col_x, current_y - 0.008),
                    bar_width,
                    0.025,
                    facecolor=color,
                    transform=fig.transFigure,
                    clip_on=False,
                )
                fig.patches.append(rect)

                # Add the Label (Abbreviation + Full Name)
                mtd_format = EnumFormat()
                label = f"{mtd.abbrev}: {mtd_format.format_value(mtd.model_type)}"
                fig.text(
                    col_x + text_offset,
                    current_y,
                    label,
                    fontsize=11,
                    family="monospace",
                    va="center",
                    transform=fig.transFigure,
                )
            else:
                # Add the Label (Abbreviation + Full Name)
                task_type_format = EnumFormat()
                label = f"--- {task_type_format.format_value(mtd.task_type)} ---"
                fig.text(
                    col_x + text_offset,
                    current_y,
                    label,
                    fontsize=11,
                    family="monospace",
                    va="center",
                    weight="bold",
                    transform=fig.transFigure,
                )

        fig.text(
            0.5,
            0.1,
            "Note: Metric 'Score' refers to R² for Regression and Accuracy for Classification.",
            fontsize=9,
            style="italic",
            ha="center",
            transform=fig.transFigure,
        )

    def _get_audit_data(self) -> Tuple[pd.DataFrame, dict]:
        """
        Parses the ModelConfiguration objects into two DataFrames:
        1. Performance metrics (one row per model)
        2. Feature importances (long-form for plotting)
        """
        performance_rows = []
        importance_dict: dict = {}

        for config in self.results:
            # 1. Extract Performance Metrics
            performance_rows.append(
                {
                    "ID": config.id,
                    "Model": config.model_type.value,
                    "Abbr": config.model_type.abbrev,
                    "Strategy": config.balancing_strategy,
                    "Available RAM": config.available_gb,
                    "Est Peak RAM": config.estimated_peak_gb,
                    "Actual Peak RAM": config.actual_peak_gb,
                    "Memory Risk": config.memory_risk_triggered,
                    "Sampling Pct": config.sampling_factor,
                    "n_jobs": config.n_jobs,
                    "CV Score (Tuning)": config.score_cv,
                    "Val Score": config.val_score,
                    "Test Score": (
                        config.r2_test
                        if config.task_type is TaskType.REGRESSION
                        else config.accuracy_test
                    ),
                    "Cleaned Score": config.score_val_cleaned,
                    "MAE": config.mae_val,
                    "Train Time (s)": config.total_duration,
                    "Efficiency": config.efficiency,
                    "Train Score": config.train_score,
                    "Gap": config.gap,
                    "Status": config.model_generalization.value,
                    "Mean Delta": config.mean_delta,
                }
            )

            # 2. Extract Feature Importances (if they exist)
            if hasattr(config, "feature_analysis") and config.feature_analysis:
                imp_df = config.feature_analysis.feature_importances.copy()
            else:
                imp_df = pd.DataFrame()

            importance_dict[config.id] = imp_df

        # Combine results
        perf_df = pd.DataFrame(performance_rows)
        return perf_df, importance_dict

    def _check_data_leakage(self):
        """Checks if the Train/Val distributions have drifted apart."""
        model = self.best_model.model
        is_safe = model.drift_index < self.summary.drift_threshold

        status = "SAFE" if is_safe else "WARNING: DRIFT DETECTED"
        color = prefs.color_success if is_safe else prefs.color_danger

        return status, color, model.drift_index

    def _plot_predictive_accuracy(self, sns, ax):
        """Plot validation accuracy bars with optional cleaned-score overlay."""
        has_cleaned = (
            "Cleaned Score" in self.summary_df.columns
            and self.summary_df["Cleaned Score"].notnull().any()
            and not (
                self.summary_df["Cleaned Score"] == self.summary_df["Val Score"]
            ).all()
        )

        # Plot the "Raw" actual score (The Reality)
        sns.barplot(
            x="Val Score",
            y="Model",
            data=self.summary_df,
            ax=ax,
            alpha=0.9,
            label="Raw Score (Actual)",
            hue="Model",
            palette=self.summary.solid_color_palette,
            legend=False,  # Prevent duplicate entries in the final legend
        )

        # Conditionally, overlay the "Cleaned" potential score (The Shadow)
        if has_cleaned:
            sns.barplot(
                x="Cleaned Score",
                y="Model",
                data=self.summary_df,
                ax=ax,
                alpha=0.3,
                label="Outliers Filtered (Potential)",
                hue="Model",
                palette=self.summary.light_color_palette,
                legend=False,  # Prevent duplicate entries in the final legend
                zorder=0,  # Keep the shadow behind the solid bar
            )

        score_desc = (
            "F1"
            if self.best_model.model.task_type == TaskType.CLASSIFICATION
            else "$R^2$"
        )

        # Create a Manual Legend (The "Professional Audit" Fix)
        # Create custom "swatches" for the legend
        legend_handles = [
            mpatches.Patch(
                color=prefs.color_neutral, alpha=0.9, label="Raw Score (Actual)"
            )
        ]

        if has_cleaned:
            legend_handles.append(
                mpatches.Patch(
                    color=prefs.color_neutral,
                    alpha=0.5,
                    label="Outliers Filtered (Potential)",
                )
            )
            title_suffix = "& Outlier Impact"
        else:
            title_suffix = "- Baseline Performance"

        # Apply the manual legend
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=8,
            frameon=True,
        )

        ax.set_title(f"Predictive Accuracy ({score_desc}) {title_suffix}")

        # Clean up axes
        ax.set_xlabel(f"Score ({score_desc})", fontsize=10)
        ax.set_ylabel("")  # Model names are usually self-explanatory
        ax.set_yticks(range(len(self.summary_df)))
        ax.set_yticklabels(
            self.summary_df["Abbr"], ha="left", va="center", position=(-0.08, -0.02)
        )

        # Narrow the focus so the difference is visible
        current_min = self.summary_df["Val Score"].min()
        ax.set_xlim(left=max(0, current_min - 0.05), right=1.0)

        # Add text labels for the 'Raw Score'
        # The first half of patches are 'Cleaned Score' bars
        # The second half are 'Val Score' bars
        raw_score_bars = cast(List[Rectangle], ax.patches[len(self.summary_df) :])

        # Add text labels for the 'Raw Score'
        # Note: Since Seaborn 0.13+, patches are often grouped differently.
        # It's safer to use the actual dataframe values for coordinates.

        x_min = ax.get_xlim()[0]  # The current 'left' boundary of our zoomed view

        for _, p in enumerate(raw_score_bars):
            width = p.get_width()
            y_pos = p.get_y()
            height = p.get_height()

            # The visual 'start' of the bar is either 0 or x_min
            bar_start = max(0, x_min)

            # Calculate center relative to the VISIBLE portion of the bar
            if width > (x_min + 0.15):
                text_color = prefs.color_paper
                # Calculate the midpoint between where the axis starts and the bar ends
                text_pos = (bar_start + width) / 2
                text_align = "center"
            else:
                text_color = prefs.color_neutral
                text_pos = width + 0.005  # Just a tiny bit past the bar
                text_align = "left"

            width_format = FloatFormat(precision=4)
            ax.text(
                text_pos,
                y_pos + height / 2,
                f"{width_format.format_value(width)}",
                va="center",
                ha=text_align,
                color=text_color,
                fontsize=10,
                weight="bold",
            )

    def _plot_efficiency_scatter(self, sns, ax):
        """Plot efficiency as accuracy vs. training time scatter plot."""
        sns.scatterplot(
            x="Train Time (s)",
            y="Val Score",
            hue="Model",
            size="MAE",
            sizes=(100, 500),
            data=self.summary_df,
            palette=self.summary.solid_color_palette,
            ax=ax,
        )

        ax.set_xscale("linear")
        ax.set_title("Efficiency: Accuracy vs. Time")

        # Fix the Legend: bbox_to_anchor helps, but we should also refine the labels
        handles, labels = ax.get_legend_handles_labels()

        # Filter out the headers ("Model" and "MAE") and the size samples
        # We keep only the items until the list hits the 'MAE' section
        try:
            stop_idx = labels.index("MAE")
            final_handles = handles[:stop_idx]
            final_labels = labels[:stop_idx]

            # If "Model" is the first label, we can skip that too
            if final_labels[0] == "Model":
                final_handles = final_handles[1:]
                final_labels = final_labels[1:]
        except ValueError:
            final_handles, final_labels = handles, labels

        # Create Note handle
        final_handles.append(mpatches.Patch(color="none"))
        final_labels.append("")

        final_handles.append(
            mpatches.Patch(color="none", label="Note: Bubble size = MAE")
        )
        final_labels.append("Note: Bubble size = MAE\n(Smaller is better)")

        leg = ax.legend(
            handles=final_handles,
            labels=final_labels,
            title=None,
            loc="best",
            fontsize=8,
            handlelength=0,  # This hides the 'icon' space for the text-only note
            frameon=True,
            ncol=2,
            columnspacing=0.8,
        )

        # Make the note gray
        plt.setp(
            leg.get_texts()[-1], color=prefs.color_neutral, fontsize=7, style="italic"
        )

    def _get_narrow_x_limit(self, y_train, y_val, percentile=99.5):
        """Compute narrow x-axis limits using percentile clipping."""
        # Combine or take the max of both sets to ensure both fit
        t_limit = np.percentile(y_train, percentile)
        v_limit = np.percentile(y_val, percentile)

        upper_limit = max(t_limit, v_limit)

        # Check the 0.5 percentile just in case of negative data errors
        lower_limit = min(np.percentile(y_train, 0.5), 0)

        return lower_limit, upper_limit

    def _plot_target_distribution(
        self,
        ax,
    ):
        """Visualizes the distribution of the target across rows."""
        model = self.best_model.model
        data_splits = self.summary.data_splits
        y_train = data_splits.train_target
        y_val = data_splits.val_target

        # Get the narrow limits
        x_min, x_max = self._get_narrow_x_limit(y_train, y_val)

        # Plot the distributions
        sns.kdeplot(
            x=y_train,
            ax=ax,
            label="Train",
            color=prefs.color_neutral,
            fill=True,
            alpha=0.2,
        )
        sns.kdeplot(
            x=y_val,
            ax=ax,
            label="Val",
            color=prefs.color_classic_blue,
            fill=True,
            alpha=0.2,
        )

        # Apply limits with a tiny bit of breathing room
        ax.set_xlim(x_min, x_max * 1.05)

        # Add a note if data was clipped (Professional Audit touch)
        if y_train.max() > x_max:
            text = ax.text(
                0.02,
                0.02,
                "* Distribution clipped at 99.5th percentile",
                fontsize=8,
                style="italic",
                color=prefs.color_neutral,
                transform=ax.transAxes,
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=3, foreground="white", alpha=0.7),
                    path_effects.Normal(),
                ]
            )

        ax.set_title(f"Target Distribution: Train vs. Val Balance")
        ax.set_xlabel(f"Target Value ({data_splits.target_column})")
        ax.set_ylabel("Frequency")
        ax.legend()

        # Draw Vertical Lines
        ax.axvline(
            model.train_mean,
            color=prefs.color_neutral,
            linestyle="--",
            linewidth=1.5,
            label="Mean",
        )
        ax.axvline(
            model.train_median,
            color=prefs.color_neutral,
            linestyle=":",
            linewidth=1.5,
            label="Median",
        )
        ax.axvline(
            model.val_mean,
            color=prefs.color_classic_blue,
            linestyle="--",
            linewidth=1.5,
        )
        ax.axvline(
            model.val_median,
            color=prefs.color_classic_blue,
            linestyle=":",
            linewidth=1.5,
        )

        # Construct the Stats Box String
        stats_format = FloatFormat(precision=2)
        delta_format = PercentageFormat(precision=4, width=8)
        stats_text = (
            f"TRAINING BASELINE\n"
            f"Mean:  {stats_format.format_value(model.train_mean)} (σ: {stats_format.format_value(model.train_std)})\n"
            f"Skew:  {stats_format.format_value(model.train_skew)} | Kurt: {stats_format.format_value(model.train_kurtosis)}\n\n"
            f"AUDIT VALIDATION\n"
            f"Mean:  {stats_format.format_value(model.val_mean)} (σ: {stats_format.format_value(model.val_std)})\n"
            f"Skew:  {stats_format.format_value(model.val_skew)} | Kurt: {stats_format.format_value(model.val_kurtosis)}\n\n"
            f"SPLIT INTEGRITY CHECK\n"
            f"Mean Δ: {delta_format.format_value(model.mean_delta)}\n"
            f"σ Δ:    {delta_format.format_value(model.std_delta)}"
        )

        at = AnchoredText(
            stats_text,
            prop=dict(
                size=7.5,
                family="monospace",
                color=prefs.color_neutral,  # Audit Blue Text
                linespacing=1.4,
            ),
            frameon=True,
            loc="upper right",
            borderpad=1.0,
        )

        # Style the box to match the Audit Branding
        at.patch.set_boxstyle("round,pad=0.5,rounding_size=0.2")
        at.patch.set_edgecolor(prefs.color_neutral)
        at.patch.set_facecolor(prefs.color_paper)  # Paper White background
        at.patch.set_alpha(0.9)
        at.patch.set_linewidth(0.8)

        ax.add_artist(at)
        ax.set_title("Target Distribution & Statistical Shape")
        ax.legend(fontsize=7, loc="upper left")

    def _plot_cumulative_importance(self, sns, ax) -> None:
        """Plot cumulative feature importance for the best model."""
        model = self.best_model.model
        best_model_name = model.model_type.value
        best_model_importance = self.importance_dict[model.id]
        best_model_importance["feature_idx"] = range(1, len(best_model_importance) + 1)

        sns.lineplot(
            x="feature_idx",
            y="cumulative_importance",
            data=best_model_importance,
            marker="o",
            color=self.summary.solid_color_palette[best_model_name],
            ax=ax,
            zorder=1,
        )
        # Define feature colors for the top points
        n_labels = 5  # Label top 5 features
        colors = sns.color_palette("tab10", n_colors=10)

        # Plot individual colored dots and capture for legend
        legend_handles = []
        for i in range(len(best_model_importance)):
            cum_imp = best_model_importance["cumulative_importance"].iloc[i]
            feat_name = best_model_importance["feature"].iloc[i]

            # Color top features or those near the threshold
            if i < n_labels or (
                i > 0
                and best_model_importance["cumulative_importance"].iloc[i - 1] < 0.95
            ):
                dot_color = colors[i % len(colors)]
                dot = ax.plot(
                    i + 1,
                    cum_imp,
                    marker="o",
                    color=dot_color,
                    markersize=6,
                    label=feat_name,
                    linestyle="None",
                    zorder=3,
                )
                legend_handles.append(dot[0])

            if cum_imp >= 0.95 and i >= n_labels:
                break

        # Add Threshold line and Legend
        thresh_line = ax.axhline(
            y=0.95,
            color=prefs.color_neutral,
            linestyle="--",
            label="95% Threshold",
            zorder=2,
        )

        ax.legend(
            handles=[thresh_line] + legend_handles,
            loc="lower right",
            fontsize=7,
            frameon=True,
            title="Top Features",
            ncol=2,
            columnspacing=0.8,
        )
        ax.set_title(f"Cumulative Importance\n({best_model_name})")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Total Variance Explained")
        ax.set_ylim(0, 1.05)

    def _render_audit_results(self) -> None:
        """
        Generates a professional performance dashboard from audit results.
        """
        pdf_page = self.pdf_doc.create_new_page(
            page_name="High-Level Competition Results"
        )
        fig = pdf_page.fig

        with sns.axes_style(style="whitegrid"):
            # Quadrant Charts
            #    Accuracy Bar Chart (Top Left)
            #    Efficiency Scatter (Top Right)
            #    Target Distribution (Bottom Left)
            #    Cumulative Importance (Bottom Right)
            gs = fig.add_gridspec(2, 2, height_ratios=[0.5, 0.5])
            ax_acc = fig.add_subplot(gs[0, 0])
            ax_eff = fig.add_subplot(gs[0, 1])
            ax_dist = fig.add_subplot(gs[1, 0])
            ax_imp = fig.add_subplot(gs[1, 1])

            # Add the Leakage Warning to the footer or as a floating text box
            status, color, diff = self._check_data_leakage()
            fig.text(
                0.5,
                0.04,
                f"Data Integrity Check: {status} (Drift: {prefs.drift_format.format_value(diff)})",
                ha="center",
                weight="bold",
                color=color,
                fontsize=12,
            )

            # Performance Comparison (F1 / R-Squared)
            self._plot_predictive_accuracy(sns, ax_acc)

            # Efficiency Scatter (Accuracy vs. Training Time)
            self._plot_efficiency_scatter(sns, ax_eff)

            # Target Distribution
            self._plot_target_distribution(ax_dist)

            # Cumulative Importance Line Plot
            self._plot_cumulative_importance(sns, ax_imp)

    def _render_cv_vs_final(self) -> None:
        """Render a CV vs. final validation comparison page."""
        pdf_page = self.pdf_doc.create_new_page(page_name="Generalization (CV vs Full)")
        fig = pdf_page.fig
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        # Plot the bars
        sns.barplot(
            data=self.summary_df,
            x="CV Score (Tuning)",
            y="Model",
            hue="Model",
            palette=self.summary.light_color_palette,
            alpha=0.7,
            hatch="//",
            ax=ax,
            legend=False,
        )
        sns.barplot(
            data=self.summary_df,
            x="Val Score",
            y="Model",
            hue="Model",
            palette=self.summary.solid_color_palette,
            alpha=0.9,
            ax=ax,
            legend=False,
        )

        # THE BULLETPROOF LABEL LOGIC
        # We iterate through EVERY rectangle in the chart
        for p in ax.patches:
            rect = cast(Rectangle, p)
            # Only look at the "Solid" bars (the ones with no hatching)
            if rect.get_hatch() is None:
                # Find which row in the dataframe matches this bar's Y position
                # rect.get_y() + height/2 is the center of the bar
                y_center = rect.get_y() + rect.get_height() / 2

                # In Seaborn, model 0 is at y=0, model 1 is at y=1, etc.
                row_idx = int(round(y_center))

                if 0 <= row_idx < len(self.summary_df):
                    final_val = float(rect.get_width())
                    delta = self.summary_df.iloc[row_idx]["Mean Delta"]

                    # Only print if the width isn't 0 (safeguard)
                    if final_val > 0:
                        t_color = (
                            prefs.color_success if delta > 0 else prefs.color_danger
                        )
                        delta_format = FloatFormat(
                            precision=4, always_include_sign=True
                        )
                        ax.text(
                            final_val + 0.01,
                            y_center,
                            f"Δ: {delta_format.format_value(delta)}",
                            va="center",
                            ha="left",
                            fontsize=9,
                            color=t_color,
                            weight="bold",
                        )

        # Legend & Formatting (Audit Blue)
        cv_patch = mpatches.Patch(
            facecolor=prefs.color_neutral,
            alpha=0.6,
            hatch="//",
            label="CV Score (Tuning)",
        )
        val_patch = mpatches.Patch(
            facecolor=prefs.color_neutral, alpha=0.8, label="Final Score (Full Audit)"
        )
        ax.legend(handles=[cv_patch, val_patch], loc="lower right", title="Audit Stage")
        ax.set_ylabel("")
        ax.set_yticks(range(len(self.summary_df)))
        ax.set_yticklabels(
            self.summary_df["Abbr"], ha="left", va="center", position=(-0.05, 0.0)
        )
        ax.set_xlim(0, 1.15)

    def _plot_model_hyperparameters(
        self,
        pdf_page: PDFDocument.Page,
        config: ModelConfiguration,
        ax_params: Axes,
        renderer: RendererBase,
    ):
        """Render a hyperparameter table for a single model configuration."""
        fig = pdf_page.fig
        ax_params.axis("off")
        ax_params.set_xlim(0.0, 1.0)
        ax_params.set_ylim(0.0, 1.0)
        ax_params.set_title(f"Configuration")
        dials = dataclasses.asdict(config.model_params)
        table_data = []

        for k, v in dials.items():
            if isinstance(v, Enum):
                val = v.value
            else:
                val = v

            table_data.append([prefs.get_hyperparmeter_display_name(k), str(val)])

        columns = [
            "HP",
            "Value",
        ]
        df = pd.DataFrame(table_data, columns=columns)
        column_header_edge_color = TableEdgeColor(
            left=prefs.color_neutral, right=prefs.color_neutral
        )
        table_columns: dict[str, TableColumn] = {
            columns[0]: TableColumn(
                header_style=TableColumnStyle(
                    fontweight="bold",
                    ha="center",
                    va="center",
                    edge_color=column_header_edge_color,
                    face_color="black",
                    text_color="white",
                ),
                detail_style=TableColumnStyle(
                    ha="left",
                    va="center",
                    edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                    text_color=prefs.color_neutral,
                ),
                even_row_style=TableColumnStyle(
                    ha="left",
                    va="center",
                    edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                    face_color=prefs.color_light_gray,
                    text_color=prefs.color_neutral,
                ),
                lpad=10.0,
                rpad=10.0,
                max_proportional_width=0.5,
            ),
            columns[1]: TableColumn(
                header_style=TableColumnStyle(
                    fontweight="bold",
                    ha="center",
                    va="center",
                    edge_color=column_header_edge_color,
                    face_color="black",
                    text_color="white",
                ),
                detail_style=TableColumnStyle(
                    ha="left",
                    va="center",
                    edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                    text_color=prefs.color_neutral,
                ),
                even_row_style=TableColumnStyle(
                    ha="left",
                    va="center",
                    edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                    face_color=prefs.color_light_gray,
                    text_color=prefs.color_neutral,
                ),
                lpad=20.0,
                rpad=10.0,
                max_proportional_width=0.5,
            ),
        }

        hp_count = len(df)
        if hp_count <= 22:
            fontsize, padding = 9, 12.0
        elif hp_count <= 30:
            fontsize, padding = 8, 6.0
        else:
            fontsize, padding = 6, 5.0

        table = Table(
            data=df,
            max_table_height=1.0,
            mid_x=0.5,
            top_y=1.0,
            fontsize=fontsize,
            columns=table_columns,
            cell_edge_linewidth=TableEdgeLinewidth.all_edges(linewidth=0.4),
            table_edge_linewidth=TableEdgeLinewidth(),
            table_edge_padding=(
                (0.01, 0.01, 0.0, 0.0) if hp_count > 30 else (0.0, 0.0, 0.0, 0.0)
            ),
            header_tpad=padding,
            header_bpad=padding,
            detail_tpad=padding,
            detail_bpad=padding,
        )
        table_layout = render_table(
            pdf_page=pdf_page,
            table=table,
            ax=ax_params,
            dry_run=True,
            renderer=renderer,
        )

        # Force the layout to calculate based on ax_params and its real title
        # Get the actual physical position of ax_params after the title is placed.
        ss = ax_params.get_subplotspec()

        if ss is None:
            raise RuntimeError(
                "ax_params must be a subplot to calculate position from gridspec"
            )

        target_pos = ss.get_position(fig)
        total_w = target_pos.width

        num_pages = len(table_layout.pages)
        if num_pages > 1:
            # Create axes at exact relative coordinates to match the ax_params height.
            w_per_col = total_w / num_pages
            spacing = 0.005

            for page_index in range(num_pages):
                # Calculate the exact rect for each column axis, relative to the figure
                col_left = (
                    target_pos.x0
                    + (page_index * w_per_col)
                    + (spacing if page_index > 0 else 0)
                )
                col_width = w_per_col - spacing
                ax = fig.add_axes(
                    (col_left, target_pos.y0, col_width, target_pos.height)
                )
                ax.axis("off")
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                render_table_from_page_layout(
                    pdf_page=pdf_page,
                    table_layout=table_layout,
                    page_index=page_index,
                    using_axis=ax,
                    adjust_mid_x=False,
                )
        else:
            ax = fig.add_axes(
                (target_pos.x0, target_pos.y0, target_pos.width, target_pos.height)
            )
            ax.axis("off")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            render_table_from_page_layout(
                pdf_page=pdf_page, table_layout=table_layout, using_axis=ax
            )

    def _plot_feature_importance(
        self,
        ax: Any,
        importance_df: pd.DataFrame,
        model_color: str,
        test_metrics_text: Optional[str],
    ) -> None:
        """Plot feature importance bars or a fallback message."""
        if not importance_df.empty:
            # Dynamic Scaling: Adjust bar thickness based on number of features
            active_features = importance_df[importance_df["importance"] != 0.0]
            active_features_len = len(active_features)
            importances = active_features["importance"][: self.summary.top_n_importance]
            max_importance = max(importances)
            features = active_features["id"]

            # Put most important at the top
            y_pos = np.arange(active_features_len)[::-1]
            bar_height = 0.6 if active_features_len > 10 else 0.4

            ax.barh(y_pos, importances, height=bar_height, color=model_color, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_title(f"Top {len(active_features)} Predictors")
            ax.set_xlabel("Relative Importance / Weight", fontsize=9)
            ax.set_xlim(0, max_importance * 1.15)
            ax.grid(axis="x", linestyle="--", alpha=0.6)

            # Remove top/right spines for a cleaner look
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            # Add value labels at the end of bars
            for i, v in enumerate(importances):
                ax.text(
                    v + (max(importances) * 0.01),
                    y_pos[i],
                    prefs.score_format.format_value(v),
                    va="center",
                    fontsize=8,
                    color=prefs.color_neutral,
                )

            if test_metrics_text is not None:
                ax.text(
                    0.65,
                    0.05,  # Positioning near bottom-right of the subplot
                    test_metrics_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
        else:
            ax.text(
                0.5,
                0.3,
                "Feature Importance not supported for this model type.",
                ha="center",
            )
            ax.axis("off")

    def _plot_regression_residuals(
        self, ax, y_true: pd.Series, y_preds: pd.Series, model_name
    ):
        """Visualizes the error distribution for the model."""
        model_color = self.summary.solid_color_palette[model_name]
        residuals = y_true - y_preds

        def get_optimized_limits(percentile=0.99):
            # Use the absolute maximum of the Nth percentile to keep the plot symmetric
            # This ensures the 0-line stays exactly in the middle of the Y-axis
            limit_val = np.percentile(np.abs(residuals), percentile * 100)

            # Add a 10% "breathing room" buffer
            buffer = limit_val * 0.1
            y_limit = limit_val + buffer

            # For the X-axis (Predicted Values), we do the same
            x_min = np.percentile(y_preds, (1 - percentile) * 100)
            x_max = np.percentile(y_preds, percentile * 100)
            x_buffer = (x_max - x_min) * 0.05

            return (-y_limit, y_limit), (x_min - x_buffer, x_max + x_buffer)

        # Scatter plot of predictions vs residuals
        ax.scatter(
            y_preds, residuals, alpha=0.1, color=model_color, s=1, rasterized=True
        )
        # ax.axhline(0, color=prefs.color_neutral, linestyle="--", lw=1)
        ax.axhline(0, color="#bdc3c7", linestyle="--", lw=1.5, zorder=3)
        y_lim, x_lim = get_optimized_limits()
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)

        ax.set_title(f"Residual Analysis")
        ax.set_xlabel("Predicted Values", fontsize=9)
        ax.set_ylabel("Residuals (Error)", fontsize=9)

        # Clean up spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    def _plot_classification_diagnostics(
        self, ax, y_preds: pd.Series, y_true: pd.Series
    ):
        """Plot a confusion matrix for classification diagnostics."""
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)

    def _plot_residual_analysis(self, config: ModelConfiguration, ax_residuals):
        """Dispatch residual analysis plot based on task type."""
        if config.preds_val is not None:
            if config.task_type == TaskType.REGRESSION:
                self._plot_regression_residuals(
                    ax=ax_residuals,
                    y_true=self.summary.data_splits.val_target,
                    y_preds=config.preds_val,
                    model_name=config.model_type.value,
                )
            else:

                def get_plot_ready_data(
                    y_true, y_preds, sample_size=prefs.default_plot_sample_size
                ):
                    """Returns a representative subset of data for visualization."""
                    if len(y_true) <= sample_size:
                        return y_true, y_preds

                    # Combine into a temporary DF for easy sampling
                    df = pd.DataFrame({"true": y_true, "pred": y_preds})

                    # Stratified sample to keep class proportions identical
                    # This ensures rare classes still show up in the heatmap
                    df_sample = df.groupby("true", group_keys=False).apply(
                        lambda x: x.sample(
                            min(len(x), sample_size // df["true"].nunique())
                        )
                    )

                    return df_sample["true"], df_sample["pred"]

                y_true, y_preds = get_plot_ready_data(
                    y_true=self.summary.data_splits.val_target,
                    y_preds=config.preds_val,
                )
                self._plot_classification_diagnostics(
                    ax=ax_residuals,
                    y_true=y_true,
                    y_preds=y_preds,
                )

    def _plot_worst_residual_errors(
        self,
        pdf_page: PDFDocument.Page,
        ax: Axes,
        y_true: pd.Series,
        y_preds: pd.Series,
        n: int,
        renderer: RendererBase,
    ):
        """Render a table of worst regression errors."""
        ax.set_title("Worst Regression Misses (Outliers)")
        ax.axis("off")

        # Calculate Absolute Error and Percent Error
        abs_error = np.abs(y_true - y_preds)
        # Avoid division by zero
        pct_error = abs_error / np.where(y_true == 0, 1, y_true)

        ACTUAL_VALUE_COL = "Actual"
        PREDICTED_VALUE_COL = "Predicted"
        ABS_ERROR_COL = "Abs Error"
        ERROR_PCT_COL = "Error Pct"

        # Build dataframe
        df = (
            pd.DataFrame(
                {
                    ACTUAL_VALUE_COL: y_true,
                    PREDICTED_VALUE_COL: y_preds,
                    ABS_ERROR_COL: abs_error,
                    ERROR_PCT_COL: pct_error,
                }
            )
            .sort_values(by=ABS_ERROR_COL, ascending=False)
            .head(n)
        )

        summary = self.summary
        column_formats: dict[str, FormatConfig] = {
            ACTUAL_VALUE_COL: summary.actual_value_fmt,
            PREDICTED_VALUE_COL: summary.predicted_value_fmt,
            ABS_ERROR_COL: summary.abs_error_fmt,
            ERROR_PCT_COL: summary.error_pct_fmt,
        }

        # Convert dataframe values to formatted strings
        for col, fmt in column_formats.items():
            df[col] = [fmt.format_value(val=v) for v in df[col]]

        table_columns: dict[str, TableColumn] = {}
        for col in df.columns:
            text_alignment = column_formats[col].matplot_alignment()
            table_columns[col] = TableColumn(
                header_style=TableColumnStyle(
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    edge_color=TableEdgeColor(
                        left=prefs.color_neutral, right=prefs.color_neutral
                    ),
                    face_color="black",
                    text_color="white",
                ),
                detail_style=TableColumnStyle(
                    ha=text_alignment,
                    va="center",
                    fontfamily="monospace",
                    edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                    face_color=prefs.color_paper,
                    text_color=prefs.color_neutral,
                ),
                even_row_style=TableColumnStyle(
                    ha=text_alignment,
                    va="center",
                    fontfamily="monospace",
                    edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                    face_color=prefs.color_light_gray,
                    text_color=prefs.color_neutral,
                ),
                has_consistent_width=True,
                has_consistent_height=True,
                lpad=10.0,
                rpad=15.0,
            )

        table = Table(
            data=df,
            max_table_height=1.0,
            mid_x=0.5,
            top_y=1.0,
            fontsize=10,
            columns=table_columns,
            cell_edge_linewidth=TableEdgeLinewidth.all_edges(linewidth=0.4),
            table_edge_linewidth=TableEdgeLinewidth(),
            table_edge_padding=(0.0, 0.0, 0.0, 0.0),
            use_full_axis_width=True,
            header_tpad=20.0,
            header_bpad=0.0,
            detail_tpad=12.0,
            detail_bpad=12.0,
        )

        render_table(
            pdf_page=pdf_page,
            table=table,
            ax=ax,
            renderer=renderer,
        )

    def _plot_worst_classification_errors(
        self,
        pdf_page: PDFDocument.Page,
        ax: Axes,
        y_true: pd.Series,
        y_probs: pd.DataFrame,
        y_preds: pd.Series,
        n: int,
        renderer: RendererBase,
    ):
        """Render a table of worst classification errors."""
        ax.set_title(f"Top {n} Model Misses (Outliers)")
        ax.axis("off")

        # Find cases where the model was wrong
        incorrect = y_true != y_preds

        # Get the probability assigned to the 'wrong' predicted class
        # (The higher this is, the 'worse' the mistake)
        confidences = y_probs.max(axis=1)

        ACTUAL_VALUE_COL = "Actual"
        PREDICTED_VALUE_COL = "Predicted"
        CONFIDENCE_COL = "Confidence"

        df = (
            pd.DataFrame(
                {
                    ACTUAL_VALUE_COL: y_true,
                    PREDICTED_VALUE_COL: y_preds,
                    CONFIDENCE_COL: confidences,
                }
            )[incorrect]
            .sort_values(CONFIDENCE_COL, ascending=False)
            .head(n)
        )

        summary = self.summary
        column_formats: dict[str, Any] = {
            ACTUAL_VALUE_COL: summary.actual_value_fmt,
            PREDICTED_VALUE_COL: summary.predicted_value_fmt,
            CONFIDENCE_COL: summary.abs_error_fmt,
        }

        # Convert dataframe values to formatted strings
        for col, fmt in column_formats.items():
            df[col] = [fmt.format_value(val=v) for v in df[col]]

        table_columns: dict[str, TableColumn] = {}
        for col in df.columns:
            text_alignment = column_formats[col].matplot_alignment()
            table_columns[col] = TableColumn(
                header_style=TableColumnStyle(
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    edge_color=TableEdgeColor(
                        left=prefs.color_light_gray, right=prefs.color_light_gray
                    ),
                    face_color="black",
                    text_color="white",
                ),
                detail_style=TableColumnStyle(
                    ha=text_alignment,
                    va="center",
                    fontfamily="monospace",
                    edge_color=TableEdgeColor.closed(color=prefs.color_light_gray),
                    face_color=prefs.color_paper,
                    text_color=prefs.color_neutral,
                ),
                even_row_style=TableColumnStyle(
                    ha=text_alignment,
                    va="center",
                    fontfamily="monospace",
                    edge_color=TableEdgeColor.closed(color=prefs.color_light_gray),
                    face_color=prefs.color_light_gray,
                    text_color=prefs.color_neutral,
                ),
            )

        table = Table(
            data=df,
            max_table_height=1.0,
            mid_x=0.5,
            top_y=1.0,
            fontsize=9,
            columns=table_columns,
            cell_edge_linewidth=TableEdgeLinewidth.all_edges(linewidth=0.4),
            table_edge_linewidth=TableEdgeLinewidth.all_edges(linewidth=0.4),
            table_edge_padding=(5.0, 5.0, 0.0, 0.0),
            table_edge_color=TableEdgeColor.closed(color=prefs.color_title),
        )

        render_table(
            pdf_page=pdf_page,
            table=table,
            ax=ax,
            renderer=renderer,
        )

    def _render_error_analysis(
        self,
        pdf_page: PDFDocument.Page,
        ax_worst_errors: Axes,
        config: ModelConfiguration,
        renderer: RendererBase,
    ):
        """Render the error analysis table for a model configuration."""
        worst_errors_n = prefs.default_worst_errors_n

        if config.preds_val is not None:
            if config.task_type == TaskType.REGRESSION:
                self._plot_worst_residual_errors(
                    pdf_page=pdf_page,
                    ax=ax_worst_errors,
                    y_true=self.summary.data_splits.val_target,
                    y_preds=config.preds_val,
                    n=worst_errors_n,
                    renderer=renderer,
                )
            else:
                if config.probs_val is not None:
                    self._plot_worst_classification_errors(
                        pdf_page=pdf_page,
                        ax=ax_worst_errors,
                        y_true=self.summary.data_splits.val_target,
                        y_probs=config.probs_val,
                        y_preds=config.preds_val,
                        n=worst_errors_n,
                        renderer=renderer,
                    )

    def _render_model_deep_dive(self, config: ModelConfiguration) -> None:
        """Render the per-model deep dive page."""
        model_type_format = EnumFormat()
        id_format = IntegerFormat()
        page_name = f"{model_type_format.format_value(config.model_type)} [{id_format.format_value(config.id)}]"
        pdf_page = self.pdf_doc.create_new_page(
            page_name=page_name,
            print_page_name=False,
        )
        fig = pdf_page.fig
        fig.draw_without_rendering()
        canvas: Any = fig.canvas
        renderer: RendererBase = canvas.get_renderer()
        pdf_page.layout_engine.set(w_pad=0.0, h_pad=0.0, hspace=0.1, wspace=0.1)

        # Use a wide layout to fit two columns
        gs = fig.add_gridspec(3, 2, height_ratios=[0.1, 0.45, 0.45])
        gs_params = gs[1, 0]
        gs_features = gs[1, 1]
        gs_residuals = gs[2, 0]
        gs_worst_errors = gs[2, 1]
        ax_params = fig.add_subplot(gs_params)
        ax_features = fig.add_subplot(gs_features)
        ax_residuals = fig.add_subplot(gs_residuals)
        ax_worst_errors = fig.add_subplot(gs_worst_errors)

        fig.text(
            0.5,
            0.88,
            page_name,
            fontsize=14,
            weight="bold",
            color=prefs.color_title,
            ha="center",
            va="center",
            transform=fig.transFigure,
            bbox=dict(
                facecolor="white",
                edgecolor=prefs.color_title,
                boxstyle="round,pad=0.3",
                lw=1.0,
            ),
        )

        if config.has_test_set_evaluation_scores:
            if config.task_type is TaskType.REGRESSION:
                test_metrics_text = (
                    f"Test Set Performance\n"
                    f"R2:  {prefs.score_format.format_value(config.r2_test)}\n"
                    f"MAE: {prefs.score_format.format_value(config.mae_test)}"
                )
            else:
                test_metrics_text = (
                    f"Test Set Performance\n"
                    f"Accuracy: {prefs.score_format.format_value(config.accuracy_test)}"
                )
        else:
            test_metrics_text = None

        # --- Top Right: Feature Importance ---
        self._plot_feature_importance(
            ax=ax_features,
            importance_df=self.importance_dict[config.id],
            model_color=self.summary.solid_color_palette[config.model_type.value],
            test_metrics_text=test_metrics_text,
        )

        # --- Bottom Left: Residual Analysis ---
        self._plot_residual_analysis(config, ax_residuals)

        # --- Bottom Right: Worst Errors ---
        self._render_error_analysis(
            pdf_page=pdf_page,
            ax_worst_errors=ax_worst_errors,
            config=config,
            renderer=renderer,
        )

        # --- Top Left: Identity & Hyperparameters ---
        # The Hyperparameters table must be rendered after the Residual Analysis
        # table, so that the bounds of ax_params will be adjusted based on the
        # bounds of ax_residuals. ax_residuals has y-axis labels that shift the
        # x position of ax_params. The subplot created for the table is not affected
        # by the shift. To align the table horizontally inside ax_params, the
        # bounds for ax_params need to be shifted before the table is drawn.
        self._plot_model_hyperparameters(
            pdf_page=pdf_page, config=config, ax_params=ax_params, renderer=renderer
        )

    def _render_detailed_audit_stats(self) -> None:
        """Render the detailed audit stats table page."""
        pdf_page = self.pdf_doc.create_new_page(
            page_name="Detailed Audit Stats", print_page_name=False
        )
        fig = pdf_page.fig

        # Header
        header_top_y = 0.88
        fig.text(
            0.5,
            header_top_y,
            f"Detailed Audit Stats - {self.summary.audit_timestamp}",
            fontsize=16,
            weight="bold",
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

        column_header_edge_color = TableEdgeColor(
            left=prefs.color_neutral, right=prefs.color_neutral
        )
        column_headers: list[str] = [
            "Model",
            "Abbr",
            "CV Score (Tuning)",
            "Val Score",
            "Test Score",
            "Train Time (s)",
            "Actual Peak RAM",
        ]
        header_style = TableColumnStyle(
            fontweight="bold",
            ha="center",
            va="center",
            edge_color=column_header_edge_color,
            face_color="black",
            text_color="white",
        )
        detail_style_left = TableColumnStyle(
            ha="left",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            text_color=prefs.color_neutral,
        )
        even_row_style_left = TableColumnStyle(
            ha="left",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            face_color=prefs.color_light_gray,
            text_color=prefs.color_neutral,
        )
        detail_style_right = TableColumnStyle(
            ha="right",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            text_color=prefs.color_neutral,
        )
        even_row_style_right = TableColumnStyle(
            ha="right",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            face_color=prefs.color_light_gray,
            text_color=prefs.color_neutral,
        )
        detail_style_center = TableColumnStyle(
            ha="center",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            text_color=prefs.color_neutral,
        )
        even_row_style_center = TableColumnStyle(
            ha="center",
            va="center",
            edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
            face_color=prefs.color_light_gray,
            text_color=prefs.color_neutral,
        )

        table_columns: dict[str, TableColumn] = {}
        for col in column_headers:
            if col == "Model":
                detail_style = detail_style_left
                even_row_style = even_row_style_left
            elif col == "Abbr":
                detail_style = detail_style_center
                even_row_style = even_row_style_center
            else:
                detail_style = detail_style_right
                even_row_style = even_row_style_right

            table_columns[col] = TableColumn(
                header_style=header_style,
                detail_style=detail_style,
                even_row_style=even_row_style,
                lpad=12.0,
                rpad=12.0,
            )

        table_df = self.summary_df[column_headers].copy()
        string_format = StringFormat()
        string_columns = ["Model", "Abbr"]
        numeric_columns = [
            "CV Score (Tuning)",
            "Val Score",
            "Test Score",
        ]

        for col in string_columns:
            table_df[col] = table_df[col].apply(string_format.format_value)

        for col in numeric_columns:
            table_df[col] = table_df[col].apply(prefs.score_format.format_value)

        table_df["Actual Peak RAM"] = table_df["Actual Peak RAM"].apply(
            prefs.gb_format.format_value
        )

        table_df["Train Time (s)"] = table_df["Train Time (s)"].apply(
            prefs._train_time_format.format_value
        )
        pc = self.pdf_doc.page_configuration
        table_top_y = header_top_y - 0.05
        table_max_height = table_top_y - pc.bottom_margin
        table = Table(
            data=table_df,
            max_table_height=table_max_height,
            mid_x=0.5,
            top_y=table_top_y,
            fontsize=9,
            columns=table_columns,
            cell_edge_linewidth=TableEdgeLinewidth.all_edges(0.4),
            table_edge_linewidth=TableEdgeLinewidth.all_edges(0.0),
            table_edge_color=TableEdgeColor.closed(color=prefs.color_title),
            header_tpad=8.0,
            header_bpad=8.0,
            detail_tpad=6.0,
            detail_bpad=6.0,
        )

        _ = render_table(pdf_page=pdf_page, table=table)

    def _render_recommendation_page(self) -> None:
        """Render the strategic recommendation page."""
        pdf_page = self.pdf_doc.create_new_page(
            page_name="Recommendation", print_page_name=False, include_footer=False
        )
        fig = pdf_page.fig
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")

        header_top_y = 0.85
        header_artist = ax.text(
            0.5,
            header_top_y,
            "STRATEGIC RECOMMENDATION",
            fontsize=14,
            ha="center",
            linespacing=1.8,
            color=prefs.color_title,  # Deep Charcoal for the text
            weight="bold",
        )
        renderer = fig.canvas.get_renderer()  # type: ignore

        if renderer is None:
            fig.draw_without_rendering()
            renderer = fig.canvas.get_renderer()  # type: ignore

        header_bbox = get_artist_bbox(
            obj=header_artist, transform_to=fig, renderer=renderer
        )
        header_bot_y = header_top_y - header_bbox.height
        table_layout = self._render_metric_table(
            pdf_page=pdf_page,
            top_y=header_bot_y - 0.05,
            df=self.strategic_recommendation_metrics,
        )
        table_rect = table_layout.pages[0].rect
        recommendation_text: List[str] = []
        buffer_width = int(table_rect.get_width() * 160)
        model = self.best_model.model
        score_cv = model.score_cv if model.score_cv is not None else 0.0

        if model.val_score - score_cv > 0.05:
            recommendation_text.append(
                format_text(
                    text="Significant score improvement on full data vs sample.",
                    buffer_width=buffer_width,
                    prefix="•",
                    suffix="",
                    insert_leading_space=True,
                    include_prefix_on_wrapped_lines=False,
                )
            )

        if model.sampling_factor < 0.3:
            recommendation_text.append(
                format_text(
                    text="This model required aggressive sampling to stay within hardware memory limits during the tuning phase.",
                    buffer_width=buffer_width,
                    prefix="•",
                    suffix="",
                    insert_leading_space=True,
                    include_prefix_on_wrapped_lines=False,
                )
            )

        if len(recommendation_text) > 0:
            y_pos = table_rect.get_y() - 0.1
            ax.text(
                0.5,
                y_pos,
                "NOTES",
                fontsize=12,
                ha="center",
                va="center",
                linespacing=1.8,
                color=prefs.color_title,
                weight="bold",
            )

            x_pos = table_rect.get_x()
            y_pos -= 0.025
            full_notes = "\n".join(recommendation_text)

            ax.text(
                x_pos,
                y_pos,
                full_notes,
                fontsize=11,
                linespacing=1.8,
                va="top",
                ha="left",
                wrap=False,
                transform=fig.transFigure,
                color=prefs.color_title,
            )

        # Add a Footer note about the Audit
        ax.text(
            0.5,
            0.15,  # Positioned near the bottom of the page
            "This recommendation is based on a balance of validation accuracy, \n"
            "memory efficiency, and training time performance.",
            fontsize=10,
            style="italic",
            transform=fig.transFigure,  # Centered on the page, not the axis
            ha="center",
            va="top",  # Aligns the top of the text to the coordinate
            linespacing=1.6,  # Matches the breatheability of the rest of the report
            color=prefs.color_neutral,
            alpha=0.85,  # Slight transparency to keep it as a 'secondary' note
        )
