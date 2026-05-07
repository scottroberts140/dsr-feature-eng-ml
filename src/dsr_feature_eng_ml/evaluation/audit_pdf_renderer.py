"""PDF rendering utilities for model audit reports."""

from __future__ import annotations

import os
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dsr_files.pdf_handler import (
    PageColors,
    PageConfiguration,
    PageOrientation,
    PageSize,
    PDFDocument,
)
from dsr_utils.formatting import (
    BoolFormat,
    BoolRepresentation,
    DateTimeFormat,
    EnumFormat,
    FloatFormat,
    FormatConfig,
    IntegerFormat,
    NumericScale,
    PercentageFormat,
    StringFormat,
    TextAlignment,
    ValueDescFormat,
    format_text,
)
from dsr_utils.matplotlib import get_artist_bbox
from dsr_utils.strings import apply_tracking
from dsr_utils.tables import (
    Table,
    TableColumn,
    TableColumnStyle,
    TableEdgeColor,
    TableEdgeLinewidth,
    TableLayout,
    render_table,
    render_table_from_page_layout,
)
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from dsr_feature_eng_ml.enums import (
    ModelEnumSortOrder,
    ModelTypeData,
    ModelTypeDataRecType,
    TaskType,
)
from dsr_feature_eng_ml.prefs_instance import prefs

from .schema import DataSplits, ModelConfiguration

if TYPE_CHECKING:
    from .model_audit_summary import ModelAuditSummary


class AuditPDFRenderer:
    """
    Render a multi-page PDF report for a `ModelAuditSummary`.

    Coordinates the generation of executive summaries, competition results,
    generalization charts, and detailed model-specific pages.
    """

    formatted_processed_row_count: str
    summary_df: pd.DataFrame
    importance_dict: dict[str, Any]
    _best_model_metrics: pd.DataFrame

    class Model:
        """Convenience wrapper for computed model indicators used by the report."""

        def __init__(self, model: ModelConfiguration, data_splits: DataSplits):
            """Precompute key model indicators used by the report."""
            self.model = model

            # Integrity assessment is only meaningful when a cleaned validation
            # score exists to compare against the raw validation score.
            self.has_integrity_score = model.score_val_cleaned is not None
            if self.has_integrity_score:
                self.model_quality = prefs.get_model_quality(model.quality_score)
                quality_fmt = ValueDescFormat(
                    precision=2,
                    description=self.model_quality.text,
                    description_leading_space=True,
                    description_decorator="()",
                )
                self.quality_score_text = (
                    f"Integrity Score: {quality_fmt.format_value(model.quality_score)}"
                )
                self.integrity_summary_value = ValueDescFormat(
                    precision=2,
                    description="/100 Score",
                ).format_value(model.quality_score)
            else:
                self.model_quality = prefs.ModelQuality(
                    score_min=0.0,
                    text="N/A",
                    text_weight="normal",
                    color=prefs.color_neutral,
                )
                self.quality_score_text = "Integrity Score: N/A"
                self.integrity_summary_value = "N/A"

            # Boolean performance checks against global prefs
            self.is_efficient = (
                model.efficiency(data_splits=data_splits)
                >= prefs.model_efficiency_threshold
            )
            self.is_accurate = (model.val_score or 0.0) > prefs.model_accuracy_limit
            self.is_acceptable = (model.val_score or 0.0) > prefs.model_acceptable_limit
            self.is_stable = (model.val_score or 0.0) > prefs.model_stability_limit

            # Generate the final recommendation verdict
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
        """
        Initialize the PDF renderer with standard hardware and performance metrics.

        Parameters
        ----------
        summary : ModelAuditSummary
            Completed audit summary containing all model results and data splits.
        report_title : str, optional
            Title string printed on the cover page. Defaults to
            ``"Model Audit Report"``.
        """
        self.summary = summary
        self.report_title = report_title

        if not summary.results:
            raise IndexError("Audit Summary does not contain any results.")

        self.results = summary.results
        _best = summary.best_overall_model

        if _best is None:
            raise ValueError("Best model could not be determined.")

        # 1. Performance and Hardware Context
        self.best_model = AuditPDFRenderer.Model(
            model=_best, data_splits=summary.data_splits
        )
        self.total_cpu_time = sum(res.total_duration for res in self.results)
        self.max_ram_observed = max(res.actual_peak_gb for res in self.results)

        # Format the Hardware Context line seen on every page
        cores_fmt = ValueDescFormat(
            precision=0,
            description="Cores Detected",
            description_leading_space=True,
            description_decorator="",
        )
        ram_fmt = ValueDescFormat(
            precision=2,
            description="GB",
            description_leading_space=True,
            description_decorator="",
        )
        vol_fmt = ValueDescFormat(
            precision=0, description="rows", description_leading_space=True
        )

        self.hardware_stats_text = (
            f"Audit Hardware Context: {cores_fmt.format_value(os.cpu_count())} | "
            f"Peak Memory Demand: {ram_fmt.format_value(self.max_ram_observed)} | "
            f"Data Volume: {vol_fmt.format_value(summary.processed_row_count)} | "
            "Status: Hardware Safety Limits Respected"
        )

        # 2. Results Data Extraction
        # summary_df contains rank data; importance_dict maps feature rankings
        self.summary_df, self.importance_dict = self._get_audit_data()
        self.summary_df.sort_values(by="Val Score", ascending=False, inplace=True)

        # 3. Winning Model Metric Matrix
        model = self.best_model.model
        row_fmt = IntegerFormat(precision=2, numeric_scale=NumericScale.AUTO)
        self.formatted_processed_row_count = row_fmt.format_value(
            self.summary.processed_row_count
        )
        eff_fmt = ValueDescFormat(
            precision=1,
            numeric_scale=NumericScale.K,
            description="rows/sec",
            description_leading_space=True,
        )
        pct_fmt = PercentageFormat(precision=1)
        enum_fmt = EnumFormat()
        dur_fmt = DateTimeFormat(use_duration_format=True, alignment=TextAlignment.LEFT)

        # Build the shared metric table for Executive Summary and Strategic Recommendation
        self._best_model_metrics = pd.DataFrame(
            [
                ["Winning Model:", f"{enum_fmt.format_value(model.model_type)}"],
                [
                    f"Validation Score ({enum_fmt.format_value(model.scoring)}):",
                    f"{prefs.score_format.format_value(model.val_score)}",
                ],
                [
                    f"Test Score ({enum_fmt.format_value(model.scoring)}):",
                    f"{prefs.score_format.format_value(model.test_score) if model.has_test_set_evaluation_scores else '-'}",
                ],
                [
                    "Throughput:",
                    f"{eff_fmt.format_value(model.efficiency(self.summary.data_splits))}",
                ],
                [
                    "Integrity:",
                    self.best_model.integrity_summary_value,
                ],
                [
                    "Audit Scale:",
                    f"{row_fmt.format_value(self.summary.processed_row_count)} processed",
                ],
                ["Audit Duration:", f"{dur_fmt.format_value(self.total_cpu_time)}"],
                [
                    "Resources:",
                    f"{os.cpu_count()} Cores | Peak Mem: {ram_fmt.format_value(self.max_ram_observed)}",
                ],
                [
                    "Methodology:",
                    f"{pct_fmt.format_value(model.sampling_factor)} used for training",
                ],
            ],
            columns=["Metric", "Value"],
        )

        # Slice for the front-page Executive Summary
        self.executive_summary_metrics = self._best_model_metrics[:5]
        # Full view for the Strategic Recommendation page
        self.strategic_recommendation_metrics = self._best_model_metrics

        # 4. PDF Document Engine initialization
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
        """
        Orchestrate the creation of the multi-page PDF document.

        This method follows the sequential "Story" structure seen in the audit:
        Title -> Executive Summary -> Anomaly Logs -> Methodology -> Competition
        -> Detailed Models -> Final Verdict.
        """
        # Sequential page generation
        self._render_title_page()
        self._render_executive_summary()
        self._render_anomaly_page()
        self._render_model_legend()
        self._render_audit_results()
        self._render_cv_vs_final()
        self._render_features_page()

        # Iterative deep-dives for every audited model
        for config in self.results:
            self._render_model_deep_dive(config=config)

        self._render_detailed_audit_stats()  # Summary Table
        self._render_recommendation_page()  # Final Verdict

        # Build Table of Contents based on generated flowables
        self.pdf_doc.render_table_of_contents()
        return self.pdf_doc

    def _draw_page_header(
        self,
        pdf_page: PDFDocument.Page,
        page_name: str,
        print_page_name: bool = True,
    ) -> None:
        """
        Render the consistent upper-third branding for every report page.
        """
        fig = pdf_page.fig
        pc = self.pdf_doc._page_configuration

        # 1. Primary Report Title (Centered, Bold)
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

        # 2. Dataset Identifier (Left Aligned, Italic)
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

        # 3. Dynamic Page Name (Sub-header)
        if print_page_name:
            fig.text(
                0.5,
                0.93,
                page_name.upper(),
                fontsize=9,
                color=prefs.color_title,
                ha="center",
            )

        # 4. Horizontal Separator
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
        """
        Render hardware telemetry and data quality indicators.
        """
        fig = pdf_page.fig

        # 1. Hardware Context (Left)
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

        # 2. Semantic Data Quality Tag (Right)
        # Uses color and weight from the computed ModelQuality object
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
        """
        Generate the high-impact cover page for the audit report.
        """
        if not self.results:
            return

        # Create a new page without standard headers/footers for a clean cover look
        pdf_page = self.pdf_doc.create_new_page(
            page_name="Title Page",
            include_header=False,
            include_footer=False,
            include_in_page_numbering=False,
            print_page_name=False,
            include_in_index=False,
        )
        fig = pdf_page.fig

        # 1. Main Header Branding
        fig.text(
            0.5,
            0.65,
            apply_tracking("PERFORMANCE AUDIT REPORT"),
            fontsize=14,
            color=prefs.color_title,
            alpha=0.7,
            ha="center",
        )

        # 2. Report Title
        fig.text(
            0.5,
            0.60,
            self.report_title.upper(),
            fontsize=28,
            weight="black",
            color=prefs.color_title,
            ha="center",
        )

        # 3. Audit Specifics
        fig.text(
            0.5,
            0.54,
            f"Subject Dataset: {self.summary.dataset_name}",
            fontsize=12,
            ha="center",
        )

        # 4. Success Verdict
        rec_color = self.best_model.recommendation.color
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

        # 5. Data Quality Score Badge
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

        # 6. Volume and Timestamp Metadata
        meta_text = (
            f"Processed {self.formatted_processed_row_count} rows | "
            "Hardware Safety Limits: Respected"
        )
        fig.text(
            0.5,
            0.10,
            meta_text,
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
                        feat_meta = self.summary.resolve_feature(feat)
                        feat_fmt = feat_meta.formatter if feat_meta else StringFormat()
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
        """
        Render the Executive Summary page with risk profile and key metrics.
        """
        if len(self.results) == 0:
            return
        pdf_page = self.pdf_doc.create_new_page(page_name="Executive Summary")
        fig = pdf_page.fig
        rec = self.best_model.recommendation
        headline_top_y = 0.85
        risk_assessment_header_top_y = 0.75

        # 1. Recommendation Header
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

        # 2. Audit Risk Profile Section
        # This section highlights skew, kurtosis, and risk concentration
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
        """
        Render the exhaustive feature list page with technical metadata.

        This method generates a multi-column table detailing every feature, its
        origin, and its participation in the model training process.
        """
        header_top_y = 0.88
        # create_new_page ensures the header/footer branding is applied
        pdf_page = self.pdf_doc.create_new_page(page_name="Feature List")

        # Configuration for tables that might span multiple pages
        pdf_page.continuation_text = "(cont.)"
        pdf_page.continuation_page_top_y = header_top_y
        fig = pdf_page.fig

        # Page Sub-Heading
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

        # 1. Define Column Styles
        column_header_edge = TableEdgeColor(
            left=prefs.color_neutral, right=prefs.color_neutral
        )

        header_style = TableColumnStyle(
            fontweight="bold",
            ha="center",
            va="center",
            edge_color=column_header_edge,
            face_color="black",
            text_color="white",
        )

        # Shared edge and color configuration for data rows
        closed_edge = TableEdgeColor.closed(color=prefs.color_neutral)

        # Base styles for left, right, and center alignment
        def get_styles(ha: str, face: str | None = None) -> TableColumnStyle:
            # We use an empty string or "white" if face is None to satisfy the str requirement
            resolved_face = face if face is not None else "none"

            return TableColumnStyle(
                ha=ha,
                va="center",
                edge_color=closed_edge,
                face_color=resolved_face,
                text_color=prefs.color_neutral,
            )

        styles = {
            "left": (get_styles("left"), get_styles("left", prefs.color_light_gray)),
            "right": (get_styles("right"), get_styles("right", prefs.color_light_gray)),
            "center": (
                get_styles("center"),
                get_styles("center", prefs.color_light_gray),
            ),
        }

        # 2. Build Column Configuration
        column_headers = [
            "ID",
            "Sub",
            "Feature",
            "Pos",
            "Short Name",
            "Parent Name",
            "Used in Fit",
            "Description",
        ]
        table_columns: dict[str, TableColumn] = {}

        for col in column_headers:
            # Match alignment to data type
            if col == "Pos":
                align = "right"
            elif col in ("Sub", "Used in Fit"):
                align = "center"
            else:
                align = "left"

            det, even = styles[align]
            table_columns[col] = TableColumn(
                header_style=header_style,
                detail_style=det,
                even_row_style=even,
                lpad=12.0,
                rpad=12.0,
            )

        # 3. Populate Table Data
        table_data: list[list[str]] = []
        pos_fmt = IntegerFormat()
        parent_fmt = StringFormat(fallback="")
        used_fmt = BoolFormat(representation=BoolRepresentation.YES_NO)

        # Sort the OHE-expanded fit set by ID so variants are grouped together
        sorted_features = sorted(self.summary.features_to_fit_set, key=lambda f: f.id)

        for fm in sorted_features:
            # Split composite IDs (e.g. "F07_01") into parent part and child index;
            # non-OHE features (e.g. "F07") have no underscore so Sub is blank.
            parent_id, sep, child_index = fm.id.rpartition("_")
            if sep:
                id_col = parent_id
                sub_col = child_index
            else:
                id_col = fm.id
                sub_col = ""

            table_data.append(
                [
                    id_col,
                    sub_col,
                    fm.name,
                    pos_fmt.format_value(fm.position),
                    fm.short_name,
                    parent_fmt.format_value(fm.parent_name),
                    used_fmt.format_value(fm.is_used_in_fit),
                    fm.description,
                ]
            )

        # 4. Render Table
        table_df = pd.DataFrame(table_data, columns=column_headers)
        pc = self.pdf_doc.page_configuration
        table_top_y = header_top_y - 0.05

        table = Table(
            data=table_df,
            max_table_height=table_top_y - pc.bottom_margin,
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

        render_table(pdf_page=pdf_page, table=table)

    def _render_anomaly_page(self) -> None:
        """
        Render the anomaly summary page identifying high-error contributors.
        """
        # 1. Resolve Data and Dynamic Context
        anomaly_data = (
            self.summary.anomaly_data
            if self.summary.anomaly_data is not None
            else pd.DataFrame()
        )
        dyn_features = (
            self.summary.anomaly_dynamic_features
            if self.summary.anomaly_dynamic_features is not None
            else []
        )

        # Create page using standard V1.2.0 orchestration
        pdf_page = self.pdf_doc.create_new_page(
            page_name="Data Anomaly Log", print_page_name=False
        )
        fig = pdf_page.fig

        # 2. Page Branding and Headers
        model = self.best_model.model
        header_y, sub_y = 0.88, 0.85
        type_fmt = EnumFormat()
        kurt_fmt = FloatFormat(precision=2)

        fig.text(
            0.5,
            header_y,
            f"Data Anomaly Log: {type_fmt.format_value(model.model_type)}",
            fontsize=16,
            weight="bold",
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

        # Subheader reflecting Page 4 formatting
        fig.text(
            0.5,
            sub_y,
            f"Top {self.summary.top_n_anomalies} Primary Contributors to a Dataset Kurtosis of "
            f"{kurt_fmt.format_value(model.val_kurtosis)} (out of {self.formatted_processed_row_count})",
            fontsize=10,
            color="gray",
            ha="center",
            va="center",
            transform=fig.transFigure,
        )

        # 3. Resolve Feature Mappings for the Table
        raw_features = list(dict.fromkeys(dyn_features))
        selected_features = self._select_anomaly_features(raw_features)
        cap_note = self._get_anomaly_cap_note(
            raw_count=len(raw_features), selected_count=len(selected_features)
        )

        if cap_note:
            fig.text(
                0.5,
                sub_y - 0.022,
                cap_note,
                fontsize=8,
                color="gray",
                ha="center",
                va="center",
                transform=fig.transFigure,
            )

        # Maps raw features to display names while preserving chosen order.
        resolved_f = [
            self.summary.anomaly_display_map.get(f, f) for f in selected_features
        ]
        unique_short_names = self._build_anomaly_dynamic_headers(
            raw_features=selected_features,
            resolved_features=resolved_f,
        )

        # Column name and header constants — mirror module-level definitions in
        # model_audit_summary.py; a local import would create a circular dependency.
        from .model_audit_summary import (  # noqa: PLC0415
            AUDIT_ANOMALY_ABS_ERROR_COL,
            AUDIT_ANOMALY_ACTUAL_COL,
            AUDIT_ANOMALY_PREDICTED_COL,
        )

        AUDIT_ANOMALY_ACTUAL_COL_HEADER = "Actual"
        AUDIT_ANOMALY_PREDICTED_COL_HEADER = "Predicted"
        AUDIT_ANOMALY_ABS_ERROR_COL_HEADER = "Abs Error"

        base_cols = [
            AUDIT_ANOMALY_ACTUAL_COL,
            AUDIT_ANOMALY_PREDICTED_COL,
            AUDIT_ANOMALY_ABS_ERROR_COL,
        ]
        base_headers = [
            AUDIT_ANOMALY_ACTUAL_COL_HEADER,
            AUDIT_ANOMALY_PREDICTED_COL_HEADER,
            AUDIT_ANOMALY_ABS_ERROR_COL_HEADER,
        ]

        columns_to_show = base_cols + resolved_f
        column_headers = base_headers + unique_short_names

        # 4. Build Table Column Configurations
        table_columns: dict[str, TableColumn] = {}
        col_formats: dict[str, FormatConfig] = {}
        header_edge = TableEdgeColor(
            left=prefs.color_neutral, right=prefs.color_neutral
        )

        for i, feat in enumerate(columns_to_show):
            # Select appropriate formatter
            if feat == AUDIT_ANOMALY_ACTUAL_COL:
                fmt = self.summary.actual_value_fmt or StringFormat()
            elif feat == AUDIT_ANOMALY_PREDICTED_COL:
                fmt = self.summary.predicted_value_fmt or StringFormat()
            elif feat == AUDIT_ANOMALY_ABS_ERROR_COL:
                fmt = self.summary.abs_error_fmt or FloatFormat()
            else:
                feat_meta = self.summary.resolve_feature(feat)
                fmt = feat_meta.formatter if feat_meta else StringFormat()

            col_formats[feat] = fmt
            align = fmt.matplot_alignment()

            # Apply semantic styles from preferences
            table_columns[column_headers[i]] = TableColumn(
                header_style=TableColumnStyle(
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    edge_color=header_edge,
                    face_color="black",
                    text_color="white",
                ),
                detail_style=TableColumnStyle(
                    ha=align,
                    va="center",
                    fontsize=10,
                    fontfamily="monospace",
                    edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                    face_color=prefs.color_paper,
                    text_color=prefs.color_neutral,
                ),
                even_row_style=TableColumnStyle(
                    ha=align,
                    va="center",
                    fontsize=10,
                    fontfamily="monospace",
                    edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                    face_color=prefs.color_light_gray,
                    text_color=prefs.color_neutral,
                ),
                lpad=20.0,
                rpad=20.0,
                has_consistent_width=True,
                has_consistent_height=True,
            )

        # 5. Process Table Data
        table_data: list[list[str]] = []
        for _, row in anomaly_data.head(self.summary.top_n_anomalies).iterrows():
            formatted_row = [
                col_formats[f].format_value(row[f]) for f in columns_to_show
            ]
            table_data.append(formatted_row)

        # 6. Render the Anomaly Table
        table_top_y = sub_y - (0.07 if cap_note else 0.05)
        table = Table(
            data=pd.DataFrame(table_data, columns=column_headers),
            max_table_height=table_top_y
            - self.pdf_doc.page_configuration.bottom_margin,
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

        table_layout = render_table(pdf_page=pdf_page, table=table, dry_run=True)
        dynamic_headers = column_headers[len(base_headers) :]
        base_header_widths = [table.columns[h].width for h in base_headers]
        avg_dynamic_width = (
            float(np.mean([table.columns[h].width for h in dynamic_headers]))
            if dynamic_headers
            else 1.0
        )
        avg_base_width = (
            float(np.mean(base_header_widths)) if base_header_widths else 1.0
        )
        recommendation_note = self._get_anomaly_column_reduction_note(
            dynamic_col_count=len(dynamic_headers),
            avg_dynamic_width=avg_dynamic_width,
            avg_base_width=avg_base_width,
        )

        render_table_from_page_layout(pdf_page=pdf_page, table_layout=table_layout)

        if recommendation_note and table_layout.pages:
            table_bottom_y = table_layout.pages[0].rect.get_y()
            note_y = max(
                self.pdf_doc.page_configuration.bottom_margin + 0.005,
                table_bottom_y - 0.02,
            )
            fig.text(
                0.5,
                note_y,
                recommendation_note,
                fontsize=7,
                color="gray",
                ha="center",
                va="top",
                transform=fig.transFigure,
            )

    def _select_anomaly_features(self, features: list[str]) -> list[str]:
        """Apply optional anomaly-table cap using feature importance priority."""
        cap = getattr(self.summary, "anomaly_table_max_columns", None)
        if cap is None or cap < 1 or len(features) <= cap:
            return features

        ranked_features = features
        importance_df = self.best_model.model.feature_analysis.feature_importances
        if (
            isinstance(importance_df, pd.DataFrame)
            and not importance_df.empty
            and "feature" in importance_df.columns
        ):
            importance_rank = {
                str(name): idx
                for idx, name in enumerate(importance_df["feature"].tolist())
            }
            matched = [f for f in features if f in importance_rank]
            matched.sort(key=lambda f: importance_rank[f])
            unmatched = [f for f in features if f not in importance_rank]
            ranked_features = matched + unmatched

        return ranked_features[:cap]

    def _get_anomaly_cap_note(self, raw_count: int, selected_count: int) -> str:
        """Return a subtitle note when anomaly feature columns are capped."""
        if not getattr(self.summary, "anomaly_table_show_notes", True):
            return ""
        if selected_count >= raw_count:
            return ""

        return (
            f"Showing {selected_count} of {raw_count} anomaly context columns "
            "by feature importance for readability"
        )

    def _build_anomaly_dynamic_headers(
        self,
        raw_features: list[str],
        resolved_features: list[str],
    ) -> list[str]:
        """Build unambiguous anomaly headers, including OHE suffixes when present."""
        fit_features = {f.name: f for f in self.summary.features_to_fit_set}
        feature_keys = self.summary.features

        headers: list[str] = []
        seen_headers: dict[str, int] = {}

        for raw_feature, resolved_feature in zip(raw_features, resolved_features):
            base_meta = self.summary.resolve_feature(resolved_feature)
            base_header = base_meta.short_name if base_meta else resolved_feature

            ohe_suffix = ""
            if (
                raw_feature not in feature_keys
                and resolved_feature in feature_keys
                and raw_feature.startswith(f"{resolved_feature}_")
            ):
                ohe_suffix = raw_feature[len(resolved_feature) + 1 :]
            else:
                raw_meta = fit_features.get(raw_feature)
                if (
                    raw_meta is not None
                    and raw_meta.parent_name is not None
                    and raw_feature.startswith(f"{raw_meta.parent_name}_")
                ):
                    ohe_suffix = raw_feature[len(raw_meta.parent_name) + 1 :]

            if ohe_suffix:
                candidate = f"{base_header} [{ohe_suffix}]"
            else:
                candidate = base_header

            count = seen_headers.get(candidate, 0) + 1
            seen_headers[candidate] = count
            headers.append(candidate if count == 1 else f"{candidate} ({count})")

        return headers

    def _get_anomaly_column_reduction_note(
        self,
        dynamic_col_count: int,
        avg_dynamic_width: float,
        avg_base_width: float,
    ) -> str:
        """Recommend anomaly_table_max_columns when columns are too compressed."""
        if not getattr(self.summary, "anomaly_table_show_notes", True):
            return ""
        cap = getattr(self.summary, "anomaly_table_max_columns", None)
        if dynamic_col_count < 1:
            return ""

        # Heuristics tuned for monospace 10pt table text.
        dynamic_compressed = avg_dynamic_width < 0.06
        base_compressed = avg_base_width < 0.085
        if not dynamic_compressed and not base_compressed:
            return ""

        if cap is not None:
            suggested = max(1, cap - 1)
            return (
                "Columns are compressed for fit. Current "
                f"model_auditor_overrides.anomaly_table_max_columns is {cap}; "
                f"consider lowering it (for example, {suggested})."
            )

        return (
            "Columns are compressed for fit. Consider setting "
            "model_auditor_overrides.anomaly_table_max_columns "
            "(for example, 8) in evaluation_settings.yaml."
        )

    def _render_model_legend(self) -> None:
        """
        Render the model type legend/glossary page with color indicators.
        """
        # create_new_page ensures standard branding and footer context
        pdf_page = self.pdf_doc.create_new_page(
            page_name="Legend / Glossary", print_page_name=False
        )
        fig = pdf_page.fig

        # 1. Page Heading
        fig.text(
            0.5, 0.88, "Model Audit Legend", fontsize=16, weight="bold", ha="center"
        )

        # 2. Prepare Data for Two-Column Layout
        # Group order is Classification -> Regression -> Unknown.
        # The right column starts at the Regression header so that no group header
        # is orphaned at the bottom of a column without its models below it.
        model_type_data = self._build_model_legend_records()

        # Find the index of the second group's header (Regression) to use as the
        # column split point, ensuring that header always leads the right column.
        second_group_start = next(
            (
                i
                for i, mtd in enumerate(model_type_data)
                if mtd.rec_type == ModelTypeDataRecType.HEADER and i > 0
            ),
            len(model_type_data) // 2,
        )
        mid = second_group_start

        # Layout constants for precise positioning
        line_height = 0.04
        start_y = 0.80
        left_x = 0.15
        right_x = 0.55
        bar_width = 0.015
        text_offset = 0.025

        for i, mtd in enumerate(model_type_data):
            # Determine column and vertical position
            is_left = i < mid
            col_x = left_x if is_left else right_x
            row_idx = i if is_left else i - mid
            current_y = start_y - (row_idx * line_height)

            if mtd.rec_type == ModelTypeDataRecType.DATA:
                # Retrieve the specific solid color for this model architecture
                color = self.summary.solid_color_palette[mtd.value]

                # 3. Add the Identifying Color Bar
                # Matches the colored bars seen in the legend for RFR, DTR, etc.
                rect = Rectangle(
                    (col_x, current_y - 0.008),
                    bar_width,
                    0.025,
                    facecolor=color,
                    transform=fig.transFigure,
                    clip_on=False,
                )
                fig.patches.append(rect)

                # 4. Add Abbreviation and Model Name
                label = f"{mtd.abbrev}: {EnumFormat().format_value(mtd.model_type)}"
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
                # 5. Add Section Headers (e.g., -- Regression --)
                label = f"-- {EnumFormat().format_value(mtd.task_type)} --"
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

        # 6. Bottom Technical Note
        enum_fmt = EnumFormat()
        score_name = enum_fmt.format_value(self.best_model.model.scoring)
        task_name = enum_fmt.format_value(self.best_model.model.task_type)
        fig.text(
            0.5,
            0.1,
            f"Note: 'Score' refers to {score_name} for this {task_name} audit.",
            fontsize=9,
            style="italic",
            ha="center",
            transform=fig.transFigure,
        )

    @staticmethod
    def _build_model_legend_records() -> list[ModelTypeData]:
        """Build ordered legend records for two-column rendering."""
        group_order = {
            TaskType.CLASSIFICATION: 0,
            TaskType.REGRESSION: 1,
            TaskType.UNKNOWN: 2,
        }

        data_rows = ModelTypeData.get_list(
            sort_order=ModelEnumSortOrder.TASK_TYPE_NAME,
            include_task_type_headers=False,
        )
        data_rows = sorted(
            data_rows,
            key=lambda mtd: (group_order.get(mtd.task_type, 99), mtd.name),
        )

        final_list: list[ModelTypeData] = []
        current_task_type: TaskType | None = None
        for mtd in data_rows:
            if mtd.task_type != current_task_type:
                final_list.append(ModelTypeData.create_header_from_item(mtd))
                current_task_type = mtd.task_type
            final_list.append(mtd)

        return final_list

    def _get_audit_data(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Parse ModelConfiguration objects into structures for reporting and plotting.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, pd.DataFrame]]
            A DataFrame of performance metrics and a dictionary of importance DataFrames keyed by ID.
        """
        performance_rows: list[dict[str, Any]] = []
        importance_dict: dict[str, pd.DataFrame] = {}

        for config in self.results:
            # 1. Extract Performance Metrics for the leaderboard and charts
            row = {
                "ID": config.id,
                "Model": config.model_type.value,
                "Abbr": config.model_type.abbrev,
                "Strategy": config.balancing_strategy,
                "Available RAM": config.available_gb,
                "Est Peak RAM": config.estimated_peak_gb,
                "Actual Peak RAM": config.actual_peak_gb,
                "Memory Risk": config.memory_risk_triggered,
                "Sampling Pct": config.sampling_factor,
                "n_jobs": config.concurrent_workers,
                "CV Score (Tuning)": config.score_cv,
                "Val Score": config.val_score,
                "Test Score": (
                    config.score_test if config.has_test_set_evaluation_scores else None
                ),
                "Cleaned Score": config.score_val_cleaned,
                "Train Time (s)": config.total_duration,
                "Efficiency": config.efficiency(self.summary.data_splits),
                "Train Score": config.train_score,
                "Gap": config.gap,
                "Status": config.model_generalization.value,
                "Mean Delta": config.mean_delta,
            }

            # 2. Add task-specific metrics (at least 3 per task type requirement)
            if config.task_type == TaskType.REGRESSION:
                # Regression: MAE, MSE, R² (3 metrics)
                row["MAE"] = config.mae_val
                row["MSE"] = config.mse_val
                row["R²"] = config.r2_val
            else:  # CLASSIFICATION
                # Classification: Accuracy, ROC-AUC, primary scoring metric (F1/etc.)
                row["Accuracy"] = config.accuracy_val
                row["ROC-AUC"] = config.roc_auc_val
                # Val Score is already added above (e.g., F1 when scoring=F1)

            performance_rows.append(row)

            # 3. Extract Feature Importances for deep-dive plotting
            # We look for the feature_analysis attribute on the configuration object
            if hasattr(config, "feature_analysis") and config.feature_analysis:
                imp_df = config.feature_analysis.feature_importances.copy()
            else:
                imp_df = pd.DataFrame()

            importance_dict[config.id] = imp_df

        perf_df = pd.DataFrame(performance_rows)
        return perf_df, importance_dict

    def _check_data_leakage(self) -> tuple[str, str, float]:
        """
        Evaluate if Train/Val distributions have drifted significantly.

        Returns
        -------
        tuple[str, str, float]
            The status string, the associated semantic color, and the raw drift index.
        """
        model = self.best_model.model
        # Check against the summary-level threshold (typically 0.05 or 5%)
        is_safe = model.drift_index < self.summary.drift_threshold

        # Status text and color mapping
        status = "SAFE" if is_safe else "WARNING: DRIFT DETECTED"
        color = prefs.color_success if is_safe else prefs.color_danger

        return status, color, model.drift_index

    def _plot_predictive_accuracy(self, sns: Any, ax: Axes) -> None:
        """
        Plot validation accuracy bars with optional cleaned-score overlay.
        """
        # 1. Determine if an 'Outlier Impact' shadow bar should be rendered
        has_cleaned = (
            "Cleaned Score" in self.summary_df.columns
            and self.summary_df["Cleaned Score"].notnull().any()
            and not (
                self.summary_df["Cleaned Score"] == self.summary_df["Val Score"]
            ).all()
        )

        # 2. Plot the 'Raw' Actual Score (Solid Bars)
        sns.barplot(
            x="Val Score",
            y="Model",
            data=self.summary_df,
            ax=ax,
            alpha=0.9,
            hue="Model",
            palette=self.summary.solid_color_palette,
            legend=False,  # Prevent duplicate entries in the final legend
        )

        # 3. Plot the 'Cleaned' Potential Score (Shadow Overlay)
        if has_cleaned:
            sns.barplot(
                x="Cleaned Score",
                y="Model",
                data=self.summary_df,
                ax=ax,
                alpha=0.3,
                hue="Model",
                palette=self.summary.light_color_palette,
                legend=False,  # Prevent duplicate entries in the final legend
                zorder=0,  # Positioned behind solid bars
            )

        # 4. Resolve Metric Description dynamically from the scoring enum
        score_desc = EnumFormat().format_value(self.best_model.model.scoring)

        # 5. Build Manual Professional Legend
        legend_handles = [
            mpatches.Patch(
                color=prefs.color_neutral, alpha=0.9, label="Raw Score (Actual)"
            )
        ]

        if has_cleaned:
            legend_handles.append(
                mpatches.Patch(
                    color=prefs.color_neutral,
                    alpha=0.3,
                    label="Outliers Filtered (Potential)",
                )
            )
            title_suffix = "& Outlier Impact"
        else:
            title_suffix = "- Baseline Performance"

        ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True)
        ax.set_title(f"Predictive Accuracy ({score_desc}) {title_suffix}")

        # 6. Axes Cleanup and Tick Alignment
        ax.set_xlabel(f"Score ({score_desc})", fontsize=10)
        ax.set_ylabel("")  # Model names are usually self-explanatory
        ax.set_yticks(range(len(self.summary_df)))
        ax.set_yticklabels(
            self.summary_df["Abbr"], ha="left", va="center", position=(-0.08, -0.02)
        )

        # Use a full-scale baseline so bar lengths remain visually meaningful
        # across datasets with narrow score ranges (e.g., Adult Income).
        c_min = float(self.summary_df["Val Score"].min())
        c_max = float(self.summary_df["Val Score"].max())
        left_limit = min(0.0, c_min - 0.05)
        right_limit = max(1.0, c_max + 0.02)
        ax.set_xlim(left=left_limit, right=right_limit)

        # 7. Add Centered Value Labels
        # Identify the solid bars directly from alpha so labels render
        # whether or not the cleaned-score overlay is present.
        raw_score_bars: list[Rectangle] = []
        for p in cast(list[Rectangle], ax.patches):
            alpha = p.get_alpha()
            alpha_val = float(alpha) if alpha is not None else 1.0
            if p.get_width() > 0 and alpha_val >= 0.85:
                raw_score_bars.append(p)
        if not raw_score_bars:
            raw_score_bars = cast(list[Rectangle], ax.patches[: len(self.summary_df)])
        x_visible_min = ax.get_xlim()[0]
        width_fmt = FloatFormat(precision=4)

        for p in raw_score_bars:
            width = p.get_width()
            y_pos = p.get_y()
            height = p.get_height()
            bar_start = max(0, x_visible_min)

            # Determine text placement: Inside bar (paper color) or Outside (neutral)
            if width > (x_visible_min + 0.15):
                text_color = prefs.color_paper
                text_pos = (bar_start + width) / 2
                text_align = "center"
            else:
                text_color = prefs.color_neutral
                text_pos = width + 0.005
                text_align = "left"

            ax.text(
                text_pos,
                y_pos + height / 2,
                f"{width_fmt.format_value(width)}",
                va="center",
                ha=text_align,
                color=text_color,
                fontsize=10,
                weight="bold",
            )

    def _plot_efficiency_scatter(self, sns: Any, ax: Axes) -> None:
        """
        Plot model efficiency as an accuracy vs. training time scatter plot.

        Bubble size represents Mean Absolute Error (MAE) for regression, or is uniform for classification.
        """
        # 1. Generate the Scatter Plot
        # Classification runs have no MAE column; if size maps to all-NaN values,
        # seaborn drops all rows and the chart appears blank.
        plot_df = self.summary_df.copy()

        # Check if MAE column exists (regression only) before trying to access it
        has_valid_mae = False
        if "MAE" in plot_df.columns:
            mae_values = pd.to_numeric(plot_df["MAE"], errors="coerce")
            has_valid_mae = mae_values.notna().any()
        else:
            mae_values = None

        # Jitter points that overlap exactly (same Val Score AND same Train Time
        # within tolerance) so both dots remain visible in the static chart.
        # The offset is ±1 % of the visible y-range — negligible on the scale of
        # the chart but enough to prevent complete occlusion.
        x_col = "Train Time (s)"
        y_col = "Val Score"
        y_range = plot_df[y_col].max() - plot_df[y_col].min()
        x_tol = (plot_df[x_col].max() - plot_df[x_col].min()) * 0.01
        y_tol = y_range * 0.01
        # Step must be large enough to visually clear the bubble radii.
        # Bubble sizes=(100,500) means a max radius of ~12 pts; 4% of the
        # y-range is typically large enough to separate dots at standard
        # figure sizes without distorting the chart scale.
        y_step = max(y_range * 0.04, 0.015)
        has_overlap = False
        seen: list[tuple[float, float]] = []
        jittered_y = plot_df[y_col].tolist()
        jittered_x = plot_df[x_col].tolist()
        for i, (xi, yi) in enumerate(zip(jittered_x, jittered_y)):
            slot = 0
            for sx, sy in seen:
                if abs(xi - sx) <= x_tol and abs(yi - sy) <= y_tol:
                    slot += 1
            if slot > 0:
                has_overlap = True
                jittered_y[i] = yi + slot * y_step
            seen.append((xi, yi))
        plot_df = plot_df.copy()
        plot_df[y_col] = jittered_y

        s_min, s_max = 100.0, 500.0
        if has_valid_mae and mae_values is not None:
            plot_df["_mae_size"] = mae_values
            valid_mae = plot_df["_mae_size"].dropna()
            if not valid_mae.empty and valid_mae.max() > valid_mae.min():
                plot_df["_bubble_size"] = s_min + (
                    (plot_df["_mae_size"] - valid_mae.min())
                    / (valid_mae.max() - valid_mae.min())
                ) * (s_max - s_min)
                plot_df["_bubble_size"] = plot_df["_bubble_size"].fillna(s_min)
            else:
                plot_df["_bubble_size"] = (s_min + s_max) / 2
        else:
            plot_df["_bubble_size"] = 220.0

        # Draw larger bubbles first so smaller overlapping models remain visible.
        # This avoids seaborn hue-order occlusion where one marker can completely
        # hide another even after jitter.
        draw_df = plot_df.sort_values("_bubble_size", ascending=False)
        for _, row in draw_df.iterrows():
            ax.scatter(
                row[x_col],
                row[y_col],
                s=float(row["_bubble_size"]),
                c=self.summary.solid_color_palette.get(
                    row["Model"], prefs.color_neutral
                ),
                alpha=0.92,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )

        ax.set_xscale("linear")
        score_desc = EnumFormat().format_value(self.best_model.model.scoring)
        ax.set_title(f"Efficiency: {score_desc} vs. Time")
        ax.set_xlabel("Train Time (s)", fontsize=10)
        ax.set_ylabel("Val Score", fontsize=10)

        # 2. Legend: note only (no model name entries).
        # Model colors are consistent across all charts and documented on the
        # Legend / Glossary page, so repeating them here adds no information and
        # risks obscuring bubbles in the tight scatter area.
        final_handles: list[Any] = []
        final_labels: list[str] = []

        final_handles.append(mpatches.Patch(color="none"))
        if has_valid_mae:
            final_labels.append("Note: Bubble size = MAE\n(Smaller is better)")
        else:
            final_labels.append("Note: Uniform marker size\n(MAE unavailable)")

        if has_overlap:
            final_handles.append(mpatches.Patch(color="none"))
            final_labels.append("")
            final_handles.append(mpatches.Patch(color="none"))
            final_labels.append("Warning: Overlapping points offset\nfor visibility")

        leg = ax.legend(
            handles=final_handles,
            labels=final_labels,
            title=None,
            loc="best",
            fontsize=8,
            handlelength=0,
            handletextpad=0,
            frameon=True,
            ncol=1,
        )

        # 3. Stylize Legend Annotation
        # Apply the neutral color and italics defined in the V1.2.0 preferences.
        if leg.get_texts():
            for txt in leg.get_texts():
                plt.setp(txt, color=prefs.color_neutral, fontsize=7, style="italic")

    def _get_narrow_x_limit(
        self, y_train: pd.Series, y_val: pd.Series, percentile: float = 99.5
    ) -> tuple[float, float]:
        """Compute narrow x-axis limits using percentile clipping to avoid outlier distortion."""
        t_limit = np.percentile(y_train, percentile)
        v_limit = np.percentile(y_val, percentile)
        upper_limit = max(t_limit, v_limit)

        # Ensure we capture zero for positive-only data like fare_amount
        lower_limit = min(np.percentile(y_train, 0.5), 0.0)
        return float(lower_limit), float(upper_limit)

    def _plot_target_distribution(self, ax: Axes) -> None:
        """Visualize the statistical overlap between Train and Validation target distributions."""
        model = self.best_model.model
        ds = self.summary.data_splits
        y_train, y_val = ds.train_target, ds.val_target

        # Classification targets can be strings/categorical labels and do not
        # support percentile/KDE numeric operations.
        if not pd.api.types.is_numeric_dtype(
            y_train
        ) or not pd.api.types.is_numeric_dtype(y_val):
            self._plot_categorical_target_distribution(
                ax=ax, y_train=y_train, y_val=y_val
            )
            return

        # 1. Coordinate Percentile Clipping
        x_min, x_max = self._get_narrow_x_limit(y_train, y_val)

        # 2. Render KDE Distributions
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

        ax.set_xlim(x_min, x_max * 1.05)

        # 3. Add Professional Clipping Note
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

        # 4. Draw Central Tendency Lines
        # Train Baseline (Neutral)
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

        # Audit Validation (Classic Blue)
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

        # 5. Build the Statistical Anchor Box
        s_fmt = FloatFormat(precision=2)
        d_fmt = PercentageFormat(precision=4, width=8)

        stats_text = (
            f"TRAINING BASELINE\n"
            f"Mean:  {s_fmt.format_value(model.train_mean)} (Std Dev: {s_fmt.format_value(model.train_std)})\n"
            f"Skew:  {s_fmt.format_value(model.train_skew)} | Kurt: {s_fmt.format_value(model.train_kurtosis)}\n\n"
            f"AUDIT VALIDATION\n"
            f"Mean:  {s_fmt.format_value(model.val_mean)} (Std Dev: {s_fmt.format_value(model.val_std)})\n"
            f"Skew:  {s_fmt.format_value(model.val_skew)} | Kurt: {s_fmt.format_value(model.val_kurtosis)}\n\n"
            f"SPLIT INTEGRITY CHECK\n"
            f"Mean Delta:    {d_fmt.format_value(model.mean_delta)}\n"
            f"Std Dev Delta: {d_fmt.format_value(model.std_delta)}"
        )

        at = AnchoredText(
            stats_text,
            prop=dict(
                size=7.5, family="monospace", color=prefs.color_neutral, linespacing=1.4
            ),
            frameon=True,
            loc="upper right",
            borderpad=1.0,
        )

        # Apply V1.2.0 Audit Branding to the box
        at.patch.set_boxstyle("round,pad=0.5,rounding_size=0.2")
        at.patch.set_edgecolor(prefs.color_neutral)
        at.patch.set_facecolor(prefs.color_paper)
        at.patch.set_alpha(0.9)
        at.patch.set_linewidth(0.8)

        ax.add_artist(at)
        ax.set_title("Target Distribution & Statistical Shape")
        ax.set_xlabel(f"Target Value ({ds.target_column})")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, loc="upper left")

    def _plot_categorical_target_distribution(
        self, ax: Axes, y_train: pd.Series, y_val: pd.Series
    ) -> None:
        """Visualize class distribution overlap for non-numeric targets."""
        ds = self.summary.data_splits

        train_counts = y_train.astype("string").fillna("<NA>").value_counts()
        val_counts = y_val.astype("string").fillna("<NA>").value_counts()

        categories = sorted(set(train_counts.index).union(set(val_counts.index)))
        x = np.arange(len(categories), dtype=float)
        width = 0.4

        train_values = [int(train_counts.get(category, 0)) for category in categories]
        val_values = [int(val_counts.get(category, 0)) for category in categories]

        ax.bar(
            x - width / 2,
            train_values,
            width,
            label="Train",
            color=prefs.color_neutral,
            alpha=0.75,
        )
        ax.bar(
            x + width / 2,
            val_values,
            width,
            label="Val",
            color=prefs.color_classic_blue,
            alpha=0.75,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
        ax.set_title("Target Distribution by Class")
        ax.set_xlabel(f"Class Label ({ds.target_column})")
        ax.set_ylabel("Count")

        # Reserve headroom so the boxed top-left categorical annotation clears
        # the tallest bar.
        max_count = max(train_values + val_values, default=0)
        ax.set_ylim(0, max(1, max_count * 1.18))

        ax.legend(fontsize=7, loc="upper right")

        ax.text(
            0.02,
            0.95,
            "Categorical target detected",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            style="italic",
            color=prefs.color_neutral,
            bbox={
                "boxstyle": "round,pad=0.25,rounding_size=0.15",
                "facecolor": prefs.color_paper,
                "edgecolor": prefs.color_neutral,
                "linewidth": 0.8,
                "alpha": 0.9,
            },
        )

    def _plot_cumulative_importance(self, sns: Any, ax: Axes) -> None:
        """
        Plot cumulative feature importance for the winning model.

        Identifies the core feature set responsible for 95% of the model's
        predictive variance.
        """
        model = self.best_model.model
        best_model_name = model.model_type.value

        # Retrieve importance data for the top-performing model
        best_model_importance = (
            self.importance_dict[model.id].copy().reset_index(drop=True)
        )

        def _render_importance_na_message(message: str) -> None:
            ax.set_title(f"Cumulative Importance\n({best_model_name})")
            ax.set_axis_off()
            ax.text(
                0.5,
                0.58,
                message,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                style="italic",
                color=prefs.color_neutral,
            )

        if best_model_importance.empty or "importance" not in best_model_importance:
            _render_importance_na_message(
                "Feature importance unavailable\nfor this model type."
            )
            return

        best_model_importance["feature_idx"] = range(1, len(best_model_importance) + 1)

        # Some models (e.g., linear models using |coef|) store raw importance
        # magnitudes rather than pre-normalized shares. Normalize here so the
        # cumulative curve and 95% threshold logic are always on a 0-1 scale.
        imp_series = pd.to_numeric(best_model_importance["importance"], errors="coerce")
        imp_series = cast(pd.Series, imp_series.fillna(0.0).abs())
        imp_total = float(imp_series.sum())
        if imp_total <= 0.0:
            _render_importance_na_message("No non-zero feature importance values.")
            return

        best_model_importance["importance_plot"] = imp_series / imp_total
        best_model_importance["cumulative_importance_plot"] = best_model_importance[
            "importance_plot"
        ].cumsum()

        # 1. Draw the Base Cumulative Line
        sns.lineplot(
            x="feature_idx",
            y="cumulative_importance_plot",
            data=best_model_importance,
            marker=None,
            color=self.summary.solid_color_palette[best_model_name],
            ax=ax,
            linewidth=1.3,
            alpha=0.6,
            zorder=1,
        )

        # 2. Apply Multi-Colored Markers for Top Features
        # Color all dots up to the 95% threshold, but cap legend entries so
        # the legend never obscures the chart on high-cardinality feature sets.
        n_labels = 5
        MAX_LEGEND_ENTRIES = 10
        # Feature names longer than this are truncated and suffixed with [ID]
        # so that one-hot-encoded column families remain identifiable while
        # keeping legend labels narrow enough not to overlap the chart.
        MAX_LABEL_CHARS = 16
        colors = sns.color_palette("tab10", n_colors=10)
        legend_handles = []

        def _legend_label(name: str, fid: str) -> str:
            if len(name) <= MAX_LABEL_CHARS:
                return name
            return name[:MAX_LABEL_CHARS].rstrip("_- ") + f"... [{fid}]"

        cum_series = pd.to_numeric(
            best_model_importance["cumulative_importance_plot"], errors="coerce"
        ).fillna(0.0)
        cum_series = cast(pd.Series, cum_series.reset_index(drop=True))
        threshold_idx = next(
            (idx for idx, value in enumerate(cum_series) if value >= 0.95),
            len(cum_series) - 1,
        )

        for i in range(len(best_model_importance)):
            row = best_model_importance.iloc[i]
            cum_imp = row["cumulative_importance_plot"]
            feat_name = row["feature"]
            feat_id = row.get("id", "")

            # Draw a marker for every feature point. Points above the 95%
            # threshold are rendered in a consistent success color.
            is_above_threshold = i > threshold_idx
            dot_color = (
                prefs.color_success if is_above_threshold else colors[i % len(colors)]
            )

            ax.scatter(
                i + 1,
                cum_imp,
                marker="o",
                s=70,
                color=dot_color,
                edgecolors="none",
                linewidths=0.0,
                alpha=1.0,
                zorder=4,
            )

            # Keep legend focused and bounded: only include features that
            # contribute to reaching 95% cumulative importance.
            contributes_to_95 = i < n_labels or (
                i > 0 and cum_series.iloc[i - 1] < 0.95
            )
            if contributes_to_95 and len(legend_handles) < MAX_LEGEND_ENTRIES:
                label = _legend_label(feat_name, feat_id)
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markerfacecolor=dot_color,
                        markeredgecolor="none",
                        markeredgewidth=0.0,
                        markersize=6,
                        label=label,
                    )
                )

        # 3. Add 95% Threshold and Multi-Column Legend
        thresh_line = ax.axhline(
            y=0.95,
            color=prefs.color_neutral,
            linestyle="--",
            label="95% Threshold",
            zorder=2,
        )

        feature_legend = ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=7,
            frameon=True,
            title="Top Features",
            ncol=2,
            columnspacing=0.5,
            handletextpad=0.4,
            labelspacing=0.3,
        )
        ax.add_artist(feature_legend)

        # Keep threshold in a separate compact legend to preserve 5x5 feature
        # alignment while still explicitly labeling the dashed reference line.
        ax.legend(
            handles=[thresh_line],
            loc="upper right",
            bbox_to_anchor=(1.0, 0.9),
            bbox_transform=ax.transAxes,
            fontsize=7,
            frameon=True,
            borderpad=0.3,
            handlelength=1.8,
            handletextpad=0.4,
        )
        # 4. Final Axes and Labeling
        ax.set_title(f"Cumulative Importance\n({best_model_name})")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Total Variance Explained")
        ax.set_ylim(0, 1.05)

    def _render_audit_results(self) -> None:
        """
        Generates a professional performance dashboard from audit results.

        Orchestrates a four-quadrant layout including accuracy benchmarks,
        computational efficiency, statistical shape, and feature importance.
        """
        # 1. Initialize Dashboard Page
        pdf_page = self.pdf_doc.create_new_page(
            page_name="High-Level Competition Results"
        )
        fig = pdf_page.fig

        # 2. Define the Quadrant Grid System
        # We use a 2x2 grid with equal height ratios for visual balance
        with sns.axes_style(style="whitegrid"):
            gs = fig.add_gridspec(2, 2, height_ratios=[0.5, 0.5])

            # Subplot Assignments
            ax_acc = fig.add_subplot(gs[0, 0])  # Top Left: Predictive Accuracy
            ax_eff = fig.add_subplot(gs[0, 1])  # Top Right: Efficiency Scatter
            ax_dist = fig.add_subplot(gs[1, 0])  # Bottom Left: Target Distribution
            ax_imp = fig.add_subplot(gs[1, 1])  # Bottom Right: Cumulative Importance

            # 3. Data Integrity "Safe-Guard" Indicator
            # This logic captures the drift score seen at the bottom of Page 6
            status, color, diff = self._check_data_leakage()

            fig.text(
                0.5,
                0.04,
                f"Data Integrity Check: {status} (Drift: {prefs.drift_format.format_value(diff)})",
                ha="center",
                weight="bold",
                color=color,
                fontsize=12,
                transform=fig.transFigure,
            )

            # 4. Populate Visual Diagnostics
            # Quadrant 1: Performance Benchmark (R² / F1)
            self._plot_predictive_accuracy(sns, ax_acc)

            # Quadrant 2: Accuracy vs. Training Latency
            self._plot_efficiency_scatter(sns, ax_eff)

            # Quadrant 3: Train/Val Population Overlap
            self._plot_target_distribution(ax_dist)

            # Quadrant 4: Pareto Feature Analysis
            self._plot_cumulative_importance(sns, ax_imp)

    def _render_cv_vs_final(self) -> None:
        """
        Render the CV vs. final validation comparison page.

        This chart visualizes the generalization gap between tuning samples
        and the full data audit.
        """
        # 1. Initialize Page and Axis
        pdf_page = self.pdf_doc.create_new_page(page_name="Generalization (CV vs Full)")
        fig = pdf_page.fig
        ax = fig.add_subplot(1, 1, 1)

        # 2. Plot CV Scores (Hatched/Light Bars)
        sns.barplot(
            data=self.summary_df,
            x="CV Score (Tuning)",
            y="Model",
            hue="Model",
            palette=self.summary.light_color_palette,
            alpha=0.7,
            hatch="//",  # Distinguishes tuning from final Fit
            ax=ax,
            legend=False,
        )

        # 3. Plot Final Validation Scores (Solid Bars)
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

        # 4. Delta Label Logic (The "$\Delta$" indicators)
        # We target solid bars to position the delta text correctly
        for p in ax.patches:
            rect = cast(Rectangle, p)
            if rect.get_hatch() is None:  # Process only solid bars
                y_center = rect.get_y() + rect.get_height() / 2
                row_idx = int(round(y_center))

                if 0 <= row_idx < len(self.summary_df):
                    final_val = float(rect.get_width())
                    delta = self.summary_df.iloc[row_idx]["Mean Delta"]

                    if final_val > 0:
                        # Success (Green) for positive gain, Danger (Red) for loss
                        t_color = (
                            prefs.color_success if delta > 0 else prefs.color_danger
                        )
                        delta_fmt = FloatFormat(precision=4, always_include_sign=True)

                        ax.text(
                            final_val + 0.01,
                            y_center,
                            f"Delta: {delta_fmt.format_value(delta)}",
                            va="center",
                            ha="left",
                            fontsize=9,
                            color=t_color,
                            weight="bold",
                        )

        # 5. Professional Legend and Axis Formatting
        cv_patch = mpatches.Patch(
            facecolor=prefs.color_neutral,
            alpha=0.6,
            hatch="//",
            label="CV Score (Tuning)",
        )
        val_patch = mpatches.Patch(
            facecolor=prefs.color_neutral, alpha=0.8, label="Final Score (Full Audit)"
        )

        ax.legend(
            handles=[cv_patch, val_patch],
            loc="lower right",
            title="Audit Stage",
            fontsize=9,
        )

        score_desc = EnumFormat().format_value(self.best_model.model.scoring)
        ax.set_title("Generalization Gap (CV vs Full Data)")
        ax.set_xlabel(f"Score ({score_desc})")
        ax.set_ylabel("")
        ax.set_yticks(range(len(self.summary_df)))
        ax.set_yticklabels(
            self.summary_df["Abbr"], ha="left", va="center", position=(-0.05, 0.0)
        )
        ax.set_xlim(0, 1.15)  # Provide space for delta labels

    def _plot_model_hyperparameters(
        self,
        pdf_page: PDFDocument.Page,
        config: ModelConfiguration,
        ax_params: Axes,
        renderer: RendererBase,
    ) -> None:
        """
        Render a hyperparameter table for a single model configuration.
        """
        ax_params.axis("off")
        ax_params.set_xlim(0.0, 1.0)
        ax_params.set_ylim(0.0, 1.0)
        # Keep deep-dive titles below the styled page header. Global rcParams
        # use a large title pad that is too high for this compact grid.
        ax_params.set_title("Configuration", pad=14)

        # 1. Prepare Data from Dataclass
        # Converts the model dials into display-friendly string pairs
        dials = config.params_dict
        table_data = []

        for k, v in dials.items():
            val = v.value if isinstance(v, Enum) else v
            table_data.append([prefs.get_hyperparameter_display_name(k), str(val)])

        columns = ["HP", "Value"]
        df = pd.DataFrame(table_data, columns=columns)

        # 2. Define Table Styles
        col_header_edge = TableEdgeColor(
            left=prefs.color_neutral, right=prefs.color_neutral
        )

        # Consistent style for HP and Value columns
        def get_col_config(
            max_width: float,
            lpad: float = 10.0,
            rpad: float = 10.0,
            max_width_chars: int | None = None,
        ) -> TableColumn:
            return TableColumn(
                header_style=TableColumnStyle(
                    fontweight="bold",
                    ha="center",
                    va="center",
                    edge_color=col_header_edge,
                    face_color="black",
                    text_color="white",
                ),
                detail_style=TableColumnStyle(
                    ha="left",
                    va="center",
                    text_color=prefs.color_neutral,
                    edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                ),
                even_row_style=TableColumnStyle(
                    ha="left",
                    va="center",
                    text_color=prefs.color_neutral,
                    face_color=prefs.color_light_gray,
                    edge_color=TableEdgeColor.closed(color=prefs.color_neutral),
                ),
                lpad=lpad,
                rpad=rpad,
                max_proportional_width=max_width,
                max_width_chars=max_width_chars,
            )

        # Keep Value readable while giving HP labels a bit more room to avoid
        # touching vertical borders in regression deep-dive pages.
        table_columns = {
            columns[0]: get_col_config(
                max_width=0.44,
                lpad=10.0,
                rpad=12.0,
                # max_width_chars=14 if max_hp_chars >= 16 else 18,
            ),
            columns[1]: get_col_config(
                max_width=0.56,
                lpad=10.0,
                rpad=12.0,
                # max_width_chars=16 if max_value_chars > 16 else None,
            ),
        }

        # 3. Dynamic Scaling based on Row Count
        hp_count = len(df)
        if hp_count <= 22:
            fontsize, padding = 9, 10.0
        elif hp_count <= 30:
            fontsize, padding = 8, 6.0
        else:
            fontsize, padding = 6, 5.0

        # 4. Table Layout Calculation (Dry Run)
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

        # 5. Column Splitting Logic
        # Re-calculates axes for multi-column hyperparameter sets (e.g., Random Forest)
        ss = ax_params.get_subplotspec()
        if ss is None:
            raise RuntimeError(
                "ax_params must be a subplot for coordinate calculation."
            )

        num_pages = len(table_layout.pages)

        if num_pages > 1:
            ss = ax_params.get_subplotspec()
            if ss is None:
                raise RuntimeError(
                    "ax_params must be a subplot for coordinate calculation."
                )

            fig = pdf_page.fig
            target_pos = ss.get_position(fig)
            w_per_col = target_pos.width / num_pages
            spacing = 0.005

            for idx in range(num_pages):
                col_left = (
                    target_pos.x0 + (idx * w_per_col) + (spacing if idx > 0 else 0)
                )
                ax = fig.add_axes(
                    (col_left, target_pos.y0, w_per_col - spacing, target_pos.height)
                )
                ax.axis("off")
                render_table_from_page_layout(
                    pdf_page=pdf_page,
                    table_layout=table_layout,
                    page_index=idx,
                    using_axis=ax,
                    adjust_mid_x=False,
                )
        else:
            # Single-column tables should render directly in the subplot axis.
            # This preserves the expected top alignment and title spacing.
            render_table_from_page_layout(
                pdf_page=pdf_page,
                table_layout=table_layout,
                using_axis=ax_params,
            )

    def _plot_feature_importance(
        self,
        ax: Axes,
        importance_df: pd.DataFrame,
        model_color: str,
        test_metrics_text: str | None,
    ) -> None:
        """
        Plot feature importance bars or a fallback message for a specific model.
        """
        if not importance_df.empty:
            max_pdf_feature_bars = max(
                int(getattr(self.summary, "pdf_feature_importance_chart_limit", 12)),
                1,
            )

            # 1. Filter for active features only
            active_features = importance_df[importance_df["importance"] != 0.0]
            requested_count = min(len(active_features), self.summary.top_n_importance)
            display_count = min(requested_count, max_pdf_feature_bars)

            if display_count == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No non-zero feature importance values are available.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    style="italic",
                )
                ax.axis("off")
                return

            importances = active_features["importance"].iloc[:display_count]
            features = active_features["id"].iloc[:display_count]
            max_val = float(importances.max()) if not importances.empty else 1.0

            # 2. Coordinate System: Most important features at the top
            y_pos = np.arange(display_count)[::-1]
            # Thinner bars for dense feature lists to maintain legibility
            bar_h = 0.6 if display_count > 10 else 0.4

            ax.barh(y_pos, importances, height=bar_h, color=model_color, alpha=0.8)

            # 3. Labeling and Aesthetics
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_title(f"Top {display_count} Predictors", pad=14)
            ax.set_xlabel("Relative Importance / Weight", fontsize=9)
            ax.set_xlim(0, max_val * 1.15)  # Provide room for value labels
            ax.grid(axis="x", linestyle="--", alpha=0.6)

            if display_count < requested_count:
                ax.text(
                    0.0,
                    1.01,
                    (
                        f"Showing {display_count} of requested top {requested_count} "
                        "for PDF readability"
                    ),
                    transform=ax.transAxes,
                    fontsize=8,
                    color=prefs.color_neutral,
                    ha="left",
                    va="bottom",
                )

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            # 4. Value Labels (End of Bars)
            for i, v in enumerate(importances):
                ax.text(
                    v + (max_val * 0.01),
                    y_pos[i],
                    prefs.score_format.format_value(v),
                    va="center",
                    fontsize=8,
                    color=prefs.color_neutral,
                )

            # 5. Test Performance Summary Overlay
            if test_metrics_text:
                ax.text(
                    0.65,
                    0.05,
                    test_metrics_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
        else:
            # Fallback for models that do not provide coefficients/importance
            ax.text(
                0.5,
                0.5,
                "Feature Importance not supported for this model type.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                style="italic",
            )
            ax.axis("off")

    def _plot_regression_residuals(
        self,
        ax: Axes,
        y_true: pd.Series,
        y_preds: pd.Series,
        model_name: str,
        include_title: bool = True,
    ) -> None:
        """
        Visualize the error distribution and heteroscedasticity of a regressor.
        """
        model_color = self.summary.solid_color_palette[model_name]
        residuals = y_true - y_preds

        def get_optimized_limits(
            percentile: float = 0.99,
        ) -> tuple[tuple[float, float], tuple[float, float]]:
            """
            Compute symmetric Y-axis limits with buffered X-axis limits using percentiles.
            """
            # Symmetric Y-limit ensures the 0-line remains the visual equator
            limit_val = float(np.percentile(np.abs(residuals), percentile * 100))
            y_limit = max(
                limit_val * 1.1, 1.0
            )  # 10% breathing room; min span guards constant residuals

            # X-limit (Predicted Values) clipping to avoid extreme outlier stretching
            x_min = float(np.percentile(y_preds, (1 - percentile) * 100))
            x_max = float(np.percentile(y_preds, percentile * 100))
            x_buffer = max(
                (x_max - x_min) * 0.05, 1.0
            )  # min span guards constant predictions

            return (-y_limit, y_limit), (x_min - x_buffer, x_max + x_buffer)

        # 1. High-Density Scatter Plot
        # Rasterized=True is critical for 1M+ row datasets to keep PDF performance high
        ax.scatter(
            y_preds, residuals, alpha=0.1, color=model_color, s=1, rasterized=True
        )

        # 2. Reference Zero-Line
        ax.axhline(0, color="#bdc3c7", linestyle="--", lw=1.5, zorder=3)

        # 3. Apply Optimized Viewport
        y_lim, x_lim = get_optimized_limits()
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)

        if include_title:
            # Lower-row titles are intentionally anchored lower so they align
            # with the inset content region on the bottom quadrants.
            ax.set_title("Validation Residual Analysis", y=0.86)
        ax.set_xlabel("Predicted Values", fontsize=9)
        ax.set_ylabel("Residuals (Error)", fontsize=9)

        # 4. Professional Styling
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    def _plot_classification_diagnostics(
        self, ax: Axes, y_preds: pd.Series, y_true: pd.Series
    ) -> None:
        """
        Plot a confusion matrix to identify specific classification error patterns.
        """
        # Confusion Matrix calculation
        cm = confusion_matrix(y_true, y_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # The parent axis provides the title and defines the bounding region.
        ax.set_axis_off()
        ax.set_title("Validation Confusion Matrix", y=0.86)

        # Place a fixed-size axis in absolute figure coordinates so the matrix
        # is centred regardless of constrained-layout adjustments.  The
        # deep-dive page disables the layout engine, so get_position() already
        # returns the final bounding box.
        fig = ax.figure
        pos = ax.get_position()  # Bbox in figure (0–1) coordinates
        # Keep the matrix slightly smaller/lower within the lower quadrant so
        # the title has visible padding above and below.
        side = min(pos.width, pos.height) * 0.66
        cx = pos.x0 + pos.width / 2
        # Shift slightly below axis midpoint to leave visual room for the title.
        cy = pos.y0 + pos.height * 0.40
        matrix_ax = fig.add_axes((cx - side / 2, cy - side / 2, side, side))
        # Exclude from any future layout passes so the position stays fixed.
        matrix_ax.set_in_layout(False)

        # Plotting using the Audit Blue theme
        disp.plot(ax=matrix_ax, cmap="Blues", colorbar=False)
        matrix_ax.set_xlabel("")
        matrix_ax.set_ylabel("")

    def _plot_residual_analysis(
        self, config: ModelConfiguration, ax_residuals: Axes
    ) -> None:
        """
        Dispatch the appropriate residual or diagnostic plot based on the task type.
        """
        if config.preds_val is None:
            ax_residuals.text(
                0.5, 0.5, "No validation predictions available.", ha="center"
            )
            ax_residuals.axis("off")
            return

        # 1. Regression Path: Residual Scatter
        if config.task_type == TaskType.REGRESSION:
            # Keep title on the parent axis and render the residual plot in a
            # shorter inset so its top aligns with the misses-table content.
            ax_residuals.set_axis_off()
            ax_residuals.set_title("Validation Residual Analysis", y=0.86)
            residual_plot_ax = ax_residuals.inset_axes((0.08, 0.14, 0.84, 0.76))
            self._plot_regression_residuals(
                ax=residual_plot_ax,
                y_true=self.summary.data_splits.val_target,
                y_preds=config.preds_val,
                model_name=config.model_type.value,
                include_title=False,
            )

        # 2. Classification Path: Sampled Confusion Matrix
        else:

            def get_plot_ready_data(
                y_true: pd.Series,
                y_preds: pd.Series,
                sample_size: int = prefs.default_plot_sample_size,
            ) -> tuple[pd.Series, pd.Series]:
                """
                Returns a representative stratified subset of data for heatmaps.
                """
                if len(y_true) <= sample_size:
                    return y_true, y_preds

                # Combine for synchronized sampling
                df = pd.DataFrame({"true": y_true, "pred": y_preds})
                n_classes = df["true"].nunique()

                # Stratified sample to preserve class balance in the diagnostic
                df_sample = df.groupby("true", group_keys=False).apply(
                    lambda x: x.sample(min(len(x), sample_size // n_classes))
                )

                return df_sample["true"], df_sample["pred"]

            # 3. Apply Sampling and Render
            y_t, y_p = get_plot_ready_data(
                y_true=self.summary.data_splits.val_target,
                y_preds=config.preds_val,
            )
            self._plot_classification_diagnostics(
                ax=ax_residuals,
                y_true=y_t,
                y_preds=y_p,
            )

    def _plot_worst_residual_errors(
        self,
        pdf_page: PDFDocument.Page,
        ax: Axes,
        y_true: pd.Series,
        y_preds: pd.Series,
        n: int,
        renderer: RendererBase,
    ) -> None:
        """
        Render a table of the top regression errors (outliers).
        """
        ax.set_title("Top Validation Misses (Regression)", y=0.86)
        ax.axis("off")

        # 1. Error Magnitude Calculation
        abs_error = np.abs(y_true - y_preds)
        # Avoid division by zero when calculating percentage impact
        pct_error = abs_error / np.where(y_true == 0, 1.0, y_true)

        actual_col, pred_col, abs_col, pct_col = (
            "Actual",
            "Predicted",
            "Abs Error",
            "Error Pct",
        )

        # 2. Identify Top Outliers
        df = (
            pd.DataFrame(
                {
                    actual_col: y_true,
                    pred_col: y_preds,
                    abs_col: abs_error,
                    pct_col: pct_error,
                }
            )
            .sort_values(by=abs_col, ascending=False)
            .head(n)
        )

        # 3. Apply Formatted Strings
        summary = self.summary
        col_fmts: dict[str, FormatConfig] = {
            actual_col: summary.actual_value_fmt,
            pred_col: summary.predicted_value_fmt,
            abs_col: summary.abs_error_fmt,
            pct_col: summary.error_pct_fmt,
        }

        for col, fmt in col_fmts.items():
            df[col] = [fmt.format_value(v) for v in df[col]]

        # 4. Table Style Orchestration
        table_columns: dict[str, TableColumn] = {}
        header_edge = TableEdgeColor(
            left=prefs.color_neutral, right=prefs.color_neutral
        )
        closed_edge = TableEdgeColor.closed(color=prefs.color_neutral)

        for col in df.columns:
            align = col_fmts[col].matplot_alignment()
            table_columns[col] = TableColumn(
                header_style=TableColumnStyle(
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    edge_color=header_edge,
                    face_color="black",
                    text_color="white",
                ),
                detail_style=TableColumnStyle(
                    ha=align,
                    va="center",
                    fontfamily="monospace",
                    edge_color=closed_edge,
                    face_color=prefs.color_paper,
                    text_color=prefs.color_neutral,
                ),
                even_row_style=TableColumnStyle(
                    ha=align,
                    va="center",
                    fontfamily="monospace",
                    edge_color=closed_edge,
                    face_color=prefs.color_light_gray,
                    text_color=prefs.color_neutral,
                ),
                has_consistent_width=True,
                has_consistent_height=True,
                lpad=10.0,
                rpad=15.0,
            )

        # 5. Render
        table = Table(
            data=df,
            max_table_height=0.90,
            mid_x=0.5,
            top_y=0.90,
            fontsize=10,
            columns=table_columns,
            cell_edge_linewidth=TableEdgeLinewidth.all_edges(0.4),
            table_edge_linewidth=TableEdgeLinewidth(),
            use_full_axis_width=True,
            header_tpad=20.0,
            header_bpad=0.0,
            detail_tpad=12.0,
            detail_bpad=12.0,
        )
        render_table(pdf_page=pdf_page, table=table, ax=ax, renderer=renderer)

    def _plot_worst_classification_errors(
        self,
        pdf_page: PDFDocument.Page,
        ax: Axes,
        y_true: pd.Series,
        y_probs: pd.DataFrame,
        y_preds: pd.Series,
        n: int,
        renderer: RendererBase,
    ) -> None:
        """
        Render a table of classification "confidence misses" (where the model was confidently wrong).
        """
        ax.set_title(f"Top {n} Validation Misses", y=0.86)
        ax.axis("off")

        # 1. Probability Logic: Find where the model was wrong
        incorrect = y_true != y_preds
        # Confidence is the peak probability assigned to the WRONG class
        confidences = y_probs.max(axis=1)

        actual_col, pred_col, conf_col = "Actual", "Predicted", "Confidence"

        df = (
            pd.DataFrame(
                {
                    actual_col: y_true,
                    pred_col: y_preds,
                    conf_col: confidences,
                }
            )[incorrect]
            .sort_values(conf_col, ascending=False)
            .head(n)
        )

        # 2. Formatter Dispatch
        summary = self.summary
        col_fmts = {
            actual_col: summary.actual_value_fmt,
            pred_col: summary.predicted_value_fmt,
            conf_col: summary.abs_error_fmt,  # Reusing for probability display
        }

        for col, fmt in col_fmts.items():
            df[col] = [fmt.format_value(v) for v in df[col]]

        # 3. Table Rendering
        if df.empty:
            ax.text(
                0.5,
                0.45,
                "No validation misses found.",
                ha="center",
                va="center",
                style="italic",
            )
            return

        table_columns: dict[str, TableColumn] = {}
        header_edge = TableEdgeColor(
            left=prefs.color_neutral, right=prefs.color_neutral
        )
        closed_edge = TableEdgeColor.closed(color=prefs.color_neutral)

        for col in df.columns:
            align = col_fmts[col].matplot_alignment()
            table_columns[col] = TableColumn(
                header_style=TableColumnStyle(
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    edge_color=header_edge,
                    face_color="black",
                    text_color="white",
                ),
                detail_style=TableColumnStyle(
                    ha=align,
                    va="center",
                    fontfamily="monospace",
                    edge_color=closed_edge,
                    face_color=prefs.color_paper,
                    text_color=prefs.color_neutral,
                ),
                even_row_style=TableColumnStyle(
                    ha=align,
                    va="center",
                    fontfamily="monospace",
                    edge_color=closed_edge,
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
            max_table_height=0.90,
            mid_x=0.5,
            top_y=0.90,
            fontsize=10,
            columns=table_columns,
            cell_edge_linewidth=TableEdgeLinewidth.all_edges(0.4),
            table_edge_linewidth=TableEdgeLinewidth(),
            use_full_axis_width=True,
            header_tpad=20.0,
            header_bpad=0.0,
            detail_tpad=12.0,
            detail_bpad=12.0,
        )
        render_table(pdf_page=pdf_page, table=table, ax=ax, renderer=renderer)

    def _render_error_analysis(
        self,
        pdf_page: PDFDocument.Page,
        ax_worst_errors: Axes,
        config: ModelConfiguration,
        renderer: RendererBase,
    ) -> None:
        """
        Render the error analysis table based on the specific model task type.
        """
        # Retrieve the display limit from central preferences (typically 5 or 10)
        worst_errors_n = prefs.default_worst_errors_n

        # Ensure validation predictions exist before attempting to find outliers
        if config.preds_val is not None:
            # 1. Regression Path: Focus on absolute magnitude of error
            if config.task_type == TaskType.REGRESSION:
                self._plot_worst_residual_errors(
                    pdf_page=pdf_page,
                    ax=ax_worst_errors,
                    y_true=self.summary.data_splits.val_target,
                    y_preds=config.preds_val,
                    n=worst_errors_n,
                    renderer=renderer,
                )

            # 2. Classification Path: Focus on confident incorrect predictions
            else:
                # Probability distribution is required to rank classification "misses"
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
                else:
                    # Fallback for classifiers that do not support probability estimates
                    ax_worst_errors.text(
                        0.5,
                        0.5,
                        "Probability data unavailable\nfor error ranking.",
                        ha="center",
                        va="center",
                        style="italic",
                    )
                    ax_worst_errors.axis("off")

    def _render_model_deep_dive(self, config: ModelConfiguration) -> None:
        """
        Render a comprehensive diagnostic dashboard for a specific model.
        """
        # 1. Page Initialization and Naming
        type_fmt = EnumFormat()
        id_fmt = IntegerFormat()
        page_name = f"{type_fmt.format_value(config.model_type)} [{id_fmt.format_value(config.id)}]"

        pdf_page = self.pdf_doc.create_new_page(
            page_name=page_name,
            print_page_name=False,  # We draw a custom styled header below
        )
        fig = pdf_page.fig

        # Disable constrained_layout explicitly.  Using None here would defer to
        # rcParams (which enable constrained_layout globally in this project),
        # causing Matplotlib to override our manual GridSpec bounds on draw.
        fig.set_layout_engine("none")

        # Access low-level renderer for precise table coordinate calculations
        fig.draw_without_rendering()
        canvas: Any = fig.canvas
        renderer: RendererBase = canvas.get_renderer()

        # 2. Define the Dashboard Grid (2 rows x 2 columns) with explicit,
        # fixed margins so positions are identical across all model pages.
        # Row 0: Config & Features | Row 1: Residuals & Errors
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[0.5, 0.5],
            left=0.09,
            right=0.93,
            bottom=0.09,
            top=0.81,
            hspace=0.30,
            wspace=0.12,
        )
        ax_params = fig.add_subplot(gs[0, 0])
        ax_features = fig.add_subplot(gs[0, 1])
        ax_residuals = fig.add_subplot(gs[1, 0])
        ax_worst_errors = fig.add_subplot(gs[1, 1])

        # 3. Styled Page Header
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

        # 4. Prepare Final Performance Strings
        test_metrics_text = None
        if config.has_test_set_evaluation_scores:
            score_label = EnumFormat().format_value(config.scoring)
            if config.task_type is TaskType.REGRESSION:
                test_metrics_text = (
                    f"Test Set Performance\n"
                    f"{score_label}:  {prefs.score_format.format_value(config.score_test)}\n"
                    f"MAE: {prefs.score_format.format_value(config.mae_test)}"
                )
            else:
                test_metrics_text = (
                    f"Test Set Performance\n"
                    f"{score_label}: {prefs.score_format.format_value(config.score_test)}"
                )

        # 5. Render Top Right: Feature Importance
        self._plot_feature_importance(
            ax=ax_features,
            importance_df=self.importance_dict[config.id],
            model_color=self.summary.solid_color_palette[config.model_type.value],
            test_metrics_text=test_metrics_text,
        )

        # 6. Render Bottom Left: Residual/Diagnostic Analysis
        self._plot_residual_analysis(config, ax_residuals)

        # 7. Render Bottom Right: Worst Errors Table
        self._render_error_analysis(
            pdf_page=pdf_page,
            ax_worst_errors=ax_worst_errors,
            config=config,
            renderer=renderer,
        )

        # 8. Render Top Left: Identity & Hyperparameters
        self._plot_model_hyperparameters(
            pdf_page=pdf_page, config=config, ax_params=ax_params, renderer=renderer
        )

    def _render_detailed_audit_stats(self) -> None:
        """
        Render the detailed audit statistics table page.
        """
        # 1. Initialize Page and Branding
        pdf_page = self.pdf_doc.create_new_page(
            page_name="Detailed Audit Stats", print_page_name=False
        )
        fig = pdf_page.fig
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

        # 2. Configure Column Styles and Alignment
        # Build column headers with task-specific metrics
        col_headers = [
            "Model",
            "Abbr",
            "CV Score (Tuning)",
            "Val Score",
        ]

        # Add task-specific validation metrics based on audit task type
        task_type = self.best_model.model.task_type
        if task_type == TaskType.REGRESSION:
            col_headers.extend(["MAE", "MSE", "R²"])
        else:  # CLASSIFICATION
            col_headers.extend(["Accuracy", "ROC-AUC"])

        # Complete with test and resource columns
        col_headers.extend(["Test Score", "Train Time (s)", "Actual Peak RAM"])

        header_edge = TableEdgeColor(
            left=prefs.color_neutral, right=prefs.color_neutral
        )
        closed_edge = TableEdgeColor.closed(color=prefs.color_neutral)

        def get_col_style(ha: str, is_even: bool = False) -> TableColumnStyle:
            face = prefs.color_light_gray if is_even else "none"
            return TableColumnStyle(
                ha=ha,
                va="center",
                edge_color=closed_edge,
                face_color=face,
                text_color=prefs.color_neutral,
            )

        table_columns: dict[str, TableColumn] = {}
        header_style = TableColumnStyle(
            fontweight="bold",
            ha="center",
            va="center",
            edge_color=header_edge,
            face_color="black",
            text_color="white",
        )

        for col in col_headers:
            # Map column types to appropriate horizontal alignments
            if col == "Model":
                ha = "left"
            elif col == "Abbr":
                ha = "center"
            else:
                ha = "right"  # Numeric performance and resource columns

            table_columns[col] = TableColumn(
                header_style=header_style,
                detail_style=get_col_style(ha),
                even_row_style=get_col_style(ha, is_even=True),
                lpad=12.0,
                rpad=12.0,
            )

        # 3. Data Formatting and Preparation
        table_df = self.summary_df[col_headers].copy()
        str_fmt = StringFormat()
        peak_ram_fmt = ValueDescFormat(
            precision=2,
            description="GB",
            description_leading_space=True,
            description_decorator="",
        )

        # Format strings, scores, and resource metrics using preferences
        for col in ["Model", "Abbr"]:
            table_df[col] = table_df[col].apply(str_fmt.format_value)

        # Score columns: CV, Val, Test, and task-specific metrics (MAE, MSE, R², Accuracy, ROC-AUC)
        score_cols = [
            "CV Score (Tuning)",
            "Val Score",
            "Test Score",
            "MAE",
            "MSE",
            "R²",
            "Accuracy",
            "ROC-AUC",
        ]
        for col in score_cols:
            if col in table_df.columns:
                table_df[col] = table_df[col].apply(prefs.score_format.format_value)

        # "Actual Peak RAM" is already stored in GB on the model config.
        # Use a plain value+suffix formatter to avoid re-scaling.
        table_df["Actual Peak RAM"] = table_df["Actual Peak RAM"].apply(
            peak_ram_fmt.format_value
        )
        table_df["Train Time (s)"] = table_df["Train Time (s)"].apply(
            prefs._train_time_format.format_value
        )

        # 4. Render the Table
        pc = self.pdf_doc.page_configuration
        table_top_y = header_top_y - 0.05

        table = Table(
            data=table_df,
            max_table_height=table_top_y - pc.bottom_margin,
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

        render_table(pdf_page=pdf_page, table=table)

    def _render_recommendation_page(self) -> None:
        """
        Render the strategic recommendation page with final verdict and audit notes.
        """
        # 1. Initialize Page and Full-Page Axis
        pdf_page = self.pdf_doc.create_new_page(
            page_name="Recommendation", print_page_name=False, include_footer=False
        )
        fig = pdf_page.fig
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")

        # 2. Strategic Header
        header_top_y = 0.85
        header_artist = ax.text(
            0.5,
            header_top_y,
            "STRATEGIC RECOMMENDATION",
            fontsize=14,
            ha="center",
            weight="bold",
            color=prefs.color_title,
            linespacing=1.8,
        )

        # Ensure renderer is available for bounding box calculations
        renderer = fig.canvas.get_renderer()  # type: ignore
        if renderer is None:
            fig.draw_without_rendering()
            renderer = fig.canvas.get_renderer()  # type: ignore

        # Calculate table placement based on header height
        header_bbox = get_artist_bbox(
            obj=header_artist, transform_to=fig, renderer=renderer
        )
        header_bot_y = header_top_y - header_bbox.height

        # 3. Comprehensive Metric Table
        # Displays all 9 strategic metrics (Winning Model, Score, Throughput, Resources, etc.)
        table_layout = self._render_metric_table(
            pdf_page=pdf_page,
            top_y=header_bot_y - 0.05,
            df=self.strategic_recommendation_metrics,
        )

        # 4. Dynamic Audit Notes
        table_rect = table_layout.pages[0].rect
        recommendation_text: list[str] = []
        buffer_width = int(table_rect.get_width() * 160)
        model = self.best_model.model
        score_cv = model.score_cv if model.score_cv is not None else 0.0

        # Note: Significant improvement check
        if model.val_score - score_cv > 0.05:
            recommendation_text.append(
                format_text(
                    text="Significant score improvement on full data vs sample.",
                    buffer_width=buffer_width,
                    prefix="-",
                    suffix="",
                    insert_leading_space=True,
                    include_prefix_on_wrapped_lines=False,
                )
            )

        # Note: Memory/Sampling constraint check
        if model.sampling_factor < 0.3:
            recommendation_text.append(
                format_text(
                    text="This model required aggressive sampling to stay within "
                    "hardware memory limits during the tuning phase.",
                    buffer_width=buffer_width,
                    prefix="-",
                    suffix="",
                    insert_leading_space=True,
                    include_prefix_on_wrapped_lines=False,
                )
            )

        # 5. Render Notes Section if applicable
        if recommendation_text:
            y_pos = table_rect.get_y() - 0.1
            ax.text(
                0.5,
                y_pos,
                "NOTES",
                fontsize=12,
                ha="center",
                va="center",
                weight="bold",
                color=prefs.color_title,
                linespacing=1.8,
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

        # 6. Final Methodology Footer
        ax.text(
            0.5,
            0.15,
            "This recommendation is based on a balance of validation accuracy, \n"
            "memory efficiency, and training time performance.",
            fontsize=10,
            style="italic",
            ha="center",
            va="top",
            transform=fig.transFigure,
            linespacing=1.6,
            color=prefs.color_neutral,
            alpha=0.85,
        )
