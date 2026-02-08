"""Preferences singleton for dsr_feature_eng_ml library configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from dsr_feature_eng_ml.enums import TaskType
from dsr_feature_eng_ml.utils.memory import validate_n_jobs
from dsr_utils.formatting import (
    DataScale,
    CurrencyFormat,
    PercentageFormat,
    FloatFormat,
    DataFormat,
)


@dataclass
class ModelColor:
    solid: str  # For "Actual" bars and Scatter dots
    light: str  # For "Potential" shadow bars


@dataclass
class Preferences:
    """Singleton container for library preferences."""

    _instance = None

    # Default values
    random_state: int = 42
    default_float_type: str = "float32"
    report_width: int = 100
    viable_f1_gap: float = 0.01
    acceptable_gap: float = 0.02
    large_gap: float = 0.05
    min_target_tuning_rows: int = 100_000
    default_worst_errors_n: int = 5
    default_plot_sample_size: int = 100_000
    classification_data_quality_penalty_multiplier: int = 600
    regression_data_quality_penalty_multiplier: int = 500
    default_efficiency_threshold: int = 50_000
    data_leakage_threshold: float = 0.05
    anomaly_threshold: float = 99.0
    anomaly_risk_concentration_threshold: float = 0.6
    model_accuracy_limit: float = 0.85
    model_acceptable_limit: float = 0.70
    model_stability_limit: float = 0.85
    model_efficiency_threshold: int = 50_000
    drift_threshold: float = 0.05
    hp_short_names: dict[str, str] = field(
        default_factory=lambda: {
            "n_estimators": "Trees",
            "max_depth": "Max Depth",
            "min_samples_split": "Min Split",
            "min_samples_leaf": "Min Leaf",
            "learning_rate": "LR",
            "max_features": "Max Feat.",
            "fit_intercept": "Intercept",
            "max_iter": "Max Iter",
            "random_state": "Seed",
            "optimization_strategy": "Opt. Strategy",
            "task_type": "Task",
            "scoring": "Metric",
            "min_weight_fraction_leaf": "Min Wt Leaf",
            "max_leaf_nodes": "Max Nodes",
            "min_impurity_decrease": "Min Impurity",
            "class_weight": "Class Wts",
        }
    )

    @dataclass
    class ModelQuality:
        score_min: float
        text: str
        text_weight: str
        color: str

        @classmethod
        def get_high_quality(cls) -> Preferences.ModelQuality:
            return Preferences.ModelQuality(
                score_min=95.0, text="HIGH", text_weight="bold", color="#27ae60"
            )

        @classmethod
        def get_average_quality(cls) -> Preferences.ModelQuality:
            return Preferences.ModelQuality(
                score_min=70.0, text="AVERAGE", text_weight="normal", color="#f39c12"
            )

        @classmethod
        def get_low_quality(cls) -> Preferences.ModelQuality:
            return Preferences.ModelQuality(
                score_min=0.0, text="LOW", text_weight="bold", color="#e74c3c"
            )

    def get_model_quality(self, quality_score: float) -> Preferences.ModelQuality:
        if quality_score >= self.model_quality["high"].score_min:
            return self.model_quality["high"]
        elif quality_score < self.model_quality["average"].score_min:
            return self.model_quality["low"]
        else:
            return self.model_quality["average"]

    @dataclass
    class ModelRecommendation:
        action: str
        color: str

        @classmethod
        def get_proceed_to_deployment(cls) -> Preferences.ModelRecommendation:
            return Preferences.ModelRecommendation(
                action="PROCEED TO DEPLOYMENT", color="#27ae60"
            )

        @classmethod
        def get_clean_data_and_retrain(cls) -> Preferences.ModelRecommendation:
            return Preferences.ModelRecommendation(
                action="CLEAN DATA & RETRAIN", color="#f39c12"
            )

        @classmethod
        def get_proceed_efficiency_win(cls) -> Preferences.ModelRecommendation:
            return Preferences.ModelRecommendation(
                action="PROCEED (EFFICIENCY WIN)", color="#2980b9"
            )

        @classmethod
        def get_review_model_architecture(cls) -> Preferences.ModelRecommendation:
            return Preferences.ModelRecommendation(
                action="REVIEW MODEL ARCHITECTURE", color="#e74c3c"
            )

    def get_model_recommendation(
        self,
        is_accurate: bool,
        is_stable: bool,
        is_efficient: bool,
        is_acceptable: bool,
    ):
        if is_accurate and is_stable:
            return self.model_recommendation["proceed_to_deployment"]

        if not is_stable:
            return self.model_recommendation["clean_data_and_retrain"]

        if is_acceptable and is_stable and is_efficient:
            return self.model_recommendation["proceed_efficiency_win"]

        return self.model_recommendation["review_model_architecture"]

    color_success = "#27ae60"  # True Green (Safe, High Quality, Profit)
    color_warning = "#f39c12"  # Amber
    color_danger = "#e74c3c"  # True Red (Drift, Low Quality, Loss)
    color_acceptable = "#2980B9"  # Blue
    color_neutral = "#34495e"  # Audit Blue (Legend, Info)
    color_classic_blue = "#2980b9"  # Classic Blue (for contrast with Audit Blue)
    color_title = "#212121"  # Dark Charcoal
    color_paper = "#fdfefe"  # Clean Paper White
    color_light_gray = "#d5dbdb"
    color_faint_gray_blue = "#f8f9f9"
    tight_layout_rect = (0, 0.05, 1, 0.92)

    @property
    def cv_verbose(self) -> int:
        return self._cv_verbose

    @cv_verbose.setter
    def cv_verbose(self, value: int):
        if value < 0:
            self._cv_verbose = 0
        elif value > 3:
            self._cv_verbose = 3
        else:
            self._cv_verbose = value

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int):
        self._n_jobs = validate_n_jobs(value)

    @property
    def currency_format(self) -> CurrencyFormat:
        return self._currency_format

    @property
    def score_format(self) -> FloatFormat:
        return self._score_format

    @property
    def gb_format(self) -> DataFormat:
        return self._gb_format

    @property
    def train_time_format(self) -> FloatFormat:
        return self._train_time_format

    @property
    def drift_format(self) -> PercentageFormat:
        return self._drift_format

    def _build_model_colors(self) -> dict[str, ModelColor]:
        from dsr_feature_eng_ml.enums import ModelType

        return {
            # --- Regression ---
            ModelType.DECISION_TREE_REGRESSOR.value: ModelColor(
                solid="#008080", light="#4db6ac"
            ),  # Teal (Moved from Green)
            ModelType.RANDOM_FOREST_REGRESSOR.value: ModelColor(
                solid="#2c3e50", light="#5d6d7e"
            ),  # Midnight Blue (Distinct from Ridge)
            ModelType.RIDGE.value: ModelColor(solid="#2980b9", light="#3498db"),  # Blue
            ModelType.LINEAR_REGRESSION.value: ModelColor(
                solid="#8e44ad", light="#9b59b6"
            ),  # Purple
            ModelType.LASSO.value: ModelColor(
                solid="#d35400", light="#e67e22"
            ),  # Orange
            ModelType.ELASTIC_NET.value: ModelColor(
                solid="#7f8c8d", light="#95a5a6"
            ),  # Gray
            # --- Classification ---
            ModelType.DECISION_TREE_CLASSIFIER.value: ModelColor(
                solid="#d4ac0d", light="#f1c40f"
            ),  # Ochre/Gold (Moved from Teal)
            ModelType.RANDOM_FOREST_CLASSIFIER.value: ModelColor(
                solid="#a04000", light="#d35400"
            ),  # Sienna/Brown (Moved from Forest)
            ModelType.LOGISTIC_REGRESSION.value: ModelColor(
                solid="#6c3483", light="#a569bd"
            ),  # Plum
            # --- Unknown ---
            ModelType.UNKNOWN.value: ModelColor(
                solid="#34495e", light="#7f8c8d"
            ),  # Deep Slate / Steel
        }

    @property
    def model_colors(self) -> dict[str, ModelColor]:
        if not hasattr(self, "_model_colors"):
            self._model_colors = self._build_model_colors()
        return self._model_colors

    def get_color(self, model_name: str, shadow: bool = False) -> str:
        color_obj = self.model_colors.get(model_name, prefs.get_default_color())
        return color_obj.light if shadow else color_obj.solid

    def get_solid_palette(self):
        return {name: color.solid for name, color in self.model_colors.items()}

    def get_light_palette(self):
        return {name: color.light for name, color in self.model_colors.items()}

    def get_default_color(self):
        return ModelColor("#333333", "#999999")

    def __post_init__(self):
        # Initialize the backing variables for the properties
        # This ensures the defaults are applied on the very first import
        self.n_jobs = -1  # Triggers the setter to calculate CPU count
        self.cv_verbose = 0
        self._currency_format = CurrencyFormat()
        self._score_format = FloatFormat(precision=4)
        self._gb_format = DataFormat(data_scale=DataScale.GB)
        self._train_time_format = FloatFormat(precision=2)
        self._drift_format = PercentageFormat(precision=3)
        self.apply_style()
        self.model_quality = {
            "high": Preferences.ModelQuality.get_high_quality(),
            "average": Preferences.ModelQuality.get_average_quality(),
            "low": Preferences.ModelQuality.get_low_quality(),
        }
        self.model_recommendation = {
            "proceed_to_deployment": Preferences.ModelRecommendation.get_proceed_to_deployment(),
            "clean_data_and_retrain": Preferences.ModelRecommendation.get_clean_data_and_retrain(),
            "proceed_efficiency_win": Preferences.ModelRecommendation.get_proceed_efficiency_win(),
            "review_model_architecture": Preferences.ModelRecommendation.get_review_model_architecture(),
        }

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Preferences, cls).__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Provides a cleanly formatted table of current library preferences."""
        header = f"{'Preference':<25} | {'Value':<20}"
        separator = "-" * 48

        lines = [header, separator]
        for field in fields(self):
            # Skip the internal instance tracker
            if field.name.startswith("_"):
                continue
            val = getattr(self, field.name)
            lines.append(f"{field.name:<25} | {str(val):<20}")

        return "\n" + "\n".join(lines) + "\n"

    def update(self, **kwargs: Any) -> None:
        """Update multiple preferences at once."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{key}' is not a valid preference.")

    def save_to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    def load_from_json(self, path: str | Path) -> None:
        with open(path, "r") as f:
            data = json.load(f)
            self.update(**data)

    def reset_defaults(self) -> None:
        """Restores preferences to the library original hardcoded defaults."""
        # This creates a fresh, temporary instance to grab the default values
        defaults = self.__class__.__dataclass_fields__
        for field_name, field_def in defaults.items():
            if not field_name.startswith("_"):
                setattr(self, field_name, field_def.default)
        self.__post_init__()  # Re-trigger the property setters

    def get_penalty_multiplier_for_task_type(self, task_type: TaskType) -> int:
        return (
            self.classification_data_quality_penalty_multiplier
            if task_type == TaskType.CLASSIFICATION
            else self.regression_data_quality_penalty_multiplier
        )

    def apply_style(self):
        """Applies a consistent 'Audit Product' look to all Matplotlib/Seaborn plots."""
        plt.rcParams.update(
            {
                # Font & Text
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "text.color": self.color_neutral,  # Default text color
                # Axes
                "axes.titlecolor": self.color_title,  # Titles stay
                "axes.titlesize": 14,
                "axes.titleweight": "bold",
                "axes.titlepad": 25,  # Space between Title and Chart
                "axes.labelpad": 10,  # Space between Axis Label and Ticks
                "axes.labelcolor": self.color_neutral,  # Axis labels use Neutral Blue
                "axes.labelsize": 10,
                "axes.edgecolor": self.color_neutral,  # Frame of the chart
                "axes.linewidth": 0.8,
                "axes.spines.top": False,  # Modern 'Clean' look
                "axes.spines.right": False,
                # Grid Styling
                "axes.grid": True,  # Turn grid on by default
                "axes.grid.axis": "both",  # Show both X and Y
                "grid.color": self.color_neutral,  # Use our Audit Blue
                "grid.linestyle": "--",  # Dashed is less 'heavy' than solid
                "grid.linewidth": 0.5,  # Keep it very thin
                "grid.alpha": 0.15,  # High transparency is key
                # Ensure the grid stays BEHIND the bars and dots
                "axes.axisbelow": True,
                # Ticks
                # X-Tick Styling
                "xtick.color": self.color_neutral,  # Color of the tick AND the label
                "xtick.labelsize": 9,
                "xtick.major.size": 4,  # Length of the little tick line
                "xtick.major.width": 0.8,
                # Y-Tick Styling
                "ytick.color": self.color_neutral,
                "ytick.labelsize": 9,
                "ytick.major.size": 4,
                "ytick.major.width": 0.8,
                # Legend
                "legend.fontsize": 9,
                "legend.title_fontsize": 10,
                "legend.frameon": True,
                "legend.edgecolor": self.color_neutral,
                "legend.framealpha": 0.9,  # Slightly transparent Paper White
                "legend.edgecolor": self.color_neutral,  # Audit Blue border
                "legend.fancybox": True,  # Rounded corners
                "legend.borderpad": 0.8,  # The 'breathing room' inside the box
                "legend.labelspacing": 0.6,  # Vertical space between entries
                "legend.handletextpad": 0.5,  # Space between icon and text
                # Figure
                "figure.titlesize": 16,
                "figure.titleweight": "bold",
                "figure.dpi": 300,  # High quality for PDF
                "figure.constrained_layout.use": True,  # Automatically prevents overlap
                "figure.constrained_layout.h_pad": 0.05,  # Spacing between subplots
                "figure.constrained_layout.w_pad": 0.05,
                "figure.subplot.top": 0.88,  # Leaves room for our Page Header helper
                "figure.constrained_layout.wspace": 0.05,
                "figure.constrained_layout.hspace": 0.05,
            }
        )

    def get_hyperparmeter_display_name(self, raw_name: str) -> str:
        return self.hp_short_names.get(raw_name, raw_name.replace("_", " ").title())


# Global singleton instance
prefs = Preferences()


__all__ = [
    "Preferences",
    "prefs",
    "ModelColor",
]
