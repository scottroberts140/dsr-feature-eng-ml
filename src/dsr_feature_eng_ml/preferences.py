"""Preferences singleton for dsr_feature_eng_ml library configuration."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from dsr_files.json_handler import load_json, save_json
from dsr_utils.formatting import (
    CurrencyFormat,
    DataFormat,
    DataScale,
    FloatFormat,
    PercentageFormat,
)

from dsr_feature_eng_ml.enums import ModelType, TaskType
from dsr_feature_eng_ml.utils.memory import validate_n_jobs


@dataclass
class ModelColor:
    """
    Container for primary and secondary model colors.

    Attributes
    ----------
    solid : str
        Hex color code for "Actual" bars, scatter dots, and primary lines.
    light : str
        Hex color code for "Potential" shadow bars or confidence intervals.
    """

    solid: str
    light: str


@dataclass
class Preferences:
    """
    Singleton configuration hub for the dsr-feature_eng_ml library.

    This class manages global heuristics, visualization styles, performance
    thresholds, and reporting formats. It uses a singleton pattern to ensure
    consistent behavior across Auditor and ModelSpecification components.

    Attributes
    ----------
    random_state : Optional[int], default 42
        Seed used for reproducibility in model training and data splitting.
    acceptable_gap : float, default 0.02
        The maximum training-to-test performance gap considered 'Well-Fit'.
    large_gap : float, default 0.05
        The gap threshold above which a model is classified as 'Overfit'.
    n_jobs : int
        The number of parallel processes to use, validated against CPU count.
    report_width : int, default 100
        The character width used for console-based reporting.
    """

    _instance = None
    _initialized = False

    # Default values
    random_state: int | None = 42
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
        """Display attributes for model quality tiers."""

        score_min: float
        text: str
        text_weight: str
        color: str

        @classmethod
        def get_high_quality(cls) -> Preferences.ModelQuality:
            return cls(score_min=95.0, text="HIGH", text_weight="bold", color="#27ae60")

        @classmethod
        def get_average_quality(cls) -> Preferences.ModelQuality:
            return cls(
                score_min=70.0, text="AVERAGE", text_weight="normal", color="#f39c12"
            )

        @classmethod
        def get_low_quality(cls) -> Preferences.ModelQuality:
            return cls(score_min=0.0, text="LOW", text_weight="bold", color="#e74c3c")

    def get_model_quality(self, quality_score: float) -> Preferences.ModelQuality:
        """Map a numeric quality score to a quality tier."""
        if quality_score >= self.model_quality["high"].score_min:
            return self.model_quality["high"]
        elif quality_score < self.model_quality["average"].score_min:
            return self.model_quality["low"]
        else:
            return self.model_quality["average"]

    @dataclass
    class ModelRecommendation:
        """Recommendation text and color based on audit outcomes."""

        action: str
        color: str

        @classmethod
        def get_proceed_to_deployment(cls) -> Preferences.ModelRecommendation:
            return cls(action="PROCEED TO DEPLOYMENT", color="#27ae60")

        @classmethod
        def get_clean_data_and_retrain(cls) -> Preferences.ModelRecommendation:
            return cls(action="CLEAN DATA & RETRAIN", color="#f39c12")

        @classmethod
        def get_proceed_efficiency_win(cls) -> Preferences.ModelRecommendation:
            return cls(action="PROCEED (EFFICIENCY WIN)", color="#2980b9")

        @classmethod
        def get_review_model_architecture(cls) -> Preferences.ModelRecommendation:
            return cls(action="REVIEW MODEL ARCHITECTURE", color="#e74c3c")

    def get_model_recommendation(
        self,
        is_accurate: bool,
        is_stable: bool,
        is_efficient: bool,
        is_acceptable: bool,
    ) -> Preferences.ModelRecommendation:
        """Select a recommendation based on accuracy, stability, and efficiency."""
        if is_accurate and is_stable:
            return self.model_recommendation["proceed_to_deployment"]

        if not is_stable:
            return self.model_recommendation["clean_data_and_retrain"]

        if is_acceptable and is_stable and is_efficient:
            return self.model_recommendation["proceed_efficiency_win"]

        return self.model_recommendation["review_model_architecture"]

    # Global Color Constants
    color_success = "#27ae60"
    color_warning = "#f39c12"
    color_danger = "#e74c3c"
    color_acceptable = "#2980B9"
    color_neutral = "#34495e"
    color_classic_blue = "#2980b9"
    color_title = "#212121"
    color_paper = "#fdfefe"
    color_light_gray = "#d5dbdb"
    color_faint_gray_blue = "#f8f9f9"
    tight_layout_rect = (0, 0.05, 1, 0.92)

    def __new__(cls, *args, **kwargs):
        """Implement Singleton pattern."""
        if not cls._instance:
            cls._instance = super(Preferences, cls).__new__(cls)
        return cls._instance

    def __post_init__(self):
        """Initialize backing variables, formats, and styles once."""
        if self._initialized:
            return

        self.n_jobs = -1
        self.cv_verbose = 0
        self.fit_verbose = 0
        self._currency_format = CurrencyFormat()
        self._score_format = FloatFormat(precision=4)
        self._gb_format = DataFormat(data_scale=DataScale.GB)
        self._train_time_format = FloatFormat(precision=2)
        self._drift_format = PercentageFormat(precision=3)

        self.apply_style()

        self.model_quality = {
            "high": self.ModelQuality.get_high_quality(),
            "average": self.ModelQuality.get_average_quality(),
            "low": self.ModelQuality.get_low_quality(),
        }
        self.model_recommendation = {
            "proceed_to_deployment": self.ModelRecommendation.get_proceed_to_deployment(),
            "clean_data_and_retrain": self.ModelRecommendation.get_clean_data_and_retrain(),
            "proceed_efficiency_win": self.ModelRecommendation.get_proceed_efficiency_win(),
            "review_model_architecture": self.ModelRecommendation.get_review_model_architecture(),
        }

        self._model_colors: dict[str, ModelColor] | None = None
        self._initialized = True

    @property
    def cv_verbose(self) -> int:
        """Verbosity level for cross-validation and tuning (0-3)."""
        return self._cv_verbose

    @cv_verbose.setter
    def cv_verbose(self, value: int):
        self._cv_verbose = max(0, min(3, value))

    @property
    def fit_verbose(self) -> int:
        """Verbosity level for estimator fit calls (non-negative integer)."""
        return self._fit_verbose

    @fit_verbose.setter
    def fit_verbose(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"fit_verbose must be a non-negative integer, got {value!r}"
            )
        self._fit_verbose = value

    @property
    def n_jobs(self) -> int:
        """Validated parallel job count."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int):
        self._n_jobs = validate_n_jobs(value)

    @property
    def currency_format(self) -> CurrencyFormat:
        """Formatter for currency values."""
        return self._currency_format

    @property
    def score_format(self) -> FloatFormat:
        """Formatter for score values."""
        return self._score_format

    @property
    def gb_format(self) -> DataFormat:
        """Formatter for memory sizes in GB."""
        return self._gb_format

    @property
    def train_time_format(self) -> FloatFormat:
        """Formatter for training durations."""
        return self._train_time_format

    @property
    def drift_format(self) -> PercentageFormat:
        """Formatter for drift percentages."""
        return self._drift_format

    def _build_model_colors(self) -> dict[str, ModelColor]:
        """
        Map ModelType values to specific color pairs.
        """
        return {
            # --- Regression Models ---
            ModelType.DECISION_TREE_REGRESSOR.value: ModelColor(
                "#008080", "#4db6ac"
            ),  # Teal
            ModelType.RANDOM_FOREST_REGRESSOR.value: ModelColor(
                "#2c3e50", "#5d6d7e"
            ),  # Slate / Navy
            ModelType.RIDGE.value: ModelColor("#2980b9", "#3498db"),  # River Blue
            ModelType.LINEAR_REGRESSION.value: ModelColor(
                "#8e44ad", "#9b59b6"
            ),  # Amethyst
            ModelType.LASSO.value: ModelColor("#d35400", "#e67e22"),  # Pumpkin
            ModelType.ELASTIC_NET.value: ModelColor(
                "#7f8c8d", "#95a5a6"
            ),  # Asbestos Gray
            # --- Classification Models ---
            ModelType.DECISION_TREE_CLASSIFIER.value: ModelColor(
                "#d4ac0d", "#f1c40f"
            ),  # Flat Gold
            ModelType.RANDOM_FOREST_CLASSIFIER.value: ModelColor(
                "#a04000", "#d35400"
            ),  # Burnt Orange
            ModelType.LOGISTIC_REGRESSION.value: ModelColor(
                "#6c3483", "#a569bd"
            ),  # Plum
            # --- NEW: Added Model Support ---
            ModelType.XGB_CLASSIFIER.value: ModelColor(
                "#1e8449", "#2ecc71"
            ),  # Emerald Green
            ModelType.K_NEIGHBORS_CLASSIFIER.value: ModelColor(
                "#1a5276", "#5499c7"
            ),  # Ocean Blue
            ModelType.RIDGE_CLASSIFIER.value: ModelColor(
                "#1b2631", "#283747"
            ),  # Charcoal Blue
            ModelType.LINEAR_SVC.value: ModelColor(
                "#922b21", "#c0392b"
            ),  # Ruby / Crimson
            # --- Fallback ---
            ModelType.UNKNOWN.value: ModelColor("#34495e", "#7f8c8d"),  # Wet Asphalt
        }

    @property
    def model_colors(self) -> dict[str, ModelColor]:
        """Lazy-loaded model color dictionary."""
        if self._model_colors is None:
            self._model_colors = self._build_model_colors()
        return self._model_colors

    def get_color(self, model_name: str, shadow: bool = False) -> str:
        """Return hex color for a specific model."""
        color_obj = self.model_colors.get(model_name, self.get_default_color())
        return color_obj.light if shadow else color_obj.solid

    def get_solid_palette(self):
        """Return a palette of solid colors keyed by model name."""
        return {name: color.solid for name, color in self.model_colors.items()}

    def get_light_palette(self):
        """Return a palette of light colors keyed by model name."""
        return {name: color.light for name, color in self.model_colors.items()}

    def get_default_color(self) -> ModelColor:
        """Fallback color pair."""
        return ModelColor("#333333", "#999999")

    def update(self, **kwargs: Any) -> None:
        """Update multiple preferences with attribute validation."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{key}' is not a valid preference.")

    def save_to_json(self, path: str | Path) -> None:
        """
        Persist preferences to a JSON file using the library's safe handler.

        Parameters
        ----------
        path : str or Path
            The file system path (e.g., 'config/prefs.json') where the
            configuration should be saved.
        """
        path_obj = Path(path)

        # 1. Convert to dict and filter out private state (e.g., _initialized)
        # Note: to_JSON_safe handles the nested dataclasses/enums automatically
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}

        # 2. Use your robust handler to save safely
        save_json(
            data=data,
            output_dir=path_obj.parent,
            filename=path_obj.stem,  # 'prefs.json' -> 'prefs'
            indent=4,
        )

    def load_from_json(self, path: str | Path) -> None:
        """
        Load preferences from a JSON file using the library's safe handler.

        Parameters
        ----------
        path : str or Path
            Path to the target JSON configuration file.

        Raises
        ------
        FileNotFoundError
            If the specified filepath does not exist on disk.
        ValueError
            If the file extension is not '.json'.
        """
        # Use the robust loader from dsr-files
        data, _ = load_json(path)

        # Update the singleton instance with the loaded data
        self.update(**data)

    def reset_defaults(self) -> None:
        """
        Restore all configuration attributes to their original hardcoded values.

        This handles both standard default values and default_factory functions
        to ensure internal dataclass sentinels (MISSING) are not assigned.
        """
        self._initialized = False
        # Access the field definitions from the dataclass
        class_fields = self.__class__.__dataclass_fields__

        for field_name, field_def in class_fields.items():
            # Skip internal state trackers
            if field_name.startswith("_"):
                continue

            # If the field has a simple default value (int, float, str, etc.)
            if field_def.default is not MISSING:
                setattr(self, field_name, field_def.default)

            # If the field uses a factory (like dict or list)
            elif field_def.default_factory is not MISSING:
                setattr(self, field_name, field_def.default_factory())

        # Re-run initialization to restore formatters and styles
        self.__post_init__()

    def get_penalty_multiplier_for_task_type(self, task_type: TaskType) -> int:
        """Return task-specific data quality penalty."""
        if task_type == TaskType.UNKNOWN:
            raise ValueError(
                "Cannot determine penalty multiplier for TaskType.UNKNOWN."
            )
        if task_type == TaskType.CLASSIFICATION:
            return self.classification_data_quality_penalty_multiplier
        return self.regression_data_quality_penalty_multiplier

    def apply_style(self) -> None:
        """Configure global Matplotlib rcParams for a consistent 'Audit' look."""
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

    def get_hyperparameter_display_name(self, raw_name: str) -> str:
        """Format hyperparameter keys for reporting."""
        return self.hp_short_names.get(raw_name, raw_name.replace("_", " ").title())


def __getattr__(name: str) -> Any:
    """Provide a lazy compatibility export for the singleton instance."""
    if name == "prefs":
        from dsr_feature_eng_ml.prefs_instance import prefs

        return prefs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


prefs: Preferences


__all__ = ["Preferences", "ModelColor", "prefs"]
