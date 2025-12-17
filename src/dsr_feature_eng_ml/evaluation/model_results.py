from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Optional, cast, Mapping, Type, Any, Final
from dsr_utils.formatting import format_text, TextAlignment
from dsr_feature_eng_ml.enums import ModelType, ModelBalancing, ModelConfigurationSortOrder
from dsr_feature_eng_ml.evaluation.model_configuration import ModelConfiguration
from dsr_feature_eng_ml.constants import DEFAULT_LARGE_GAP, DEFAULT_ACCEPTABLE_GAP, REPORT_WIDTH, F1_FORMAT
from dsr_feature_eng_ml.models.decision_tree import DecisionTree
from dsr_feature_eng_ml.models.random_forest import RandomForest
from dsr_feature_eng_ml.models.logistic_regression import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


class ModelResults:
    """Manages and analyzes results from multiple model configurations.

    Tracks all model configurations tested, identifies best performers, and provides
    utilities for filtering, sorting, and evaluating models on test sets.

    Attributes:
        model_configurations (list[ModelConfiguration]): All tested configurations.
        best_model_configuration_last_results (ModelConfiguration): Best from last batch.
        best_model_configuration_overall (ModelConfiguration): Best overall configuration.
        results_text (str): Comprehensive detailed log of all model evaluations including
            hyperparameter search results, confusion matrices, feature importance rankings,
            and performance metrics for every model configuration tested. Updated by
            process_results() and model evaluation methods.
        summary_text (str): Condensed summary of key evaluation milestones and final results,
            containing only the most important information such as best models per balancing
            strategy and overall winners. Updated selectively by model_evaluation_footer()
            when add_to_summary_text=True.
        final_summary_text (str): Summary containing only the best model for an evaluation
            phase. Updated by BestModelResults.compare_model_results().

    Example:
        >>> results = ModelResults()
        >>> results.process_results(
        ...     results=dtree_results,
        ...     report_text=report_text,
        ...     step_title='Decision Tree - Imbalanced'
        ... )
        >>> results.evaluate_best_model_test_set()
    """

    def __init__(
            self,
    ):
        self.model_configurations: list[ModelConfiguration] = []
        self.best_model_configuration_last_results = ModelConfiguration.empty()
        self.best_model_configuration_overall = ModelConfiguration.empty()
        self.results_text: str = ''
        self.summary_text: str = ''
        self.final_summary_text: str = ''

    def clear_results_text(
            self
    ):
        """Resets the results_text and summary_text properties to empty strings.
        """
        self.results_text = ''
        self.summary_text = ''
        self.final_summary_text = ''

    def model_evaluation_title(
            self,
            report_title: str,
            border_char: str,
            acceptable_gap: float = DEFAULT_ACCEPTABLE_GAP,
            large_gap: float = DEFAULT_LARGE_GAP,
    ):
        """Generate and append a formatted title for a model evaluation step.

        Creates a centered title with border decoration to mark the beginning of
        a new evaluation section in the results text.

        Args:
            report_title (str): The title for the evaluation step.
            border_char (str): Character to use for borders.
            acceptable_gap (float): Threshold for acceptable generalization gap (default: DEFAULT_ACCEPTABLE_GAP).
            large_gap (float): Threshold for large gap indicating overfitting (default: DEFAULT_LARGE_GAP).
        """
        title_width = REPORT_WIDTH
        border = border_char * title_width

        title_text = format_text(
            text=report_title,
            buffer_width=title_width,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Center
        )

        blank_line = format_text(
            text=' ',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            fill_buffer=True
        )

        model_generalization_text = format_text(
            text='Model Generalization',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Left,
            insert_leading_space=True
        )

        acceptable_gap_text = format_text(
            text=f'    Acceptable gap    {acceptable_gap}',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Left,
            insert_leading_space=True
        )

        large_gap_text = format_text(
            text=f'    Large gap         {large_gap}',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Left,
            insert_leading_space=True
        )

        full_title_text = f'''{border}
{title_text}
{blank_line}
{model_generalization_text}
{acceptable_gap_text}
{large_gap_text}
{border}
'''
        self.results_text += full_title_text
        self.summary_text += full_title_text
        self.final_summary_text += full_title_text

    def model_evaluation_header(
            self,
            step_description: str,
            border_char: str = '*'
    ):
        """Generate and append a formatted header for a model evaluation step.

        Creates a centered header with border decoration to mark the beginning of
        a new evaluation section in the results text.

        Args:
            step_description (str): Description of the evaluation step.
            border_char (str): Character to use for borders (default: '*').
        """
        header_width = REPORT_WIDTH
        border = border_char * header_width

        header_text = format_text(
            text=step_description,
            buffer_width=header_width,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Center
        )

        self.results_text += f'''
{border}
{header_text}
{border}
'''

    def model_evaluation_footer(
            self,
            step_description: str,
            best_f1: float,
            best_f1_overall: float,
            best_dtree_f1: Optional[float],
            best_rf_f1: Optional[float],
            best_lr_f1: Optional[float],
            best_dtree_f1_overall: float,
            best_rf_f1_overall: float,
            best_lr_f1_overall: float,
            include_dtree: bool,
            include_rf: bool,
            include_lr: bool,
            viable_f1_gap: float,
            border_char: str = '*',
            add_to_summary_text: bool = True
    ):
        """Generate and append a formatted footer with evaluation summary.

        Creates a comprehensive summary footer showing best F1 scores for this step
        and overall, model inclusion status, viability gap, and the best overall
        model configuration.

        Args:
            step_description (str): Description of the evaluation step.
            best_f1 (float): Best F1 score from this evaluation step.
            best_f1_overall (float): Best F1 score overall across all evaluations.
            best_dtree_f1 (Optional[float]): Best DT F1 from this step (None if not run).
            best_rf_f1 (Optional[float]): Best RF F1 from this step (None if not run).
            best_lr_f1 (Optional[float]): Best LR F1 from this step (None if not run).
            best_dtree_f1_overall (float): Best Decision Tree F1 overall.
            best_rf_f1_overall (float): Best Random Forest F1 overall.
            best_lr_f1_overall (float): Best Logistic Regression F1 overall.
            include_dtree (bool): Whether Decision Tree is still viable.
            include_rf (bool): Whether Random Forest is still viable.
            include_lr (bool): Whether Logistic Regression is still viable.
            viable_f1_gap (float): F1 gap threshold for model viability.
            border_char (str): Character to use for borders (default: '*').
            add_to_summary_text (bool): Whether to also append to summary_text.
        """
        border = border_char * REPORT_WIDTH

        divider_line = format_text(
            text='-',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            fill_buffer=True
        )

        blank_summary_line = format_text(
            text=' ',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            fill_buffer=True
        )

        step_description_header = format_text(
            text=step_description,
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Center
        )

        this_step_header = format_text(
            text='This Step',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Center
        )

        overall_header = format_text(
            text='Overall',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Center
        )

        models_failed_viability = (
            not include_dtree or not include_rf or not include_lr)
        model_inclusion = ''

        if models_failed_viability:
            model_inclusion += format_text(
                text='The following models failed the viability gap test.',
                buffer_width=REPORT_WIDTH,
                prefix=border_char,
                suffix=border_char,
                insert_leading_space=True
            )

            model_inclusion += format_text(
                text='These models will not be included in further evaluations.',
                buffer_width=REPORT_WIDTH,
                prefix=border_char,
                suffix=border_char,
                include_start_lf=True,
                insert_leading_space=True
            )

            if not include_dtree:
                model_inclusion += format_text(
                    text='    Decision Tree',
                    buffer_width=REPORT_WIDTH,
                    prefix=border_char,
                    suffix=border_char,
                    include_start_lf=True,
                    insert_leading_space=True
                )

            if not include_rf:
                model_inclusion += format_text(
                    text='    Random Forest',
                    buffer_width=REPORT_WIDTH,
                    prefix=border_char,
                    suffix=border_char,
                    include_start_lf=True,
                    insert_leading_space=True
                )

            if not include_lr:
                model_inclusion += format_text(
                    text='    Logistic Regression',
                    buffer_width=REPORT_WIDTH,
                    prefix=border_char,
                    suffix=border_char,
                    include_start_lf=True,
                    insert_leading_space=True
                )
        else:
            model_inclusion += format_text(
                text='All models passed the viability gap test.',
                buffer_width=REPORT_WIDTH,
                prefix=border_char,
                suffix=border_char,
                insert_leading_space=True
            )

        this_step_f1_description = ''

        if best_dtree_f1 is not None:
            this_step_f1_description += format_text(
                text=f'Best Decision Tree F1:       {best_dtree_f1:{F1_FORMAT}}',
                buffer_width=REPORT_WIDTH,
                prefix=border_char,
                suffix=border_char,
                insert_leading_space=True
            )

        if best_rf_f1 is not None:
            this_step_f1_description += format_text(
                text=f'Best Random Forest F1:       {best_rf_f1:{F1_FORMAT}}',
                buffer_width=REPORT_WIDTH,
                prefix=border_char,
                suffix=border_char,
                include_start_lf=(len(this_step_f1_description) > 0),
                insert_leading_space=True
            )

        if best_lr_f1 is not None:
            this_step_f1_description += format_text(
                text=f'Best Logistic Regression F1: {best_lr_f1:{F1_FORMAT}}',
                buffer_width=REPORT_WIDTH,
                prefix=border_char,
                suffix=border_char,
                include_start_lf=(len(this_step_f1_description) > 0),
                insert_leading_space=True
            )

        best_model_overall_description = ''
        best_model_overall_info = self.best_model_configuration_overall.info()
        info_lines = best_model_overall_info.splitlines()

        for line in info_lines:
            best_model_overall_description += format_text(
                text=line,
                buffer_width=REPORT_WIDTH,
                prefix=border_char,
                suffix=border_char,
                include_start_lf=True,
                insert_leading_space=True
            )

        best_f1_score_text = format_text(
            text=f'Highest F1 score:            {best_f1:{F1_FORMAT}}',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            insert_leading_space=True
        )

        best_dtree_f1_score_text = format_text(
            text=f'Best Decision Tree F1:       {best_dtree_f1_overall:{F1_FORMAT}}',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            insert_leading_space=True
        )

        best_rf_f1_score_text = format_text(
            text=f'Best Random Forest F1:       {best_rf_f1_overall:{F1_FORMAT}}',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            insert_leading_space=True
        )

        best_lr_f1_score_text = format_text(
            text=f'Best Logistic Regression F1: {best_lr_f1_overall:{F1_FORMAT}}',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            insert_leading_space=True
        )

        best_f1_score_overall_text = format_text(
            text=f'Highest F1 score:            {best_f1_overall:{F1_FORMAT}}',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            insert_leading_space=True
        )

        viability_gap_text = format_text(
            text=f'Viability gap:               {viable_f1_gap:{F1_FORMAT}}',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            insert_leading_space=True
        )

        footer_text = f'''
{border}
{step_description_header}
{divider_line}
{this_step_header}
{this_step_f1_description}
{blank_summary_line}
{best_f1_score_text}
{divider_line}
{overall_header}
{best_dtree_f1_score_text}
{best_rf_f1_score_text}
{best_lr_f1_score_text}
{blank_summary_line}
{best_f1_score_overall_text}
{viability_gap_text}
{divider_line}
{model_inclusion}
{divider_line}
{best_model_overall_description}
{border}
'''

        self.results_text += footer_text

        if add_to_summary_text:
            self.summary_text += footer_text

    def update_final_summary_text(
            self,
            best_model_configuration: ModelConfiguration,
    ):
        """Append final evaluation summary to final_summary_text attribute.

        Generates a formatted summary section displaying the best overall model configuration
        and viability triage results, identifying which model type (Decision Tree, Random Forest,
        or Logistic Regression) emerged as the champion and which types were discarded due to
        poor performance across all balancing strategies.

        Args:
            best_model_configuration (ModelConfiguration): The best performing model
                configuration identified across all evaluations.

        Note:
            This method appends directly to the final_summary_text attribute. The summary
            includes the complete model configuration details and a viability triage section
            showing the champion model type and failed model types.
        """
        border_char = '*'
        border = border_char * REPORT_WIDTH

        divider_line = format_text(
            text='-',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            fill_buffer=True
        )

        best_phase_model_header = format_text(
            text='Best Model',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Center,
        )

        blank_summary_line = format_text(
            text=' ',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            fill_buffer=True
        )

        best_model_overall_description = ''
        best_model_overall_info = best_model_configuration.info()
        info_lines = best_model_overall_info.splitlines()
        include_start_lf = False

        for line in info_lines:
            best_model_overall_description += format_text(
                text=line,
                buffer_width=REPORT_WIDTH,
                prefix=border_char,
                suffix=border_char,
                include_start_lf=include_start_lf,
                insert_leading_space=True
            )

            include_start_lf = True

        viability_triage_header = format_text(
            text='Viability Triage and Next Phase Models',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Center,
        )

        champion_model_type = ''
        failed_model_types: list[str] = []

        match best_model_configuration.model_type:
            case ModelType.Decision_Tree:
                champion_model_type = 'Decision Tree'
                failed_model_types: list[str] = [
                    'Random Forest',
                    'Logistic Regression'
                ]
            case ModelType.Random_Forest:
                champion_model_type = 'Random Forest'
                failed_model_types: list[str] = [
                    'Decision Tree',
                    'Logistic Regression'
                ]
            case ModelType.Logistic_Regression:
                champion_model_type = 'Logistic Regression'
                failed_model_types: list[str] = [
                    'Decision Tree',
                    'Random Forest',
                ]

        champion_model_type_text = format_text(
            text=f'**Champion Model Type:** {champion_model_type} (Carried forward to next phase)',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Left,
            insert_leading_space=True
        )

        failed_model_types_text = format_text(
            text='**Model Types Discarded:**',
            buffer_width=REPORT_WIDTH,
            prefix=border_char,
            suffix=border_char,
            alignment=TextAlignment.Left,
            insert_leading_space=True,
        )

        for fmt in failed_model_types:
            failed_model_types_text += format_text(
                text=f'- {fmt}: Failed to reach competitive performance in any balancing method.',
                buffer_width=REPORT_WIDTH,
                prefix=border_char,
                suffix=border_char,
                alignment=TextAlignment.Left,
                insert_leading_space=True,
                include_start_lf=True
            )

        best_phase_model_text = f'''{best_phase_model_header}
{best_model_overall_description}
{divider_line}
{viability_triage_header}
{divider_line}
{champion_model_type_text}
{blank_summary_line}
{failed_model_types_text}
{divider_line}'''

        self.final_summary_text += best_phase_model_text

    def process_results(
            self,
            results: list[ModelConfiguration],
            report_text: str,
            step_title: str
    ):
        """Process results from a batch of model evaluations. Appends a description
        of the results to the results_text property.

        Adds configurations to the overall list, identifies the best configuration
        from this batch, and updates the overall best configuration.

        Args:
            results (list[ModelConfiguration]): Configurations to process.
            report_text (str): Logging text to be included in the generated results text.
            step_title (str): Descriptive title for this evaluation step.
        """
        report_title = f'Model Evaluation Results - {step_title}'.center(
            REPORT_WIDTH, '-')
        rtext = f'''{report_text}
{report_title}
'''

        for mc in results:
            rtext += mc.info()

        rtext += '\n'
        self.model_configurations.extend(results)

        self.best_model_configuration_last_results, t = ModelResults.best_model_configuration(
            model_configurations=results,
            step_title=step_title
        )

        rtext += t
        rtext += self.best_model_configuration_last_results.info()
        rtext += '\n'

        self.best_model_configuration_overall, t = ModelResults.best_model_configuration(
            model_configurations=self.model_configurations,
            step_title='Overall'
        )

        rtext += t
        rtext += self.best_model_configuration_overall.info()

        self.results_text += rtext

    def evaluate_best_model_test_set(self) -> tuple[float, str]:
        """Evaluate the best overall model configuration on the test set.

        Trains and evaluates the best model on the held-out test set to assess
        final performance.

        Returns:
            float: The F1 score.
            str: Formatted test results including F1 score.
        """
        return ModelResults.evaluate_best_model_configuration_test_set(
            model_configuration=self.best_model_configuration_overall
        )

    @classmethod
    def evaluate_best_model_configuration_test_set(
        cls,
        model_configuration: Optional[ModelConfiguration]
    ) -> tuple[float, str]:
        """Evaluate a model configuration on the test set.

        Creates a new model instance with the given configuration's hyperparameters,
        trains it on combined train+validation data, and evaluates on the test set.
        Delegates to the appropriate model type's evaluate_test_set method.

        Args:
            model_configuration: Configuration specifying model type, hyperparameters,
                and data splits. If None, returns 0.0 score and error message.

        Returns:
            tuple[float, str]: A tuple containing:
                - F1 score achieved on test set (0.0 if configuration is None or unsupported)
                - Formatted report text with test evaluation results and metrics

        Note:
            Supports Decision Tree, Random Forest, and Logistic Regression models.
            Returns appropriate error message for None or unsupported model types.
        """
        if model_configuration is None:
            return 0.0, 'No model configuration provided'
        else:
            f1_score = 0.0
            report_text = ''
            model_type = model_configuration.model_type

            match model_type:
                case ModelType.Decision_Tree:
                    f1_score, t = DecisionTree.evaluate_test_set(
                        model_configuration)
                case ModelType.Random_Forest:
                    f1_score, t = RandomForest.evaluate_test_set(
                        model_configuration)
                case ModelType.Logistic_Regression:
                    f1_score, t = LogisticRegression.evaluate_test_set(
                        model_configuration)
                case _:
                    f1_score, t = 0.0, f'Unsupported Model Type for Best Model: {model_configuration.model_type}'

            report_text += t
            return f1_score, report_text

    @classmethod
    def filter_model_configurations(
        cls,
        model_configurations: list[ModelConfiguration],
        model_type: Optional[ModelType] = None,
        model_balancing: Optional[ModelBalancing] = None
    ) -> list[ModelConfiguration]:
        """Filter model configurations by type and balancing strategy.

        Args:
            model_configurations (list[ModelConfiguration]): Configurations to filter.
            model_type (Optional[ModelType]): Filter by model type (None = all).
            model_balancing (Optional[ModelBalancing]): Filter by balancing (None = all).

        Returns:
            list[ModelConfiguration]: Filtered list of configurations.
        """
        model_configurations_to_consider: list[ModelConfiguration] = []

        for mc in model_configurations:
            if model_type is not None and mc.model_type is not model_type:
                continue

            if model_balancing is not None and mc.model_balancing is not model_balancing:
                continue

            model_configurations_to_consider.append(mc)

        return model_configurations_to_consider

    @classmethod
    def best_model_configuration(
        cls,
        model_configurations: list[ModelConfiguration],
        model_type: Optional[ModelType] = None,
        model_balancing: Optional[ModelBalancing] = None,
        step_title: str = 'Overall'
    ) -> tuple[ModelConfiguration, str]:
        """Find the best model configuration based on F1 score.

        Args:
            model_configurations (list[ModelConfiguration]): Configurations to evaluate.
            model_type (Optional[ModelType]): Filter by model type.
            model_balancing (Optional[ModelBalancing]): Filter by balancing strategy.
            step_title (str): Title for display purposes.

        Returns:
            ModelConfiguration: Configuration with highest F1 score, or empty if none match.
        """
        model_configurations_to_consider = ModelResults.filter_model_configurations(
            model_configurations=model_configurations,
            model_type=model_type,
            model_balancing=model_balancing
        )

        report_title = f'Best Model Configuration - {step_title}'.center(
            REPORT_WIDTH, '-')
        report_text = f'''{report_title}
              --------- Filter ---------
              Model Type: {model_type.name if model_type is not None else "Any"}
              Model Balancing: {model_balancing.name if model_balancing is not None else "Any"}
'''

        if len(model_configurations_to_consider) == 0:
            print('No model configurations match the given criteria')
            return ModelConfiguration.empty(), report_text

        best_mc = max(model_configurations_to_consider,
                      key=lambda item: item.f1_score_valid)
        return best_mc, report_text

    @classmethod
    def print_model_configurations(
        cls,
        model_configurations: list[ModelConfiguration],
        model_type: Optional[ModelType] = None,
        model_balancing: Optional[ModelBalancing] = None,
        sort_order: ModelConfigurationSortOrder = ModelConfigurationSortOrder.No_Sort,
        reverse: bool = False
    ):
        """Prints model configurations based on filtering and sort order.

        Args:
            model_configurations (list[ModelConfiguration]): Configurations to display.
            model_type (Optional[ModelType]): Filter by model type.
            model_balancing (Optional[ModelBalancing]): Filter by balancing strategy.
            sort_order (ModelConfigurationSortOrder): How to sort the results.
            reverse (bool): Whether to reverse sort order.
        """
        model_configurations_to_consider = ModelResults.filter_model_configurations(
            model_configurations=model_configurations,
            model_type=model_type,
            model_balancing=model_balancing
        )

        if len(model_configurations_to_consider) == 0:
            print('No model configurations match the given criteria')
            return

        if sort_order is ModelConfigurationSortOrder.No_Sort:
            mcs = model_configurations_to_consider
        elif sort_order is ModelConfigurationSortOrder.F1_Score:
            mcs = sorted(model_configurations_to_consider,
                         key=lambda item: item.f1_score_valid, reverse=reverse)
        elif sort_order is ModelConfigurationSortOrder.Model_Type:
            mcs = sorted(model_configurations_to_consider,
                         key=lambda item: item.model_type.name, reverse=reverse)
        elif sort_order is ModelConfigurationSortOrder.Model_Balancing:
            mcs = sorted(model_configurations_to_consider,
                         key=lambda item: item.model_balancing.name, reverse=reverse)
        else:
            mcs = model_configurations_to_consider

        for mc in mcs:
            mc.info()


@dataclass
class BestModelResults:
    """Tracks best model configurations across multiple evaluation phases.

    Maintains the best performing model configuration for each model type
    (Decision Tree, Random Forest, Logistic Regression) and overall best
    model across all phases evaluated.

    Attributes:
        best_dtree_model_configuration: Best decision tree configuration found.
        best_rf_model_configuration: Best random forest configuration found.
        best_lr_model_configuration: Best logistic regression configuration found.
        best_model_configuration: Overall best model configuration.
    """
    best_dtree_model_configuration: Optional[ModelConfiguration] = None
    best_rf_model_configuration: Optional[ModelConfiguration] = None
    best_lr_model_configuration: Optional[ModelConfiguration] = None
    best_model_configuration: Optional[ModelConfiguration] = None

    def compare_model_results(
            self,
            model_results: ModelResults,
    ) -> None:
        """Update best configurations if new results are better.

        Compares the provided model results against currently stored best
        configurations for each model type. Updates stored configurations
        if the new results show better performance.

        Args:
            model_results: Results from a model evaluation phase.
        """
        bmc, _ = ModelResults.best_model_configuration(
            model_results.model_configurations,
            model_type=ModelType.Decision_Tree
        )

        if self.best_dtree_model_configuration is None or \
                bmc > self.best_dtree_model_configuration:
            self.best_dtree_model_configuration = bmc

        if self.best_model_configuration is None or \
                bmc > self.best_model_configuration:
            self.best_model_configuration = bmc

        bmc, _ = ModelResults.best_model_configuration(
            model_configurations=model_results.model_configurations,
            model_type=ModelType.Random_Forest
        )

        if self.best_rf_model_configuration is None or \
                bmc > self.best_rf_model_configuration:
            self.best_rf_model_configuration = bmc

        if self.best_model_configuration is None or \
                bmc > self.best_model_configuration:
            self.best_model_configuration = bmc

        bmc, _ = ModelResults.best_model_configuration(
            model_configurations=model_results.model_configurations,
            model_type=ModelType.Logistic_Regression
        )

        if self.best_lr_model_configuration is None or \
                bmc > self.best_lr_model_configuration:
            self.best_lr_model_configuration = bmc

        if self.best_model_configuration is None or \
                bmc > self.best_model_configuration:
            self.best_model_configuration = bmc

        model_results.update_final_summary_text(
            best_model_configuration=self.best_model_configuration,
        )

    def best_dtree_model(
            self,
            class_weight_override: Optional[Union[
                Mapping[str, float],
                str
            ]] = None,
    ) -> DecisionTreeClassifier:
        """Create DecisionTreeClassifier from best configuration.

        Args:
            class_weight (Optional[Union[Mapping[str, float], str]]): Class balancing strategy.
                Use this parameter to override the class_weight value in the model configuration.

        Returns:
            DecisionTreeClassifier configured with best hyperparameters found.

        Raises:
            ValueError: If no decision tree configuration has been stored yet.
        """
        if self.best_dtree_model_configuration is None:
            raise ValueError("No decision tree configuration available")

        return DecisionTree.model_from_hyperparameters(
            params=self.best_dtree_model_configuration.params,
            random_state=self.best_dtree_model_configuration.data_splits.random_state,
            class_weight=self.best_dtree_model_configuration.class_weight if class_weight_override is None else class_weight_override
        )

    def best_rf_model(
            self,
            class_weight_override: Optional[Union[
                Mapping[str, float],
                str
            ]] = None,
    ) -> RandomForestClassifier:
        """Create RandomForestClassifier from best configuration.

        Args:
            class_weight (Optional[Union[Mapping[str, float], str]]): Class balancing strategy.
                Use this parameter to override the class_weight value in the model configuration.

        Returns:
            RandomForestClassifier configured with best hyperparameters found.

        Raises:
            ValueError: If no random forest configuration has been stored yet.
        """
        if self.best_rf_model_configuration is None:
            raise ValueError("No random forest configuration available")

        return RandomForest.model_from_hyperparameters(
            params=self.best_rf_model_configuration.params,
            random_state=self.best_rf_model_configuration.data_splits.random_state,
            class_weight=self.best_rf_model_configuration.class_weight if class_weight_override is None else class_weight_override
        )

    def best_lr_model(
            self,
            class_weight_override: Optional[Union[
                Mapping[str, float],
                str
            ]] = None,
    ) -> SklearnLogisticRegression:
        """Create LogisticRegression from best configuration.

        Args:
            class_weight (Optional[Union[Mapping[str, float], str]]): Class balancing strategy.
                Use this parameter to override the class_weight value in the model configuration.

        Returns:
            LogisticRegression configured with best hyperparameters found.

        Raises:
            ValueError: If no logistic regression configuration has been stored yet.
        """
        if self.best_lr_model_configuration is None:
            raise ValueError("No logistic regression configuration available")

        return LogisticRegression.model_from_hyperparameters(
            params=self.best_lr_model_configuration.params,
            random_state=self.best_lr_model_configuration.data_splits.random_state,
            class_weight=self.best_lr_model_configuration.class_weight if class_weight_override is None else class_weight_override
        )

    def info(
            self,
    ) -> str:
        """Generate summary report of best model configurations.

        Returns:
            Formatted string containing details of all best model configurations
            and the highest test set F1 score achieved.
        """
        best_dtree_text = 'None' if self.best_dtree_model_configuration is None else \
            self.best_dtree_model_configuration.info()

        best_rf_text = 'None' if self.best_rf_model_configuration is None else \
            self.best_rf_model_configuration.info()

        best_lr_text = 'None' if self.best_lr_model_configuration is None else \
            self.best_lr_model_configuration.info()

        best_mc_text = 'None' if self.best_model_configuration is None else \
            self.best_model_configuration.info()

        return f'''
Best Decision Tree Model Configuration Overall:
{best_dtree_text}

Best Random Forest Model Configuration Overall:
{best_rf_text}

Best Logistic Regression Model Configuration Overall:
{best_lr_text}

Best Model Configuration Overall:
{best_mc_text}
'''
