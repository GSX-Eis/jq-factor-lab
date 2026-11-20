"""Time-series research contracts, standard CSV readers and calendar views."""
from .contracts import (
    EXCLUSION_PRIORITY,
    Bar,
    ContractValidationError,
    DuplicateKeyError,
    ExclusionReason,
    FactorValue,
    LabelRecord,
    ResultIdentity,
    invalid_price_mask,
    primary_exclusion_reason,
    primary_exclusion_reasons,
    sha256_file,
    validate_bar_frame,
    validate_factor_frame,
)
from .csv_reader import BarAvailability, filter_bars_by_view, read_bar_csv, read_factor_csv
from .labels import BarReturnLabeler
from .inference import BarTrialFamily
from .evaluator import (
    RESULT_SCHEMA_VERSION, BarCrossSectionalEvaluator, EvaluationResult, TimeSeriesEvaluator, evaluate_bars,
)
from .artifacts import write_evaluation_artifacts

__all__ = [
    'EXCLUSION_PRIORITY', 'Bar', 'ContractValidationError', 'DuplicateKeyError',
    'ExclusionReason', 'FactorValue', 'LabelRecord', 'ResultIdentity',
    'invalid_price_mask', 'primary_exclusion_reason', 'primary_exclusion_reasons',
    'sha256_file', 'validate_bar_frame', 'validate_factor_frame',
    'BarAvailability', 'filter_bars_by_view', 'read_bar_csv', 'read_factor_csv',
    'BarReturnLabeler',
    'BarCrossSectionalEvaluator', 'EvaluationResult', 'TimeSeriesEvaluator', 'evaluate_bars',
    'BarTrialFamily', 'RESULT_SCHEMA_VERSION',
    'write_evaluation_artifacts',
]
