from typing import NamedTuple
import numpy as np

INTEGER_SCHEMA_WARNING = "Hint: Inferred schema contains integer column"

class TrainingResult(NamedTuple):
    run_name: str
    run_id: str

class EvaluationResult(NamedTuple):
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float
    brier: float