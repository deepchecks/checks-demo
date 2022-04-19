from deepchecks.tabular.checks import SimpleModelComparison
from deepchecks.tabular.checks.distribution import TrainTestFeatureDrift, TrainTestLabelDrift
from deepchecks.tabular.checks.integrity import StringMismatch, DataDuplicates
from deepchecks.tabular.checks.performance import SegmentPerformance

import run_train_test_feature_drift, run_train_test_label_drift, run_string_mismatch, run_data_duplicates, \
    run_segment_performance, run_simple_model_comparison

__all__ = ['get_checks_options']


def get_checks_options():
    return {
        TrainTestFeatureDrift: run_train_test_feature_drift.run,
        TrainTestLabelDrift: run_train_test_label_drift.run,
        StringMismatch: run_string_mismatch.run,
        DataDuplicates: run_data_duplicates.run,
        SegmentPerformance: run_segment_performance.run,
        SimpleModelComparison: run_simple_model_comparison.run
    }

