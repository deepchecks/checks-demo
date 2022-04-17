from typing import TypedDict, Tuple, Callable, List, Dict, Any

import streamlit as st

from deepchecks import BaseCheck
from deepchecks.tabular.checks.distribution import TrainTestFeatureDrift, TrainTestLabelDrift
from deepchecks.tabular.checks.integrity import StringMismatch, DataDuplicates
from deepchecks.tabular.checks.performance import SegmentPerformance
from src.checks import train_test_feature_drift, train_test_label_drift, string_mismatch, data_duplicates, \
    segment_performance

__all__ = ['get_checks_options']



class ControlWidget(TypedDict):
    type: str
    params: Any
    name: str


class DatasetManipulation(TypedDict):
    name: str
    function: Any
    controls: List


class CheckOptions(TypedDict):
    run: Callable
    snippet: str
    docstring: str


def get_checks_options():
    return {
        TrainTestFeatureDrift.name(): CheckOptions(
            run=train_test_feature_drift.run,
            snippet='from deepchecks.tabular.checks import TrainTestFeatureDrift\n'
                    'TrainTestFeatureDrift().run(train, test)',
            docstring=TrainTestFeatureDrift.__doc__
        ),
        TrainTestLabelDrift.name(): CheckOptions(
            run=train_test_label_drift.run,
            snippet='from deepchecks.tabular.checks import TrainTestLabelDrift\n'
                    'TrainTestLabelDrift().run(train, test)',
            docstring=TrainTestLabelDrift.__doc__
        ),
        StringMismatch.name(): CheckOptions(
            run=string_mismatch.run,
            snippet='from deepchecks.tabular.checks import StringMismatch\n'
                    'StringMismatch().run(data)',
            docstring=StringMismatch.__doc__
        ),
        DataDuplicates.name(): CheckOptions(
            run=data_duplicates.run,
            snippet='from deepchecks.tabular.checks import DataDuplicates\n'
                    'DataDuplicates().run(data)',
            docstring=DataDuplicates.__doc__
        ),
        SegmentPerformance.name(): CheckOptions(
            run=segment_performance.run,
            snippet='from deepchecks.tabular.checks import SegmentPerformance\n'
                    'SegmentPerformance().run(data, model)',
            docstring=SegmentPerformance.__doc__
        )
    }

