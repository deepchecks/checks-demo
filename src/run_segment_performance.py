import streamlit as st

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import SegmentPerformance

from datasets import DatasetOption
from utils import build_snippet, put_data_on_state


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    dataset: Dataset = dataset_option.test

    with check_param_col:
        column_1: str = st.selectbox('Select first column', dataset.features)
        second_features = set(dataset.features) - {column_1}
        column_2: str = st.selectbox('Select second column', second_features)

    properties = dict(feature_1=column_1, feature_2=column_2, max_segments=3)
    check = SegmentPerformance(**properties)
    snippet = build_snippet(check, dataset_option, properties=properties, model=True)
    put_data_on_state(dataset)
    return check.run(dataset, model=dataset_option.model, features_importance=dataset_option.features_importance), snippet
