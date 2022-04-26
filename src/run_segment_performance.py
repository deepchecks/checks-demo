import streamlit as st

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import SegmentPerformance

from datasets import DatasetOption
from utils import build_snippet, add_download_button


def run(dataset_option: DatasetOption):
    dataset: Dataset = dataset_option['test']

    st.sidebar.subheader(f'Check Parameters')

    column_1: str = st.sidebar.selectbox('Select first column', dataset.features)
    second_features = set(dataset.features) - {column_1}
    column_2: str = st.sidebar.selectbox('Select second column', second_features)

    check = SegmentPerformance(feature_1=column_1, feature_2=column_2)
    snippet = build_snippet(check, dataset_option, properties={'feature_1': column_1, 'feature_2': column_2},
                            model=True)
    add_download_button(dataset)
    return check.run(dataset, model=dataset_option['model'],
                     features_importance=dataset_option['features_importance']), snippet
