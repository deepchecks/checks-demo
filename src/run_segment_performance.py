import streamlit as st

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import SegmentPerformance

from datasets import DatasetOption


def run(dataset_option: DatasetOption):
    dataset: Dataset = dataset_option['test']

    st.sidebar.subheader(f'Check Parameters')

    column_1: str = st.sidebar.selectbox('Select first column', dataset.features)
    second_features = set(dataset.features) - {column_1}
    column_2: str = st.sidebar.selectbox('Select second column', second_features)

    check = SegmentPerformance(feature_1=column_1, feature_2=column_2)
    return check.run(dataset_option['test'], model=dataset_option['model'])
