import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates

from corruptions import insert_duplicates
from datasets import DatasetOption
from utils import build_snippet, put_data_on_state


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    dataset: Dataset = dataset_option.train

    with check_param_col:
        st.text('No parameters to control')
    with manipulate_col:
        st.subheader('Add Corruption to Data')
        rows_to_duplicate = st.slider('Number rows to duplicate', min_value=1, max_value=5, value=5)
        percent = st.slider('Duplicate percent', value=20, min_value=0, max_value=100, step=1)
        if percent > 0:
            new_data = insert_duplicates(dataset.data, rows_to_duplicate, percent)
            dataset = dataset.copy(new_data)

    check = DataDuplicates().add_condition_ratio_not_greater_than(0.1)
    snippet = build_snippet(check, dataset_option, condition_name='add_condition_ratio_not_greater_than(0.1)')
    put_data_on_state(dataset)
    return check.run(dataset), snippet
