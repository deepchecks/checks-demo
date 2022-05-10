import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import StringMismatch

from corruptions import insert_variants
from datasets import DatasetOption
from utils import build_snippet, put_data_on_state


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    dataset: Dataset = dataset_option.train
    new_data = dataset.data.copy()

    if not dataset.cat_features:
        raise Exception('No categorical features in dataset, should not have reached here')

    with check_param_col:
        # Show column selector
        column: str = st.selectbox('Select a column', dataset.cat_features)

    with manipulate_col:
        percent = st.slider('Variants Percent', value=10, min_value=0, max_value=100, step=1)
        if percent > 0:
            new_data[column] = insert_variants(new_data[column], percent)

    check = StringMismatch(columns=[column]).add_condition_ratio_variants_not_greater_than(0.01)
    snippet = build_snippet(check, dataset_option, condition_name='add_condition_ratio_variants_not_greater_than(0.01)',
                            properties={'columns': [column]})
    dataset = dataset.copy(new_data)
    put_data_on_state(dataset)
    return check.run(dataset), snippet
