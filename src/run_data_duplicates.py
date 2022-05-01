import numpy as np
import pandas as pd
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates

from datasets import DatasetOption
from utils import build_snippet


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    dataset: Dataset = dataset_option['train']

    with check_param_col:
        st.text('No parameters to control')
    with manipulate_col:
        rows_to_duplicate = st.slider('Number rows to duplicate', min_value=1, max_value=5, value=5)
        percent = st.slider('Duplicate percent', value=20, min_value=0, max_value=100, step=1)
        if percent > 0:
            new_data = insert_duplicates(dataset.data, rows_to_duplicate, percent)
            dataset = dataset.copy(new_data)

    check = DataDuplicates().add_condition_ratio_not_greater_than(0.1)
    snippet = build_snippet(check, dataset_option, condition_name='add_condition_ratio_not_greater_than(0.1)')
    return check.run(dataset), snippet, (dataset,)


def insert_duplicates(data, rows_to_duplicate_num, percent):
    np_data = data.to_numpy()
    indices_to_duplicate = np.random.choice(len(data), size=rows_to_duplicate_num, replace=False)
    probabilities = np.random.default_rng().random(rows_to_duplicate_num)
    probabilities = probabilities / np.sum(probabilities)
    amount_to_replace = min(int(len(data) * percent / 100), len(data))
    rows_to_put = np_data[np.random.choice(indices_to_duplicate, size=amount_to_replace, p=probabilities)]
    indices_to_replace = np.random.choice(len(data), size=amount_to_replace, replace=False)
    np_data[indices_to_replace] = rows_to_put
    return pd.DataFrame(np_data, columns=data.columns)
