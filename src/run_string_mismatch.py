import random

import numpy as np
import pandas as pd
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import StringMismatch

from datasets import DatasetOption
from utils import build_snippet, add_download_button


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    dataset: Dataset = dataset_option['train']
    new_data = dataset.data.copy()

    if not dataset.cat_features:
        return 'No categorical features found in dataset, try another dataset', '', lambda: None

    with check_param_col:
        # Show column selector
        column: str = st.selectbox('Select a column', dataset.cat_features)

    with manipulate_col:
        if st.checkbox('Insert variants', value=True):
            new_data[column] = insert_variants(new_data[column])

    check = StringMismatch(columns=[column]).add_condition_ratio_variants_not_greater_than(0.01)
    snippet = build_snippet(check, dataset_option, condition_name='add_condition_ratio_variants_not_greater_than(0.01)',
                            properties={'columns': [column]})
    dataset = dataset.copy(new_data)

    return check.run(dataset), snippet, lambda: add_download_button(dataset)


def insert_variants(column: pd.Series):
    column = column.to_numpy()
    value = np.random.choice(column)
    possible_indices = np.where(column != value)
    indices_to_replace = np.random.choice(possible_indices[0], size=4, replace=False)
    for index in indices_to_replace:
        column[index] = random_variant(value)
    return column


def random_variant(value):
    special_chars = '!@#$%^&*()_+{}[]|\:;"<>?,./'
    value = str(value)
    random_index = random.randint(0, len(value) - 1)
    val_in_index = value[random_index]
    if val_in_index.isalpha():
        changed_char = val_in_index.upper() if val_in_index.islower() else val_in_index.lower()
        value = value[:random_index] + changed_char + value[random_index + 1:]
    else:
        value = value[:random_index] + random.choice(special_chars) + value[random_index:]

    return value
