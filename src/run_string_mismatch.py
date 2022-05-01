import random

import numpy as np
import pandas as pd
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import StringMismatch

from datasets import DatasetOption
from utils import build_snippet


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    dataset: Dataset = dataset_option['train']
    new_data = dataset.data.copy()

    if not dataset.cat_features:
        return 'No categorical features found in dataset, try another dataset', '', lambda: None

    with check_param_col:
        # Show column selector
        column: str = st.selectbox('Select a column', dataset.cat_features)

    with manipulate_col:
        percent = st.slider('Variants Percent', value=0, min_value=0, max_value=100, step=1)
        if percent > 0:
            new_data[column] = insert_variants(new_data[column], percent)

    check = StringMismatch(columns=[column]).add_condition_ratio_variants_not_greater_than(0.01)
    snippet = build_snippet(check, dataset_option, condition_name='add_condition_ratio_variants_not_greater_than(0.01)',
                            properties={'columns': [column]})
    dataset = dataset.copy(new_data)

    return check.run(dataset), snippet, (dataset,)


def insert_variants(column: pd.Series, percent):
    column = column.to_numpy()
    value = np.random.choice(column)
    variants = random_variants(value)
    size = min(int(len(column) * percent / 100), len(column))
    indices_to_replace = np.random.choice(len(column), size=size, replace=False)
    for index in indices_to_replace:
        column[index] = random.choice(variants)
    return column


def flip_case(value):
    return value.upper() if value.islower() else value.lower()


def random_variants(value):
    functions = [
        lambda x: flip_case(x[0]) + x[1:],  # Switch upper lower first letter
        lambda x: x.replace(' ', '-'),  # Replace space with -
        lambda x: x + '.',  # Add . at the end
        lambda x: x.upper(),  # Upper case all letters
        lambda x: x.lower(),  # Lower case all letters
        lambda x: ' ' + x,  # Add space at the beginning
    ]

    return [f(value) for f in functions if f(value) != value]
