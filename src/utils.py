from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from deepchecks import BaseCheck, TrainTestBaseCheck
from deepchecks.tabular import Dataset

from datasets import DatasetOption


def insert_categorical_drift(column: pd.Series, percent, category):
    column = column.to_numpy()
    categories = np.delete(np.unique(column), [category])
    ratio = percent / 100
    category_count = np.count_nonzero(column == category)
    category_ratio = category_count / column.shape[0]
    amount_to_replace = int(column.shape[0] * (ratio - category_ratio))
    if amount_to_replace == 0:
        return column

    possible_indices = np.where(column != category) if amount_to_replace > 0 else np.where(column == category)
    indices_to_replace = np.random.choice(possible_indices[0], abs(amount_to_replace), replace=False)
    for index in indices_to_replace:
        column[index] = category if amount_to_replace > 0 else np.random.choice(categories)

    return column


def insert_numerical_drift(column: pd.Series, mean, std):
    return column + np.random.normal(mean, std, size=(column.shape[0])) * np.average(column)


def build_snippet(check: BaseCheck,
                  dataset_opt: DatasetOption,
                  properties: dict = None,
                  model: bool = False,
                  condition_name: str = None,
                  condition_params: dict = None):
    check_name = check.__class__.__name__
    is_train_test = isinstance(check, TrainTestBaseCheck)
    dataset_params = prepare_properties_string(dataset_opt['dataset_params'])

    if is_train_test:
        check_arguments = 'train_dataset, test_dataset'
        dataset_string = (f'train_dataset = Dataset(pd.read_csv("train.csv"), {dataset_params})\n'
                          f'test_dataset = Dataset(pd.read_csv("test.csv"), {dataset_params})')
    else:
        check_arguments = 'dataset'
        dataset_string = f'dataset = Dataset(pd.read_csv("data.csv"), {dataset_params})'

    if model:
        check_arguments += ', model=model'
    properties_string = prepare_properties_string(properties)
    condition_params_string = prepare_properties_string(condition_params)
    condition_string = f'.{condition_name}({condition_params_string})' if condition_name else ''
    snippet = ('import os; import sys; os.system(f"{sys.executable} -m pip install -U --quiet deepchecks")\n'
               f'import pandas as pd; from deepchecks.tabular.checks import {check_name}; from deepchecks.tabular import Dataset\n\n'
               f'{dataset_string}\n\n'
               f'check = {check_name}({properties_string}){condition_string}\n'
               f'result = check.run({check_arguments})\n'
               'result.show()')

    return snippet


def quote_params(params):
    if isinstance(params, str):
        return f'"{params}"'
    elif isinstance(params, list):
        return '[' + ', '.join([f'"{x}"' for x in params]) + ']'
    else:
        return params


def prepare_properties_string(properties: dict):
    return ', '.join([f'{k}={quote_params(v)}' for k, v in properties.items()]) if properties else ''


def add_download_button(train: Dataset, test: Optional[Dataset] = None):
    if test is None:
        st.sidebar.download_button('Download Data', data=train.data.to_csv(), file_name='data.csv')
    else:
        st.sidebar.download_button('Download Train Data', data=train.data.to_csv(), file_name='train.csv')
        st.sidebar.download_button('Download Test Data', data=test.data.to_csv(), file_name='test.csv')
