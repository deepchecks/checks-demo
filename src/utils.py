from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from deepchecks import BaseCheck, TrainTestBaseCheck
from deepchecks.tabular import Dataset

from datasets import DatasetOption


def insert_categorical_drift(column: pd.Series, percent, category):
    column = column.to_numpy()
    categories = list(set(np.unique(column)) - {category})
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
    return column + np.random.normal(mean, std, size=(column.shape[0]))


def build_snippet(check: BaseCheck,
                  dataset_opt: DatasetOption,
                  properties: dict = None,
                  model: bool = False,
                  condition_name: str = None):
    check_name = check.__class__.__name__
    is_train_test = isinstance(check, TrainTestBaseCheck)
    dataset_params = prepare_properties_string(dataset_opt['dataset_params'])

    if is_train_test:
        check_arguments = 'train_dataset, test_dataset'
        dataset_string = (f'path_to_train_data = "train.csv"\n'
                          f'path_to_test_data = "test.csv"\n'
                          f'train_dataset = Dataset(pd.read_csv(path_to_train_data), {dataset_params})\n'
                          f'test_dataset = Dataset(pd.read_csv(path_to_test_data), {dataset_params})')
    else:
        check_arguments = 'dataset'
        dataset_string = f'path_to_data = "data.csv"\n' \
                         f'dataset = Dataset(pd.read_csv(path_to_data), {dataset_params})'

    if model:
        check_arguments += ', model=model'
        model_load_string = dataset_opt['model_snippet']
    else:
        model_load_string = ''
    properties_string = prepare_properties_string(properties)
    condition_string = f'.{condition_name}' if condition_name else ''
    snippet = ('import os; import sys; os.system(f"{sys.executable} -m pip install -U --quiet deepchecks")\n'
               f'import pandas as pd\n'
               f'from deepchecks.tabular.checks import {check_name}\n'
               f'from deepchecks.tabular import Dataset\n'
               f'{model_load_string}\n'
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


def add_download_button(dataset_tuple: Tuple[Dataset, Optional[Dataset]]):
    if len(dataset_tuple) == 1:
        st.download_button('Download Data', data=dataset_tuple[0].data.to_csv(), file_name='data.csv',
                                   on_click=lambda: st.balloons())
    else:
        st.download_button('Download Train Data', data=dataset_tuple[0].data.to_csv(), file_name='train.csv')
        st.download_button('Download Test Data', data=dataset_tuple[1].data.to_csv(), file_name='test.csv')
