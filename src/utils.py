from typing import Optional, Tuple

import streamlit as st
from deepchecks import BaseCheck, TrainTestBaseCheck
from deepchecks.tabular import Dataset

from datasets import DatasetOption


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


def get_query_param(param_name: str):
    query_params = st.experimental_get_query_params()
    # Get selected check from query params if exists
    if param_name in query_params:
        return query_params[param_name][0]
    # Set default query params if not exists
    else:
        return None


def set_query_param(param_name: str, state_id):
    st.experimental_set_query_params(**{param_name: st.session_state[state_id]})
