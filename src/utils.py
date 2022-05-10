from contextlib import contextmanager
from threading import current_thread

import numpy as np
import pandas as pd
import streamlit as st
from deepchecks import BaseCheck, TrainTestBaseCheck
from streamlit.scriptrunner.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME

from constants import DATA_STATE_ID
from datasets import DatasetOption


def build_run_params(is_train_test: bool, model: bool, dataset_opt: DatasetOption):
    dataset_params = prepare_properties_string(dataset_opt.dataset_params)

    if is_train_test:
        run_arguments = 'train_dataset, test_dataset'
        dataset_string = (f'path_to_train_data = "train.csv"\n'
                          f'path_to_test_data = "test.csv"\n'
                          f'train_dataset = Dataset(pd.read_csv(path_to_train_data), {dataset_params})\n'
                          f'test_dataset = Dataset(pd.read_csv(path_to_test_data), {dataset_params})')
    else:
        run_arguments = 'dataset'
        dataset_string = f'path_to_data = "data.csv"\n' \
                         f'dataset = Dataset(pd.read_csv(path_to_data), {dataset_params})'

    if model:
        run_arguments += ', model=model'
        model_load_string = dataset_opt.model_snippet
    else:
        model_load_string = ''

    return run_arguments, dataset_string, model_load_string


def build_snippet(check: BaseCheck,
                  dataset_opt: DatasetOption,
                  properties: dict = None,
                  model: bool = False,
                  condition_name: str = None):
    check_name = check.__class__.__name__
    is_train_test = isinstance(check, TrainTestBaseCheck)

    run_arguments, dataset_string, model_load_string = build_run_params(is_train_test, model, dataset_opt)
    properties_string = prepare_properties_string(properties)
    condition_string = f'.{condition_name}' if condition_name else ''
    snippet = ('import os; import sys; os.system(f"{sys.executable} -m pip install -U --quiet deepchecks")\n'
               f'import pandas as pd\n'
               f'from deepchecks.tabular.checks import {check_name}\n'
               f'from deepchecks.tabular import Dataset\n'
               f'{model_load_string}\n'
               f'{dataset_string}\n\n'
               f'check = {check_name}({properties_string}){condition_string}\n'
               f'result = check.run({run_arguments})\n'
               'result.show()')

    return snippet


def build_suite_snippet(suite_func, dataset_opt: DatasetOption, is_train_test: bool):
    run_arguments, dataset_string, model_load_string = build_run_params(is_train_test, True, dataset_opt)

    snippet = ('import os; import sys; os.system(f"{sys.executable} -m pip install -U --quiet deepchecks")\n'
               f'import pandas as pd\n'
               f'from deepchecks.tabular.suites import {suite_func.__name__}\n'
               f'from deepchecks.tabular import Dataset\n'
               f'{model_load_string}\n'
               f'{dataset_string}\n\n'
               f'suite = {suite_func.__name__}()\n'
               f'result = suite.run({run_arguments})\n'
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


def put_data_on_state(*datasets):
    data_frames = [d.data for d in datasets]
    st.session_state[DATA_STATE_ID] = data_frames


def add_download_button():
    if DATA_STATE_ID not in st.session_state:
        return
    data_frames = st.session_state[DATA_STATE_ID]
    if len(data_frames) == 1:
        st.download_button('Download Data', data=data_frames[0].to_csv().encode('utf-8'), file_name='data.csv',
                           mime="text/csv", on_click=lambda: st.balloons())
    else:
        st.download_button('Download Train Data', data=data_frames[0].to_csv().encode('utf-8'), file_name='train.csv', mime="text/csv")
        st.download_button('Download Test Data', data=data_frames[1].to_csv().encode('utf-8'), file_name='test.csv', mime="text/csv")


def get_query_param(param_name: str):
    query_params = st.experimental_get_query_params()
    # Get selected check from query params if exists
    if param_name in query_params:
        return query_params[param_name][0]
    return None


def set_query_param(param_name: str, state_id):
    st.experimental_set_query_params(**{param_name: st.session_state[state_id]})


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)
    old_write = src.write

    def new_write(b):
        if getattr(current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
            output_func(b)
        else:
            old_write(b)

    try:
        src.write = new_write
        yield
    finally:
        src.write = old_write
        placeholder.empty()


def std_without_outliers(data, outlier_threshold=0.025):
    data = data.to_numpy() if isinstance(data, pd.Series) else data
    data = data[np.where((data < np.quantile(data, 1 - outlier_threshold)) & (data > np.percentile(data, outlier_threshold)))]
    return np.std(data)
