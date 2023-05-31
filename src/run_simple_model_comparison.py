import streamlit as st
from deepchecks.tabular.checks import SimpleModelComparison

from datasets import DatasetOption
from streamlit_persist import persist
from utils import build_snippet, put_data_on_state


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    with check_param_col:
        model_type = st.selectbox('Simple Model Type', ['tree', 'random', 'constant'], key=persist('simple_model_type'))
    check = SimpleModelComparison(simple_model_type=model_type).add_condition_gain_greater_than(0.1)
    snippet = build_snippet(check, dataset_option, properties={'simple_model_type': model_type}, model=True,
                            condition_name='add_condition_gain_greater_than(0.1)')
    put_data_on_state(dataset_option.train, dataset_option.test, dataset_type='test', corrupted_dataset_index=1)
    return check.run(dataset_option.train, dataset_option.test, model=dataset_option.model,
                     feature_importance=dataset_option.features_importance), \
        snippet
