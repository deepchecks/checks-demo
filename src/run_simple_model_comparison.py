import streamlit as st
from deepchecks.tabular.checks import SimpleModelComparison

from datasets import DatasetOption
from utils import build_snippet


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    with check_param_col:
        model_type = st.selectbox('Simple Model Type', ['tree', 'random', 'constant'])
    with manipulate_col:
        st.text('No corruption to perform')
    check = SimpleModelComparison(simple_model_type=model_type).add_condition_gain_not_less_than(0.1)
    snippet = build_snippet(check, dataset_option, properties={'simple_model_type': model_type}, model=True,
                            condition_name='add_condition_gain_not_less_than(0.1)')
    return check.run(dataset_option.train, dataset_option.test, model=dataset_option.model,
                     features_importance=dataset_option.features_importance), \
        snippet, (dataset_option.train, dataset_option.test)
