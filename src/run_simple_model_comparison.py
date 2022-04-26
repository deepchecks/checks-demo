import streamlit as st
from deepchecks.tabular.checks import SimpleModelComparison

from datasets import DatasetOption
from utils import build_snippet, add_download_button


def run(dataset_option: DatasetOption):
    st.sidebar.subheader(f'Check Parameters')
    model_type = st.sidebar.selectbox('Simple Model Type', ['tree', 'random', 'constant'])
    check = SimpleModelComparison(simple_model_type=model_type).add_condition_gain_not_less_than()
    snippet = build_snippet(check, dataset_option, properties={'simple_model_type': model_type}, model=True,
                            condition_name='add_condition_gain_not_less_than')
    add_download_button(dataset_option['train'], dataset_option['test'])
    return check.run(dataset_option['train'], dataset_option['test'], model=dataset_option['model'],
                     features_importance=dataset_option['features_importance']), snippet
