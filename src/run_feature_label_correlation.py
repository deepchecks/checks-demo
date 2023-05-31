import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation

from corruptions import relate_column_to_label
from datasets import DatasetOption
from streamlit_persist import persist
from utils import build_snippet, put_data_on_state


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    dataset: Dataset = dataset_option.test

    with check_param_col:
        st.text('No parameters to configure')
    with manipulate_col:
        st.subheader('Add Corruption to Train Data')
        # Allow manipulation only for numeric columns
        column: str = st.selectbox('Select a column', dataset.numerical_features)
        power = st.slider('Label correlation power', min_value=0., max_value=10., value=1., step=0.1, key=persist('single_feature_label_power'))

    if power > 0:
        new_data = dataset.data.copy()
        new_data[column] = relate_column_to_label(dataset, new_data[column], power)
        dataset = dataset.copy(new_data)

    check = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.2)
    snippet = build_snippet(check, dataset_option,
                            condition_name='add_condition_feature_pps_less_than(0.2)')
    put_data_on_state(dataset)
    return check.run(dataset), snippet
