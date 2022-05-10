import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import SingleFeatureContributionTrainTest

from corruptions import relate_column_to_label
from datasets import DatasetOption
from utils import build_snippet, put_data_on_state


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    train_dataset: Dataset = dataset_option.train
    new_data = train_dataset.data.copy()

    with check_param_col:
        st.text('No parameters to configure')
    with manipulate_col:
        st.subheader('Add Corruption to Train Data')
        # Allow manipulation only for numeric columns
        column: str = st.selectbox('Select a column', train_dataset.numerical_features)
        power = st.slider('Label correlation power', min_value=0., max_value=10., value=1., step=0.1)

    if power > 0:
        new_data[column] = relate_column_to_label(train_dataset, new_data[column], power)
        train_dataset = train_dataset.copy(new_data)

    check = SingleFeatureContributionTrainTest().add_condition_feature_pps_difference_not_greater_than(0.2)
    snippet = build_snippet(check, dataset_option,
                            condition_name='add_condition_feature_pps_difference_not_greater_than(0.2)')
    put_data_on_state(train_dataset, dataset_option.test, dataset_type='train')
    return check.run(train_dataset, dataset_option.test), snippet
