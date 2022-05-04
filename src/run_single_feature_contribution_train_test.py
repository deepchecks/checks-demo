import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import SingleFeatureContributionTrainTest

from corruptions import relate_column_to_label
from datasets import DatasetOption
from utils import build_snippet


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    test_dataset: Dataset = dataset_option.test
    new_data = test_dataset.data.copy()

    with check_param_col:
        st.text('No parameters to configure')
    with manipulate_col:
        # Allow manipulation only for numeric columns
        column: str = st.selectbox('Select a column', test_dataset.numerical_features)
        power = st.slider('Label correlation power', min_value=0., max_value=10., value=1., step=0.1)

    if power > 0:
        new_data[column] = relate_column_to_label(test_dataset, new_data[column], power)
        test_dataset = test_dataset.copy(new_data)

    check = SingleFeatureContributionTrainTest().add_condition_feature_pps_in_train_not_greater_than(0.7)
    snippet = build_snippet(check, dataset_option,
                            condition_name='add_condition_feature_pps_in_train_not_greater_than(0.7)')
    return check.run(dataset_option.train, test_dataset), snippet, (dataset_option.train, test_dataset)
