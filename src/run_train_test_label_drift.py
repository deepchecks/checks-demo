import math

import numpy as np
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestLabelDrift

from datasets import DatasetOption
from utils import build_snippet, std_without_outliers, put_data_on_state
from corruptions import insert_numerical_drift, insert_categorical_drift


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    test_dataset: Dataset = dataset_option.test
    new_data = test_dataset.data.copy()
    label_name = test_dataset.label_name

    with check_param_col:
        st.text('No parameters to control')
    with manipulate_col:
        st.subheader('Add Corruption to Test Data')
        # Allow numeric drift
        if test_dataset.label_type == 'regression_label':
            col_std = std_without_outliers(new_data[label_name])
            st.text('Add gaussian noise')
            mean = st.slider('Mean', min_value=0.0, max_value=col_std * 5, step=col_std / 20, value=col_std / 2)
            std = st.slider('Std', min_value=0.0, max_value=col_std * 5, step=col_std / 20, value=0.0)

            if mean > 0 or std > 0:
                new_data[label_name] = insert_numerical_drift(new_data[label_name], mean, std)
        elif test_dataset.label_type == 'classification_label':
            category_to_drift = st.selectbox('Select a category to drift', new_data[label_name].unique())

            # Calc current ratio
            category_ratio = np.count_nonzero(new_data[label_name] == category_to_drift) / new_data[label_name].shape[0]
            category_percent = math.floor(category_ratio * 10_000)/100.0

            percent_in_data = st.slider('Percent in test data', 0.0, 100.0, value=category_percent)
            if category_percent != percent_in_data:
                new_data[label_name] = insert_categorical_drift(new_data[label_name], percent_in_data, category_to_drift)

    check = TrainTestLabelDrift().add_condition_drift_score_less_than()
    snippet = build_snippet(check, dataset_option,
                            condition_name='add_condition_drift_score_less_than(max_allowed_drift_score = 0.15)')
    test_dataset = test_dataset.copy(new_data)
    put_data_on_state(dataset_option.train, test_dataset, dataset_type='test', corrupted_dataset_index=1)
    return check.run(dataset_option.train, test_dataset), snippet
