import math

import streamlit as st
import numpy as np

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestFeatureDrift

from datasets import DatasetOption
from utils import build_snippet, std_without_outliers, put_data_on_state
from corruptions import insert_numerical_drift, insert_categorical_drift


def run(dataset_option: DatasetOption, check_param_col, manipulate_col):
    test_dataset: Dataset = dataset_option.test
    new_data = test_dataset.data.copy()
    # Show column selector
    with check_param_col:
        columns = test_dataset.numerical_features + test_dataset.cat_features
        column: str = st.selectbox('Select a column', columns)

    with manipulate_col:
        st.subheader('Add Corruption to Test Data')
        # Allow numeric drift
        if column in test_dataset.numerical_features:
            col_std = std_without_outliers(new_data[column])
            st.text('Add gaussian noise')
            mean = st.slider('Mean', min_value=0.0, max_value=col_std * 5, step=col_std / 20)
            std = st.slider('Std', min_value=0.0, max_value=col_std * 5, step=col_std / 20)

            if mean > 0 or std > 0:
                new_data[column] = insert_numerical_drift(new_data[column], mean, std)

        # Allow categorical drift
        else:
            category_to_drift = st.selectbox('Select a category to drift', test_dataset.data[column].unique())

            # Calc current ratio
            category_ratio = np.count_nonzero(new_data[column] == category_to_drift) / new_data[column].shape[0]
            category_percent = math.floor(category_ratio * 10_000)/100.0

            percent_in_data = st.slider('Percent in test data', 0.0, 100.0, value=category_percent)
            if percent_in_data != category_percent:
                new_data[column] = insert_categorical_drift(new_data[column], percent_in_data, category_to_drift)

    check_props = {'columns': [column], 'show_categories_by': 'largest_difference'}
    check = TrainTestFeatureDrift(**check_props).add_condition_drift_score_not_greater_than()
    snippet = build_snippet(check, dataset_option, properties=check_props,
                            condition_name='add_condition_drift_score_not_greater_than(max_allowed_psi_score = '
                                           '0.2, max_allowed_earth_movers_score = 0.1)')

    test_dataset = test_dataset.copy(new_data)

    put_data_on_state(dataset_option.train, test_dataset, dataset_type='test', corrupted_dataset_index=1)
    return check.run(dataset_option.train, test_dataset), snippet
