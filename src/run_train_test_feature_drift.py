import math

import streamlit as st
import numpy as np

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestFeatureDrift

from datasets import DatasetOption
from utils import insert_categorical_drift, insert_numerical_drift, build_snippet, add_download_button


def run(dataset_option: DatasetOption):
    test_dataset: Dataset = dataset_option['test']
    new_data = test_dataset.data.copy()
    # Show column selector
    st.sidebar.subheader(f'Check Parameters')
    columns = test_dataset.numerical_features + test_dataset.cat_features
    column: str = st.sidebar.selectbox('Select a column', columns)

    st.sidebar.subheader(f'Manipulate column "{column}"')
    # Allow numeric drift
    if column in test_dataset.numerical_features:
        mean = st.sidebar.slider('Mean', min_value=0.0, max_value=3.0, step=0.1)
        std = st.sidebar.slider('Std', min_value=0.0, max_value=3.0, step=0.1)

        if mean > 0 or std > 0:
            new_data[column] = insert_numerical_drift(new_data[column], mean, std)
    # Allow categorical drift
    else:
        category_to_drift = st.sidebar.selectbox('Select a category to drift', test_dataset.data[column].unique())

        # Calc current ratio
        category_ratio = np.count_nonzero(new_data[column] == category_to_drift) / new_data[column].shape[0]
        category_percent = math.floor(category_ratio * 10_000)/100.0

        percent_in_data = st.sidebar.slider('Percent in test data', 0.0, 100.0, value=category_percent)
        if percent_in_data != category_percent:
            new_data[column] = insert_categorical_drift(new_data[column], percent_in_data, category_to_drift)

    check = TrainTestFeatureDrift(columns=[column]).add_condition_drift_score_not_greater_than()
    snippet = build_snippet(check, dataset_option, properties={'columns': [column]},
                            condition_name='add_condition_drift_score_not_greater_than')

    test_dataset = test_dataset.copy(new_data)
    add_download_button(dataset_option['train'], test_dataset)

    return check.run(dataset_option['train'], test_dataset), snippet
