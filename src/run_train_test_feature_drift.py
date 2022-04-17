import math

import streamlit as st
import numpy as np

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestFeatureDrift

from datasets import DatasetOption
from utils import insert_categorical_drift, insert_numerical_drift


def run(dataset_option: DatasetOption):
    dataset: Dataset = dataset_option['test']
    new_data = dataset.data.copy()
    # Show column selector
    columns = dataset.numerical_features + dataset.cat_features
    column: str = st.sidebar.selectbox('Select a column', columns)

    st.sidebar.subheader(f'Manipulate column "{column}"')
    # Allow numeric drift
    if column in dataset.numerical_features:
        mean = st.sidebar.slider('Mean', min_value=0.0, max_value=3.0, step=0.1)
        std = st.sidebar.slider('Std', min_value=0.0, max_value=3.0, step=0.1)

        if mean > 0 or std > 0:
            new_data[column] = insert_numerical_drift(new_data[column], mean, std)
    # Allow categorical drift
    else:
        category_to_drift = st.sidebar.selectbox('Select a category to drift', dataset.data[column].unique())

        # Calc current ratio
        category_ratio = np.count_nonzero(new_data[column] == category_to_drift) / new_data[column].shape[0]
        category_percent = math.floor(category_ratio * 10_000)/100.0

        percent_in_data = st.sidebar.slider('Percent in test data', 0.0, 100.0, value=category_percent)
        if percent_in_data != category_percent:
            new_data[column] = insert_categorical_drift(new_data[column], percent_in_data, category_to_drift)

    check = TrainTestFeatureDrift(columns=[column]).add_condition_drift_score_not_greater_than()
    return check.run(dataset_option['train'], dataset.copy(new_data))
