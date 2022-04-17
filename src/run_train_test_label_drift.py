import math

import numpy as np
import streamlit as st
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestLabelDrift

from datasets import DatasetOption
from utils import insert_categorical_drift, insert_numerical_drift


def run(dataset_option: DatasetOption):
    dataset: Dataset = dataset_option['test']
    new_data = dataset.data.copy()
    label_name = dataset.label_name

    st.sidebar.subheader(f'Manipulate label "{label_name}"')

    # Allow numeric drift
    if dataset.label_type == 'regression_label':
        mean = st.sidebar.slider('Mean', min_value=0.0, max_value=3.0, step=0.1)
        std = st.sidebar.slider('Std', min_value=0.0, max_value=3.0, step=0.1)

        if mean > 0 or std > 0:
            new_data[label_name] = insert_numerical_drift(new_data[label_name], mean, std)
    elif dataset.label_type == 'classification_label':
        category_to_drift = st.sidebar.selectbox('Select a category to drift', new_data[label_name].unique())

        # Calc current ratio
        category_ratio = np.count_nonzero(new_data[label_name] == category_to_drift) / new_data[label_name].shape[0]
        category_percent = math.floor(category_ratio * 10_000)/100.0

        percent_in_data = st.sidebar.slider('Percent in test data', 0.0, 100.0, value=category_percent)
        if category_percent != percent_in_data:
            new_data[label_name] = insert_categorical_drift(new_data[label_name], percent_in_data, category_to_drift)

    check = TrainTestLabelDrift().add_condition_drift_score_not_greater_than()
    return check.run(dataset_option['train'], dataset.copy(new_data))

