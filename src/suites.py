import io
import random

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import single_dataset_integrity, train_test_leakage, train_test_validation, \
    model_evaluation

from constants import NO_SUITE_SELECTED, SUITE_STATE_ID
from corruptions import insert_numerical_drift, insert_categorical_drift, relate_column_to_label, insert_variants, \
    insert_duplicates
from datasets import get_dataset_options, DatasetOption
from streamlit_persist import persist

suites = {
    'Integrity Suite': single_dataset_integrity(),
    'Train-Test Leakage Suite': train_test_leakage(),
    'Train-Test Validation Suite': train_test_validation(),
    'Model Evaluation Suite': model_evaluation(),
}


def show_suites_page():
    TEMPLATE_WRAPPER = """
    <div style="height:{height}px;overflow-y:auto;position:relative;">
        {body}
    </div>
    """

    datasets = get_dataset_options()
    suites_options_names = [NO_SUITE_SELECTED] + list(suites.keys())

    # select a check
    selected_suite = st.sidebar.selectbox('Select a suite', suites_options_names, key=persist(SUITE_STATE_ID))

    if selected_suite == NO_SUITE_SELECTED:
        return

    dataset_name = st.sidebar.selectbox('Select a dataset', datasets.keys())
    dataset_opt: DatasetOption = datasets[dataset_name]

    # Corruptions
    corruptions_container = st.sidebar.container()
    with corruptions_container:
        corrupt_dataset = add_corruptions_to_test(dataset_opt)
        st.markdown('Corruptions are applied on random features, separately for each corruption.')

    suite_instance = suites[selected_suite]
    with st.spinner(f'Running {selected_suite} on {dataset_name}'):
        result = suite_instance.run(train_dataset=dataset_opt.train, test_dataset=corrupt_dataset, model=dataset_opt.model,
                                    features_importance=dataset_opt.features_importance)
        string_io = io.StringIO()
        result.save_as_html(string_io)
        result_html = string_io.getvalue()

    height_px = 1200
    html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
    components.html(html, height=height_px)


def add_corruptions_to_test(dataset_opt: DatasetOption):
    st.header('Corruptions')
    MAX_DRIFT_POWER = 3.
    drift_power = st.slider('Drift power', 0., MAX_DRIFT_POWER, step=0.1, value=0.)
    column_label_relation_power = st.slider('Column label relation power', 0., 10., step=0.1, value=0.)
    if dataset_opt.contain_categorical_columns:
        percent_string_variants = st.slider('Percent string variants', 0, 100, step=1, value=0)
    else:
        percent_string_variants = 0
    percent_duplicate = st.slider('Percent duplicate', 0, 100, step=1, value=0)

    test_data = dataset_opt.test.data.copy()

    if drift_power > 0:
        for feature in get_random_features(dataset_opt.test.features):
            col = test_data[feature].to_numpy()
            if feature in dataset_opt.test.numerical_features:
                mean = np.mean(col) * drift_power
                std = np.std(col) * drift_power
                test_data[feature] = insert_numerical_drift(col, mean, std)
            elif feature in dataset_opt.test.cat_features:
                random_value = np.random.choice(col)
                ratio = np.count_nonzero(col == random_value) / col.shape[0]
                ratio_for_drift = (1 - ratio) * drift_power / MAX_DRIFT_POWER
                percent = max((ratio + ratio_for_drift) * 100, 100)
                test_data[feature] = insert_categorical_drift(col, percent, random_value)

    if column_label_relation_power > 0:
        # We know to relate only numerical features to label
        for feature in get_random_features(dataset_opt.test.numerical_features):
            test_data[feature] = relate_column_to_label(dataset_opt.test, test_data[feature], column_label_relation_power)

    if percent_string_variants > 0:
        for feature in get_random_features(dataset_opt.test.cat_features):
            test_data[feature] = insert_variants(test_data[feature], percent_string_variants)

    if percent_duplicate > 0:
        test_data = insert_duplicates(test_data, rows_to_duplicate_num=1, percent=percent_duplicate)

    # Return test dataset with new data
    return dataset_opt.test.copy(test_data)


def get_random_features(features):
    num_features = random.randint(1, len(features))
    return np.random.choice(features, size=num_features, replace=False)
