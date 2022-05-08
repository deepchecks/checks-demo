import io
import random
import sys

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from deepchecks.tabular.checks import TrainTestFeatureDrift, SingleFeatureContribution, \
    SingleFeatureContributionTrainTest
from deepchecks.tabular.suites import single_dataset_integrity, train_test_leakage, train_test_validation, \
    model_evaluation
from deepchecks.utils.strings import widget_to_html, get_random_string

from constants import NO_SUITE_SELECTED, SUITE_STATE_ID
from corruptions import insert_numerical_drift, insert_categorical_drift, relate_column_to_label, insert_variants, \
    insert_duplicates
from datasets import get_dataset_options, DatasetOption
from streamlit_persist import persist
from utils import st_redirect, add_download_button, build_suite_snippet

suites = {
    'Single Dataset Integrity Suite': {'suite': single_dataset_integrity, 'is_train_test': False},
    'Train-Test Leakage Suite': {'suite': train_test_leakage, 'is_train_test': True},
    'Train-Test Validation Suite': {'suite': train_test_validation, 'is_train_test': True},
    'Model Evaluation Suite': {'suite': model_evaluation, 'is_train_test': True},
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

    suite_func = suites[selected_suite]['suite']
    is_train_test = suites[selected_suite]['is_train_test']

    dataset_name = st.sidebar.selectbox('Select a dataset', datasets.keys())
    dataset_opt: DatasetOption = datasets[dataset_name]
    corruptions_cols = set_corruption_columns(dataset_name, dataset_opt)

    # Corruptions
    corruptions_container = st.sidebar.container()
    run_suite = st.sidebar.button('Run suite')
    if not run_suite:
        st.subheader('Corrupt the data and run suite on it')
        st.markdown('Applied corruptions will be shown below. To run the suite click on the "Run Suite" button'
                    'in the sidebar.')
        st.markdown('The corruptions are applied on random features, separately for each corruption. Only the top'
                    ' corruptions are displayed below.')
        add_corruptions_to_test(dataset_opt, corruptions_cols, corruptions_container, show_diff=True)
    else:
        corrupt_dataset = add_corruptions_to_test(dataset_opt, corruptions_cols, corruptions_container, show_diff=False)
        # Create layout
        suite_col, snippet_col = st.columns([2, 1])

        with suite_col:
            if run_suite:
                with st.spinner(f'Running {selected_suite} on {dataset_name}'):
                    # Create placeholder to show stuff while running
                    placeholder = st.empty()
                    # placeholder.markdown('This may take a while... In the meantime here is a cute monkey<br>'
                    #                      '<img src="https://wallpaperaccess.com/full/1137839.jpg" alt="drawing" width="400"/>'
                    #                      , unsafe_allow_html=True)
                    placeholder.markdown('This may take a while... In the meantime here is a cute CEO <br>'
                                         '<img src="https://media-exp1.licdn.com/dms/image/C4D03AQFhBQ0CgfTgDg/profile-displayphoto-shrink_800_800/0/1600202016336?e=1657756800&v=beta&t=dEqExsbmlLbT1kOSq4U5aDE2pvCHEoelnc8KEmrU82A" alt="drawing" width="400"/>',
                                         unsafe_allow_html=True)

                    with st_redirect(sys.stdout, 'code'):
                        if is_train_test:
                            suite_params = {'train_dataset': dataset_opt.train, 'test_dataset': corrupt_dataset}
                        else:
                            suite_params = {'train_dataset': corrupt_dataset}
                        suite_instance = suite_func()
                        result = suite_instance.run(**suite_params,
                                                    model=dataset_opt.model,
                                                    features_importance=dataset_opt.features_importance)

                    result_html = save_suite_result(result, suite_instance.name)
                    placeholder.empty()

                height_px = 1200
                html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
                components.html(html, height=height_px)

        with snippet_col:
            st.subheader('Run this example in your own environment')
            st.markdown('In order to run the snippet, download the data and change the paths accordingly. '
                        'The data you download will correspond to the latest corruptions applied.')
            if is_train_test:
                data_to_download = (dataset_opt.train, corrupt_dataset)
            else:
                data_to_download = (corrupt_dataset,)
            add_download_button(data_to_download)
            suite_snippet = build_suite_snippet(suite_func, dataset_opt, is_train_test)
            st.code(suite_snippet, language='python')

    footnote = """
    <br><br>
    **Notes**: 
    1. For suites that involve train and test, corruption is applied to the test set.
    2. Due to limitations of Streamlit, some checks may be cropped on small screens. In this case, please run the check on your own environment using the code on the right.
    <br><br>
    If you liked this, please ⭐&nbsp;us on [GitHub](https://github.com/deepchecks/deepchecks)<br>
    For more info, check out our [docs](https://docs.deepchecks.com/stable/)
    """
    st.sidebar.markdown(footnote, unsafe_allow_html=True)


def set_corruption_columns(name, dataset_opt: DatasetOption):
    if 'corrupt_columns' not in st.session_state:
        st.session_state['corrupt_columns'] = {}
    if name not in st.session_state['corrupt_columns']:
        st.session_state['corrupt_columns'][name] = {
            'drift': get_random_features(dataset_opt.test.features),
            'label_correlation': get_random_features(dataset_opt.test.numerical_features),
            'string_variants': get_random_features(dataset_opt.test.cat_features),
        }
        get_random_features(dataset_opt.test.numerical_features)
    return st.session_state['corrupt_columns'][name]


def add_corruptions_to_test(dataset_opt: DatasetOption, columns, corruptions_container, show_diff: bool):
    # Show corruptions controls
    corruptions_container.header('Corruptions')
    MAX_DRIFT_POWER = 1.
    drift_power = corruptions_container.slider('Drift power', 0., MAX_DRIFT_POWER, step=0.1, value=0.)
    column_label_relation_power = corruptions_container.slider('Column label relation power', 0., 3., step=0.1, value=0.)
    if dataset_opt.contain_categorical_columns:
        percent_string_variants = corruptions_container.slider('Percent string variants', 0, 100, step=1, value=0)
    else:
        percent_string_variants = 0
    percent_duplicate = corruptions_container.slider('Percent duplicate', 0, 100, step=1, value=0)

    # If need to show corruptions diff, create layout
    drift_col, label_col, string_variants_col, duplicates_col = st.columns(4) if show_diff else [None] * 4
    if show_diff:
        drift_col.subheader('Drift')
        label_col.subheader('Label correlation')
        string_variants_col.subheader('String variants')
        duplicates_col.subheader('Duplicates rows')

    # Apply corruptions on data
    corrupt_data = dataset_opt.test.data.copy()

    # Drift
    if drift_power > 0:
        for feature in columns['drift']:
            col = corrupt_data[feature].to_numpy()
            if feature in dataset_opt.test.numerical_features:
                mean = np.mean(col) * drift_power
                std = np.std(col) * drift_power
                corrupt_data[feature] = insert_numerical_drift(col, mean, std)
            elif feature in dataset_opt.test.cat_features:
                random_value = np.random.choice(col)
                ratio = np.count_nonzero(col == random_value) / col.shape[0]
                ratio_for_drift = (1 - ratio) * drift_power / MAX_DRIFT_POWER
                percent = min((ratio + ratio_for_drift) * 100, 100)
                corrupt_data[feature] = insert_categorical_drift(col, percent, random_value)
        drift_graph(dataset_opt.test, corrupt_data, columns['drift'], drift_col)

    if column_label_relation_power > 0:
        # We know to relate only numerical features to label
        for feature in columns['label_correlation']:
            corrupt_data[feature] = relate_column_to_label(dataset_opt.test, corrupt_data[feature], column_label_relation_power)
        label_correlation_graph(dataset_opt.test, corrupt_data, columns['label_correlation'], label_col)

    if percent_string_variants > 0:
        for feature in columns['string_variants']:
            corrupt_data[feature] = insert_variants(corrupt_data[feature], percent_string_variants)

    if percent_duplicate > 0:
        corrupt_data = insert_duplicates(corrupt_data, rows_to_duplicate_num=1, percent=percent_duplicate)

    # Return test dataset with new data
    return dataset_opt.test.copy(corrupt_data)


def get_random_features(features):
    if not features:
        return []
    num_features = random.randint(1, len(features))
    return np.random.choice(features, size=num_features, replace=False)


def save_suite_result(result, header):
    string_io = io.StringIO()

    widget_to_html(
        result.to_widget(
            unique_id=None,
        ),
        html_out=string_io,
        title=header,
    )

    return string_io.getvalue()


def drift_graph(origin_dataset, corrupted_df, columns, container):
    if container is None:
        return
    corrupted_dataset = origin_dataset.copy(corrupted_df)
    result = TrainTestFeatureDrift(columns=list(columns), n_top_columns=3).run(origin_dataset, corrupted_dataset)
    for fig in result.display[1:]:
        container.plotly_chart(fig, use_container_width=True)


def label_correlation_graph(origin_dataset, corrupted_df, columns, container):
    if container is None:
        return
    corrupted_dataset = origin_dataset.copy(corrupted_df)
    result = SingleFeatureContributionTrainTest(columns=list(columns)).run(origin_dataset, corrupted_dataset)
    container.plotly_chart(result.display[0], use_container_width=True)


def string_variants_graph(origin_dataset, corrupted_df, columns, container):
    if container is None:
        return


def duplicate_data_graph(origin_dataset, corrupted_df, columns, container):
    if container is None:
        return
