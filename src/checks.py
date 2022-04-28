import io
import json
from typing import Sequence

import streamlit as st
from deepchecks.tabular.checks import SimpleModelComparison
from deepchecks.tabular.checks.distribution import TrainTestFeatureDrift, TrainTestLabelDrift
from deepchecks.tabular.checks.integrity import StringMismatch, DataDuplicates
from deepchecks.tabular.checks.performance import SegmentPerformance
import streamlit.components.v1 as components

import run_train_test_feature_drift, run_train_test_label_drift, run_string_mismatch, run_data_duplicates, \
    run_segment_performance, run_simple_model_comparison
from datasets import get_dataset_options


__all__ = ['show_checks_page']

from encoder import AppEncoder


def get_checks_options():
    return {
        TrainTestFeatureDrift: run_train_test_feature_drift.run,
        TrainTestLabelDrift: run_train_test_label_drift.run,
        StringMismatch: run_string_mismatch.run,
        DataDuplicates: run_data_duplicates.run,
        SegmentPerformance: run_segment_performance.run,
        SimpleModelComparison: run_simple_model_comparison.run
    }


def update_query_param():
    # Set in the query params the selected check
    st.experimental_set_query_params(check=st.session_state.check_select)


def show_checks_page():
    TEMPLATE_WRAPPER = """
    <div style="height:{height}px;overflow:auto;position:relative;">
        {body}
    </div>
    """

    datasets = get_dataset_options()

    # Normal st.header can't be aligned to the center, must use html
    # st.markdown("<h1 style='text-align: center;'>Inject a corruption and see what Deepchecks would find</h1>", unsafe_allow_html=True)
    st.header('Inject a corruption and see what Deepchecks would find')

    checks = get_checks_options()
    # Translate check classes to names
    name_to_class = {check_key.name(): check_key for check_key in checks.keys()}
    # Add default option of no check selected
    NO_CHECK_SELECTED = 'No check selected'
    check_options_names = [NO_CHECK_SELECTED] + list(name_to_class.keys())

    query_params = st.experimental_get_query_params()
    # Get selected check from query params if exists
    if 'check' in query_params:
        start_check = query_params['check'][0]
    # Set default query params if not exists
    else:
        st.experimental_set_query_params(check=NO_CHECK_SELECTED)
        start_check = NO_CHECK_SELECTED

    # ========= Create the page layout =========
    check_col, check_params_col, dataset_col, manipulate_col = st.columns([1, 1, 1, 1])
    st.markdown("""---""")
    result_col, snippet_col = st.columns([2, 1])

    check_col.subheader('Select check')
    check_params_col.subheader('Select check parameters')
    dataset_col.subheader('Select dataset')
    manipulate_col.subheader('Corrupt dataset')

    with check_col:
        # select a check
        selected_check = st.selectbox('Select a check', check_options_names, key='check_select',
                                      index=check_options_names.index(start_check),
                                      on_change=update_query_param)
        if selected_check == NO_CHECK_SELECTED:
            st.subheader('Select check to start')
            return
        check_class = name_to_class[selected_check]
        check_run = checks[check_class]

    with dataset_col:
        # select a dataset
        dataset_name = st.selectbox('Select a dataset', datasets.keys())
        dataset = datasets[dataset_name]

    # Run the check
    with st.spinner('Running check'):
        check_result, snippet, download_btn_fn = check_run(dataset, check_params_col, manipulate_col)
        if isinstance(check_result, str):
            st.error(check_result)
            result_html = None
            result_value = None
        else:
            string_io = io.StringIO()
            check_result.save_as_html(string_io)
            result_html = string_io.getvalue()
            result_value = check_result.value

    with snippet_col:
        st.subheader('Run this example in your own environment')
        st.text('In order to run snippet download the data')
        download_btn_fn()
        st.code(snippet, language='python')
        with st.expander(f'Dataset "{dataset_name}" Head'):
            st.text('Showing the first 5 rows of the dataset')
            st.dataframe(dataset['train'].data.head(5))
        with st.expander(f'Check documentation (docstring)'):
            st.text(check_class.__doc__)

    result_col.subheader('Check result')

    if result_value is not None:
        with result_col:
            with st.expander('print(result.value)'):
                # If the result value is simple type (e.g. int, float, str) it can't be displayed as json
                if isinstance(result_value, (dict, Sequence)):
                    result_value = json.dumps(result_value, indent=4, sort_keys=False, cls=AppEncoder)
                    st.json(result_value)
                else:
                    st.code(str(result_value), language='python')

    if result_html:
        height_px = 1000
        html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
        with result_col:
            components.html(html, height=height_px)
