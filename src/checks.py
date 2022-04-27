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
    <div style="height:{height}px;overflow-y:auto;position:relative;">
        {body}
    </div>
    """

    col1, col2 = st.columns(2)

    with st.spinner('Loading datasets...'):
        datasets = get_dataset_options()

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

    # select a dataset and check
    dataset_name = st.sidebar.selectbox('Select a dataset', datasets.keys())
    selected_check = st.sidebar.selectbox('Select a check', check_options_names, key='check_select',
                                          index=check_options_names.index(start_check), on_change=update_query_param)

    if selected_check == NO_CHECK_SELECTED:
        st.title('Select check to start')
        return

    dataset = datasets[dataset_name]
    check_class = name_to_class[selected_check]
    check_run = checks[check_class]

    # Run the check
    with col1:
        with st.spinner('Running check'):
            check_result, snippet = check_run(dataset)
            if isinstance(check_result, str):
                with col1:
                    st.error(check_result)
                result_html = None
                result_value = None
            else:
                string_io = io.StringIO()
                check_result.save_as_html(string_io)
                result_html = string_io.getvalue()
                result_value = check_result.value

        st.code(snippet, language='python')

        if result_value is not None:
            with st.expander('print(result.value)'):
                # If the result value is simple type (e.g. int, float, str) it can't be displayed as json
                if isinstance(result_value, (dict, Sequence)):
                    result_value = json.dumps(result_value, indent=4, sort_keys=False, cls=AppEncoder)
                    st.json(result_value)
                else:
                    st.code(str(result_value), language='python')

        if result_html:
            height_px = 800
            html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
            components.html(html, height=height_px)

    with col2:
        st.subheader(f'Check {check_class.__name__} docstring')
        st.text(check_class.__doc__)
