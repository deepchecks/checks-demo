import io
import json
from typing import Sequence

import npdoc_to_md
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
from utils import add_download_button


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


START_PAGE_MD = """
# Welcome to Deepchecks' Interactive Checks Demo 🚀

In this demo you can play with some of the existing checks and see how they work on various datasets. <br/>
Each check enables custom corruptions to the dataset to showcase its value. 

If you like what we're doing at Deepchecks, please ⭐&nbsp;us on [GitHub](https://github.com/deepchecks/deepchecks).<br/>
And if you'd like to dive in a bit more, check out our [documentation](https://docs.deepchecks.com/stable/).

### ⬅️ To start select a check on the left sidebar

![](https://docs.deepchecks.com/stable/_images/checks_and_conditions.png)
"""


def show_checks_page():
    TEMPLATE_WRAPPER = """
    <div style="height:{height}px;overflow-y:auto;position:relative;">
        {body}
    </div>
    """

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

    # select a check
    selected_check = st.sidebar.selectbox('Select a check', check_options_names, key='check_select',
                                          index=check_options_names.index(start_check),
                                          on_change=update_query_param)
    st.sidebar.markdown('Those are just a few of all the checks deepchecks offers, see '
                        '[gallery](https://docs.deepchecks.com/stable/checks_gallery/tabular/index.html)',
                        unsafe_allow_html=True)
    if selected_check == NO_CHECK_SELECTED:
        st.markdown(START_PAGE_MD, unsafe_allow_html=True)
        return

    # ========= Create the page layout =========
    st.header('Inject a Corruption and See What Deepchecks Would Find')
    result_col, snippet_col = st.columns([2, 1])

    # select a dataset
    # For check "string mismatch" we need only datasets that contains categorical features
    if selected_check == StringMismatch.name():
        datasets = {name: dataset for name, dataset in datasets.items() if dataset['contain_categorical_columns']}
    dataset_name = st.sidebar.selectbox('Select a dataset', datasets.keys())
    dataset = datasets[dataset_name]

    st.sidebar.subheader('Check Parameters')
    check_params_col = st.sidebar.container()
    st.sidebar.subheader('Add Corruption')
    manipulate_col = st.sidebar.container()
    # Run the check
    with st.spinner('Running check'):
        check_class = name_to_class[selected_check]
        check_run = checks[check_class]
        check_result, snippet, dataset_tuple = check_run(dataset, check_params_col, manipulate_col)
        string_io = io.StringIO()
        check_result.save_as_html(string_io)
        result_html = string_io.getvalue()
        result_value = check_result.value

    with snippet_col:
        st.subheader('Run this example in your own environment')
        st.markdown('In order to run the snippet, download the data and change the paths accordingly. '
                    'The data you download will correspond to the latest corruptions applied.')
        add_download_button(dataset_tuple)
        st.code(snippet, language='python')
        if result_value is not None:
            with st.expander('print(result.value)'):
                # If the result value is simple type (e.g. int, float, str) it can't be displayed as json
                if isinstance(result_value, (dict, Sequence)):
                    result_value = json.dumps(result_value, indent=4, sort_keys=False, cls=AppEncoder)
                    st.json(result_value)
                else:
                    st.code(str(result_value), language='python')
        with st.expander(f'Dataset "{dataset_name}" Head', expanded=True):
            dataset_name = 'dataset' if len(dataset_tuple) == 1 else 'test dataset'
            st.markdown(f'Showing the first 5 rows of the {dataset_name}')
            # If we have single dataset show it, if we have 2 datasets show the last one which is test dataset
            st.dataframe(dataset_tuple[-1].data.head(5))
        with st.expander(f'Documentation of the Check (docstring)'):
            namespace = check_class.__name__
            docs_md = npdoc_to_md.render_md_from_obj_docstring(check_class, namespace)
            st.markdown(docs_md, unsafe_allow_html=True)

    result_col.subheader('Check result')

    if result_html:
        height_px = 1000
        html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
        with result_col:
            components.html(html, height=height_px)

    footnote = """
    <br><br>
    **Notes**: 
    1. For checks that involve 2 datasets, corruption is applied to the test set.
    2. Due to limitations of Streamlit, some checks may be cropped on small screens. In this case, please run the check on your own environment using the code on the right.
    <br><br>
    If you liked this, please ⭐&nbsp;us on [GitHub](https://github.com/deepchecks/deepchecks)<br>
    For more info, check out our [docs](https://docs.deepchecks.com/stable/)
    """
    st.sidebar.markdown(footnote, unsafe_allow_html=True)
