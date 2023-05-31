import io
import json
from typing import Sequence, TypedDict, Callable, Any

import npdoc_to_md
import streamlit as st
from deepchecks.tabular.checks import SimpleModelComparison, FeatureLabelCorrelation
from deepchecks.tabular.checks import TrainTestFeatureDrift, TrainTestLabelDrift
from deepchecks.tabular.checks import StringMismatch, DataDuplicates
from deepchecks.tabular.checks import SegmentPerformance
import streamlit.components.v1 as components

import run_train_test_feature_drift, run_train_test_label_drift, run_string_mismatch, run_data_duplicates, \
    run_segment_performance, run_simple_model_comparison, run_single_feature_contribution_train_test
from constants import NO_CHECK_SELECTED, CHECK_STATE_ID, DATA_STATE_ID
from datasets import get_dataset_options


__all__ = ['show_checks_page']

from encoder import AppEncoder
from streamlit_persist import persist
from utils import add_download_button


class CheckOption(TypedDict):
    type: str
    class_var: Any
    run_function: Callable


def get_checks_options():
    return [
        CheckOption(type='distribution', class_var=TrainTestFeatureDrift,
                    run_function=run_train_test_feature_drift.run),
        CheckOption(type='distribution', class_var=TrainTestLabelDrift,
                    run_function=run_train_test_label_drift.run),
        CheckOption(type='integrity', class_var=StringMismatch,
                    run_function=run_string_mismatch.run),
        CheckOption(type='integrity', class_var=DataDuplicates,
                    run_function=run_data_duplicates.run),
        CheckOption(type='performance', class_var=SegmentPerformance,
                    run_function=run_segment_performance.run),
        CheckOption(type='performance', class_var=SimpleModelComparison,
                    run_function=run_simple_model_comparison.run),
        CheckOption(type='methodology', class_var=FeatureLabelCorrelation,
                    run_function=run_single_feature_contribution_train_test.run),
    ]


def show_checks_page():
    TEMPLATE_WRAPPER = """
    <div style="height:{height}px;overflow-y:auto;position:relative;">
        {body}
    </div>
    """

    datasets = get_dataset_options()
    checks: list = get_checks_options()
    # Translate check classes to names
    name_to_check_opt = {f'{check_opt["class_var"].name()} ({check_opt["type"]})': check_opt
                         for index, check_opt in enumerate(checks)}
    # Add default option of no check selected
    check_options_names = [NO_CHECK_SELECTED] + list(name_to_check_opt.keys())

    # select a check
    selected_check = st.sidebar.selectbox('Select a check', check_options_names, key=persist(CHECK_STATE_ID))
    st.sidebar.markdown('These are just a few of the checks deepchecks offers, full list in the '
                        '[gallery](https://docs.deepchecks.com/stable/checks_gallery/tabular/index.html?utm_campaign=gallery_button&utm_medium=referral&utm_source=checks-demo.deepchecks.com)',
                        unsafe_allow_html=True)
    if selected_check == NO_CHECK_SELECTED:
        return

    # ========= Create the page layout =========
    st.markdown('# Inject a Corruption and See What Deepchecks Would Find \n'
                '\* This demo focuses on individual checks. To get a feel for test suites, we recommend running the example from the '
                '[Quickstart](https://docs.deepchecks.com/stable/auto_tutorials/tabular/plot_quickstart_in_5_minutes.html?utm_campaign=quickstart_button&utm_medium=referral&utm_source=checks-demo.deepchecks.com)'
                ' on your environment.')

    result_col, snippet_col = st.columns([2, 1])

    # select a dataset
    # For check "string mismatch" we need only datasets that contains categorical features
    if StringMismatch.name() in selected_check:
        datasets = {name: dataset for name, dataset in datasets.items() if dataset.contain_categorical_columns}
    dataset_name = st.sidebar.selectbox('Select a dataset', datasets.keys())
    dataset = datasets[dataset_name]

    st.sidebar.subheader('Check\'s Parameters')
    check_params_col = st.sidebar.container()
    manipulate_col = st.sidebar.container()
    # Run the check
    with st.spinner('Running check'):
        check_opt = name_to_check_opt[selected_check]
        check_run = check_opt['run_function']
        check_result, snippet = check_run(dataset, check_params_col, manipulate_col)
        string_io = io.StringIO()
        check_result.save_as_html(string_io)
        result_html = string_io.getvalue()
        result_value = check_result.value

    with snippet_col:
        st.subheader('Run this example in your own environment')
        st.markdown('In order to run the snippet, download the data and change the paths accordingly. '
                    'The data you download will correspond to the latest corruptions applied.')
        add_download_button()
        st.code(snippet, language='python')
        if result_value is not None:
            with st.expander('print(result.value)'):
                # If the result value is simple type (e.g. int, float, str) it can't be displayed as json
                if isinstance(result_value, (dict, Sequence)):
                    result_value = json.dumps(result_value, indent=4, sort_keys=False, cls=AppEncoder)
                    st.json(result_value)
                else:
                    st.code(str(result_value), language='python')
        data_state = st.session_state[DATA_STATE_ID]
        with st.expander(f'Dataset "{dataset_name}" Head', expanded=True):
            st.markdown(f'Showing the first 5 rows of the {data_state["dataset_type"]}')
            st.dataframe(data_state['data'][data_state['corrupted_dataset_index']].head(5))
        with st.expander(f'Documentation of the Check (docstring)'):
            check_class = check_opt['class_var']
            docs_md = npdoc_to_md.render_md_from_obj_docstring(check_class, check_class.__name__)
            st.markdown(docs_md, unsafe_allow_html=True)

    result_col.subheader(selected_check)

    if result_html:
        height_px = 1000
        html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
        with result_col:
            components.html(html, height=height_px)

    footnote = """
    <br><br>
    **Note**: 
    Due to limitations of Streamlit, some checks may be cropped on small screens. In this case, please run the check on your own environment using the code on the right.
    <br><br>
    If you liked this, please ⭐&nbsp;us on [GitHub](https://github.com/deepchecks/deepchecks)<br>
    For more info, check out our [docs](https://docs.deepchecks.com/stable?utm_campaign=docs_button&utm_medium=referral&utm_source=checks-demo.deepchecks.com)
    """
    st.sidebar.markdown(footnote, unsafe_allow_html=True)
