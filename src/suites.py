import io

import streamlit as st
import streamlit.components.v1 as components
from deepchecks.tabular.suites import single_dataset_integrity

from constants import NO_SUITE_SELECTED, SUITE_STATE_ID, SUITE_QUERY_PARAM
from datasets import get_dataset_options
from streamlit_persist import persist
from utils import set_query_param

suites = {
    'Integrity Suite': single_dataset_integrity()
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
    dataset = datasets[dataset_name]

    suite_instance = suites[selected_suite]
    with st.spinner(f'Running {selected_suite} on {dataset_name}'):
        result = suite_instance.run(train_dataset=dataset['train'], test_dataset=dataset['test'], model=dataset['model'],
                                    features_importance=dataset['features_importance'])
        string_io = io.StringIO()
        result.save_as_html(string_io)
        result_html = string_io.getvalue()

    height_px = 1200
    html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
    components.html(html, height=height_px)
