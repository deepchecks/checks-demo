import io
import json
from pathlib import Path
from typing import Sequence

from PIL import Image

import streamlit as st
import streamlit.components.v1 as components

from analytics import inject_ga
from datasets import get_dataset_options
from checks import get_checks_options
from encoder import AppEncoder
from dotenv import load_dotenv


# Add google analytics
load_dotenv()
inject_ga()

TEMPLATE_WRAPPER = """
<div style="height:{height}px;overflow-y:auto;position:relative;">
    <style>
        table, th, td {{
            border: 1px solid;
        }}
    </style>
    {body}
</div>
"""

icon = Image.open(Path(__file__).parent.parent / 'resources' / 'favicon.ico')
logo = Image.open(Path(__file__).parent.parent / 'resources' / 'deepchecks_logo.png')

st.set_page_config(page_title='Deepchecks Demo', page_icon=icon, layout='wide')
st.sidebar.image(logo, use_column_width=True)
col1, col2 = st.columns(2)

with st.spinner('Loading datasets...'):
    datasets = get_dataset_options()

checks = get_checks_options()
name_to_class = {check_key.name(): check_key for check_key in checks.keys()}

# First select a dataset and check
dataset_name = st.sidebar.selectbox('Select a dataset', datasets.keys())
check_name = st.sidebar.selectbox('Select a check', name_to_class.keys())

dataset = datasets[dataset_name]
check_class = name_to_class[check_name]
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

    if snippet:
        # st.subheader(f'Run Check {check_name}')
        st.code(snippet, language='python')

    if result_html:
        height_px = 800
        html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
        components.html(html, height=height_px)

    if result_value is not None:
        st.code('print(result.value)', language='python')
        # If the result value is simple type (e.g. int, float, str) it can't be displayed as json
        if isinstance(result_value, (dict, Sequence)):
            result_value = json.dumps(result_value, indent=4, sort_keys=False, cls=AppEncoder)
            st.json(result_value)
        else:
            st.code(str(result_value), language='python')

with col2:
    st.subheader(f'Check {check_class.__name__} docstring')
    st.text(check_class.__doc__)
