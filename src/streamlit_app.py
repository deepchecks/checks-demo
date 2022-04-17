import io
import json
from pathlib import Path
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

# First select a dataset and check
dataset_name = st.sidebar.selectbox('Select a dataset', datasets.keys())
check_name = st.sidebar.selectbox('Select a check', checks.keys())

dataset = datasets[dataset_name]
check = checks[check_name]

# Run the check
with col1:
    with st.spinner('Running check'):
        check_result = check['run'](dataset)
        if isinstance(check_result, str):
            with col1:
                st.error(check_result)
            result_html = None
            result_value = None
        else:
            string_io = io.StringIO()
            check_result.save_as_html(string_io)
            result_html = string_io.getvalue()
            result_value = json.dumps(check_result.value, indent=4, sort_keys=False, cls=AppEncoder)

    if check['snippet']:
        st.subheader('Run')
        st.code(check['snippet'], language='python')

    if result_html:
        height_px = 800
        html = TEMPLATE_WRAPPER.format(body=result_html, height=height_px)
        components.html(html, height=height_px)

    if result_value:
        st.subheader('Result Value')
        st.json(result_value)

with col2:
    if check['docstring']:
        st.subheader('Docstring')
        st.text(check['docstring'])
