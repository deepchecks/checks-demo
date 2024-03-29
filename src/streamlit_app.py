from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from analytics import inject_hotjar, inject_meta_tags, inject_gtm
from checks import show_checks_page

from constants import NO_CHECK_SELECTED, CHECK_STATE_ID, CHECK_QUERY_PARAM, SUITE_QUERY_PARAM, NO_SUITE_SELECTED, \
    SUITE_STATE_ID
from streamlit_persist import load_widget_state
# from suites import show_suites_page
from utils import get_query_param, set_query_param

# Inject to streamlit index page analytics code and meta tags
load_dotenv()
inject_hotjar()
inject_meta_tags()
inject_gtm()

icon = Image.open(Path(__file__).parent.parent / 'resources' / 'favicon.ico')
logo = open(Path(__file__).parent.parent / 'resources' / 'deepchecks_logo.svg').read()
logo_with_link = f'<a href="https://deepchecks.com" target="_blank">{logo}</a>'

st.set_page_config(page_title='Deepchecks Checks Demo', page_icon=icon, layout='wide')
st.sidebar.markdown(logo_with_link, unsafe_allow_html=True)

# Hack to allow widgets state to be saved when widget is removed from the page
load_widget_state()
# Set default state or load state from URL query params
if SUITE_STATE_ID not in st.session_state:
    st.session_state[SUITE_STATE_ID] = get_query_param(SUITE_QUERY_PARAM) or NO_SUITE_SELECTED
if CHECK_STATE_ID not in st.session_state:
    st.session_state[CHECK_STATE_ID] = get_query_param(CHECK_QUERY_PARAM) or NO_CHECK_SELECTED


def mode_change():
    new_mode = st.session_state['mode-radio']
    if new_mode == 'Checks':
        set_query_param(CHECK_QUERY_PARAM, CHECK_STATE_ID)
    else:
        set_query_param(SUITE_QUERY_PARAM, SUITE_STATE_ID)


# Select mode, using URL query params if exists
# suite_query_value = get_query_param(SUITE_QUERY_PARAM)
# st.session_state['mode-radio'] = 'Suite' if suite_query_value is not None else 'Checks'
# mode = st.sidebar.radio('Mode', ['Checks', 'Suite'], key='mode-radio', on_change=mode_change)
mode = 'Checks'

if mode == 'Checks':
    set_query_param(CHECK_QUERY_PARAM, CHECK_STATE_ID)
    show_checks_page()
else:
    set_query_param(SUITE_QUERY_PARAM, SUITE_STATE_ID)
    # show_suites_page()

# If nothing is chosen show the open page
if (mode == 'Suite' and st.session_state.get(SUITE_STATE_ID) == NO_SUITE_SELECTED) or \
   (mode == 'Checks' and st.session_state.get(CHECK_STATE_ID) == NO_CHECK_SELECTED):
    st.markdown(
    """
    # Welcome to Deepchecks' Interactive Checks Demo 🚀
    
    In this demo you can play with some of the existing checks and see how they work on various datasets. <br/>
    The demo enables custom corruptions to the datasets to showcase the checks' value. 
    
    If you like what we're doing at Deepchecks, please ⭐&nbsp;us on [GitHub](https://github.com/deepchecks/deepchecks).<br/>
    And if you'd like to dive in a bit more, check out our [documentation](https://docs.deepchecks.com/stable?utm_campaign=docs_button&utm_medium=referral&utm_source=checks-demo.deepchecks.com).
    
    ### ⬅️ Start by selecting a check in the menu
    
    ![](https://docs.deepchecks.com/stable/_images/general/checks-and-conditions.png)
    """, unsafe_allow_html=True)
