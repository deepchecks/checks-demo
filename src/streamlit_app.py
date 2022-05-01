from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from analytics import inject_ga, inject_hotjar, inject_meta_tags
from checks import show_checks_page

# Add google analytics
load_dotenv()
inject_ga()
inject_hotjar()
inject_meta_tags()

icon = Image.open(Path(__file__).parent.parent / 'resources' / 'favicon.ico')
logo = open(Path(__file__).parent.parent / 'resources' / 'deepchecks_logo.svg').read()
logo_with_link = f'<a href="https://deepchecks.com" target="_blank">{logo}</a>'

st.set_page_config(page_title='Deepchecks Checks Demo', page_icon=icon, layout='wide')
st.sidebar.markdown(logo_with_link, unsafe_allow_html=True)


show_checks_page()
