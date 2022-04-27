from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from analytics import inject_ga, inject_hotjar
from checks import show_checks_page

# Add google analytics
load_dotenv()
inject_ga()
inject_hotjar()

icon = Image.open(Path(__file__).parent.parent / 'resources' / 'favicon.ico')
logo = open(Path(__file__).parent.parent / 'resources' / 'deepchecks_logo.svg').read()

st.set_page_config(page_title='Deepchecks Demo', page_icon=icon, layout='wide')
st.sidebar.image(logo, use_column_width=True)

show_checks_page()
