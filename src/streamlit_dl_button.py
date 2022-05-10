"""
Taken from https://gitlab.com/-/snippets/2106399
The streamlit built-in download button isn't working well https://github.com/streamlit/streamlit/issues/4347
"""
import base64
import json
import pickle
import uuid
import re

import streamlit as st
import pandas as pd


def download_button(object_to_download, download_filename, button_text, pickle_it=False,
                    mimetype='file/csv'):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    prim_color = st.config.get_option('theme.primaryColor') or '#F43365'
    bg_color = st.config.get_option('theme.backgroundColor') or '#FFFFFF'
    sbg_color = st.config.get_option('theme.secondaryBackgroundColor') or '#000000'
    txt_color = st.config.get_option('theme.textColor') or '#8a0294'
    font = st.config.get_option('theme.font') or 'sans serif'

    custom_css = f"""
        <style>
            #{button_id} {{
                background-color: {bg_color};
                color: {txt_color};
                padding: 0.25rem 0.75rem;
                position: relative;
                line-height: 1.6;
                border-radius: 0.25rem;
                border-width: 1px;
                border-style: solid;
                border-color: {sbg_color};
                border-image: initial;
                filter: brightness(105%);
                justify-content: center;
                margin: 0px;
                width: auto;
                appearance: button;
                display: inline-flex;
                family-font: {font};
                font-weight: 400;
                letter-spacing: normal;
                word-spacing: normal;
                text-align: center;
                text-rendering: auto;
                text-transform: none;
                text-indent: 0px;
                text-shadow: none;
                text-decoration: none;
                margin: 0px 5px 10px;
            }}
            #{button_id}:hover {{

                border-color: {prim_color};
                color: {prim_color};
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: {prim_color};
                color: {sbg_color};
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" class= "" id="{button_id}" ' \
                           f'href="data:{mimetype};base64,{b64}">{button_text}</a>'

    return dl_link
