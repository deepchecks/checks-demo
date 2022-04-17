import os

import streamlit as st
import shutil
from pathlib import Path
from bs4 import BeautifulSoup


def inject_ga():
    ga_id = os.environ.get('GA_ID')
    if ga_id is None:
        print('No GA_ID found in environment variables')
        return

    GA_ELEMENT = "google_analytics"

    GA_JS = f"""
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-{ga_id}"></script>
        <script id={GA_ELEMENT}>
          window.dataLayer = window.dataLayer || [];
          function gtag(){{dataLayer.push(arguments);}}
          gtag('js', new Date());
        
          gtag('config', 'G-WQ5PEJPN1Q');
        </script>
    """

    # Insert the script in the head tag of the static template inside your virtual
    index_path = Path(st.__file__).parent / "static" / "index.html"
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")
    if not soup.find(id=GA_ELEMENT):  # if cannot find tag
        bck_index = index_path.with_suffix('.bck')
        if bck_index.exists():
            shutil.copy(bck_index, index_path)  # recover from backup
        else:
            shutil.copy(index_path, bck_index)  # keep a backup
        html = str(soup)
        new_html = html.replace('<head>', '<head>\n' + GA_JS)
        index_path.write_text(new_html)
