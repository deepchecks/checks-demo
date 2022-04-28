import os

import streamlit as st
import shutil
from pathlib import Path
from bs4 import BeautifulSoup


HOTJAR_TRACK_CODE = """
<!-- Hotjar Tracking Code for https://checks-demo.deepchecks.com -->
<script id="element_id">
    (function(h,o,t,j,a,r){
        h.hj=h.hj||function(){(h.hj.q=h.hj.q||[]).push(arguments)};
        h._hjSettings={hjid:hotjar_id,hjsv:6};
        a=o.getElementsByTagName('head')[0];
        r=o.createElement('script');r.async=1;
        r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;
        a.appendChild(r);
    })(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
</script>
"""

GA_TRACK_CODE = """
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-ga_id"></script>
<script id="element_id">
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-ga_id');
</script>
"""


def inject_ga():
    ga_id = os.environ.get('GA_ID')
    if ga_id is None:
        print('No GA_ID found in environment variables')
        return

    GA_ELEMENT = "google_analytics"
    script = GA_TRACK_CODE.replace('ga_id', ga_id).replace('element_id', GA_ELEMENT)
    inject_script_to_streamlit(script, GA_ELEMENT)


def inject_hotjar():
    hotjar_id = os.environ.get('HOTJAR_ID')
    if hotjar_id is None:
        print('No HOTJAR_ID found in environment variables')
        return

    HOTJAR_ELEMENT = "hotjar"
    script = HOTJAR_TRACK_CODE.replace('hotjar_id', hotjar_id).replace('element_id', HOTJAR_ELEMENT)
    inject_script_to_streamlit(script, HOTJAR_ELEMENT)


def inject_script_to_streamlit(script, element_id):
    # Insert the script in the head tag of the static template inside your virtual
    index_path = Path(st.__file__).parent / "static" / "index.html"
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")
    if not soup.find(id=element_id):  # if cannot find tag
        bck_index = index_path.with_suffix('.bck')
        if bck_index.exists():
            shutil.copy(bck_index, index_path)  # recover from backup
        else:
            shutil.copy(index_path, bck_index)  # keep a backup
        html = str(soup)
        new_html = html.replace('<head>', '<head>\n' + script)
        index_path.write_text(new_html)
        print(f'Injected {element_id}')
