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

GTM_HEAD_CODE = """
<!-- Google Tag Manager -->
<script id="element_id">(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-gtm_id');</script>
<!-- End Google Tag Manager -->
"""

GTM_BODY_CODE = """
<!-- Google Tag Manager (noscript) -->
<noscript id="element_id"><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-gtm_id"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->
"""

META_TAGS = """
<meta id="meta-title" name="title" content="Deepchecks Checks Demo">
<meta name="description" content="Experiment with Deepchecks library online">

<!-- Open Graph / Facebook -->
<meta property="og:type" content="website">
<meta property="og:url" content="https://checks-demo.deepchecks.com/">
<meta property="og:title" content="Deepchecks Checks Demo">
<meta property="og:description" content="Experiment with Deepchecks library online">
<meta property="og:image" content="https://docs.deepchecks.com/stable/_images/checks_and_conditions.png">

<!-- Twitter -->
<meta property="twitter:card" content="summary_large_image">
<meta property="twitter:url" content="https://checks-demo.deepchecks.com/">
<meta property="twitter:title" content="Deepchecks Checks Demo">
<meta property="twitter:description" content="Experiment with Deepchecks library online">
<meta property="twitter:image" content="https://docs.deepchecks.com/stable/_images/checks_and_conditions.png">
"""


def inject_hotjar():
    hotjar_id = os.environ.get('HOTJAR_ID')
    if hotjar_id is None:
        print('No HOTJAR_ID found in environment variables')
        return

    HOTJAR_ELEMENT = "hotjar"
    script = HOTJAR_TRACK_CODE.replace('hotjar_id', hotjar_id).replace('element_id', HOTJAR_ELEMENT)
    inject_script_to_streamlit(script, HOTJAR_ELEMENT)


def inject_meta_tags():
    TAGS_ID = "meta-title"
    inject_script_to_streamlit(META_TAGS, TAGS_ID)


def inject_gtm():
    gtm_id = os.environ.get('GTM_ID')
    if gtm_id is None:
        print('No GTM_ID found in environment variables')
        return

    GTM_HEAD_ELEMENT = "google_tag_manager_head"
    head_code = GTM_HEAD_CODE.replace('gtm_id', gtm_id).replace('element_id', GTM_HEAD_ELEMENT)
    inject_script_to_streamlit(head_code, GTM_HEAD_ELEMENT)

    GTM_BODY_ELEMENT = "google_tag_manager_body"
    body_code = GTM_BODY_CODE.replace('gtm_id', gtm_id).replace('element_id', GTM_BODY_ELEMENT)
    inject_script_to_streamlit(body_code, GTM_BODY_ELEMENT, inject_head=False)


def inject_script_to_streamlit(script, element_id, inject_head=True):
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
        if inject_head:
            new_html = html.replace('<head>', '<head>\n' + script)
        else:
            new_html = html.replace('<body>', '<body>\n' + script)
        index_path.write_text(new_html)
        print(f'Injected {element_id}')
