"""Use this file to run the application inside an IDE and allow easy debugging."""
from streamlit import bootstrap

real_script = 'src/streamlit_app.py'

bootstrap.run(real_script, f'run.py {real_script}', [], {})
