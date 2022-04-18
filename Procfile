// This file defines the run command for heroku deployment.
// Heroku provides for us the port to run in env var, therefore passing the port to streamlit
// server.headless = true -> Will prevent streamlit from trying to open a browser window.
web: streamlit run src/streamlit_app.py --server.port $PORT --server.headless true
