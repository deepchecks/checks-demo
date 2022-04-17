FROM python:3.9.10-alpine3.14

EXPOSE 8501

WORKDIR /opt/app
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY /resources /resources
COPY /src /src

CMD ["streamlit", "run", "src/streamlit_app.py"]
