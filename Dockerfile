FROM python:3.9.10-slim

ENV PORT 8000

WORKDIR /opt/app

RUN apt update && apt install git -y

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["sh", "run.sh"]
