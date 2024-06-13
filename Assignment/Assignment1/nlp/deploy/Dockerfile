FROM python:3.8-slim

RUN pip install transformers torch

COPY deploy.py /app/deploy.py
COPY /app /app

ENTRYPOINT ["python", "/app/deploy.py"]
