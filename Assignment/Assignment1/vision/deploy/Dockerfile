FROM python:3.8-slim

RUN pip install torch transformers

COPY deploy.py /app/deploy.py
COPY train.py /app/train.py

ENTRYPOINT ["python", "/app/deploy.py"]
