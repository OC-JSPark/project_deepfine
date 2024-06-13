FROM python:3.8-slim

RUN pip install transformers torch tqdm

COPY train.py /app/train.py
COPY /app /app

ENTRYPOINT ["python", "/app/train.py"]
