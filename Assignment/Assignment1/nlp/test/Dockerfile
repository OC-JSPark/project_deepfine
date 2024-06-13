FROM python:3.8-slim

RUN pip install transformers torch scikit-learn

COPY test.py /app/test.py
COPY /app /app

ENTRYPOINT ["python", "/app/test.py"]
