FROM python:3.8-slim

RUN pip install torch torchvision transformers evaluate

COPY test.py /app/test.py
COPY train.py /app/train.py

ENTRYPOINT ["python", "/app/test.py"]
