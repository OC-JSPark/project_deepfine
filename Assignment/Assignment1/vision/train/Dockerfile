FROM python:3.8-slim

RUN pip install torch torchvision transformers datasets

COPY train.py /app/train.py
COPY preprocess.py /app/preprocess.py

ENTRYPOINT ["python", "/app/train.py"]
