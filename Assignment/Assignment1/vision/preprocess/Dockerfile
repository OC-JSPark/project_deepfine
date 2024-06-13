FROM python:3.8-slim

RUN pip install torch torchvision

COPY preprocess.py /app/preprocess.py

ENTRYPOINT ["python", "/app/preprocess.py"]
