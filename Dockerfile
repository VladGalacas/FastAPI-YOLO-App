FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY best.onnx .
COPY templates/index.html ./templates/
COPY static/style.css ./static/

ENV DEVICE="cuda" \
    APP_HOST="0.0.0.0" \
    APP_PORT=8000

ENTRYPOINT ["sh", "-c", "uvicorn main:app --host=${APP_HOST:-0.0.0.0} --port=${APP_PORT:-8000}"]