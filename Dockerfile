# Triển khai trên máy chủ Linux:
#   docker build -t facereg .
#   docker run -d -p 8000:8000 --env-file .env -v facereg_data:/app/data -v facereg_models:/app/models facereg
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir cython && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY scripts ./scripts

ENV DATA_DIR=/app/data MODEL_ROOT=/app/models
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
