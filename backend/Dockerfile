FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential libgomp1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-lock.txt ./
RUN sed '/^xgboost==/d' requirements-lock.txt > /tmp/requirements-docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && pip install --no-cache-dir --no-deps xgboost==3.2.0 \
    && rm -f /tmp/requirements-docker.txt
COPY . .

RUN useradd --create-home --uid 10001 qsarify \
    && mkdir -p /var/lib/qsarify \
    && chown -R qsarify:qsarify /app /var/lib/qsarify

ENV FLASK_ENV=production \
    QSARIFY_BACKEND_HOST=0.0.0.0 \
    QSARIFY_BACKEND_PORT=5051 \
    QSARIFY_RUNTIME_DIR=/var/lib/qsarify \
    QSARIFY_SESSION_DIR=/var/lib/qsarify/flask_session \
    QSARIFY_UPLOAD_FOLDER=/var/lib/qsarify/uploads \
    QSARIFY_MODEL_ARTIFACT_ROOT=/var/lib/qsarify/models

USER qsarify
EXPOSE 5051
CMD ["python", "run.py"]
