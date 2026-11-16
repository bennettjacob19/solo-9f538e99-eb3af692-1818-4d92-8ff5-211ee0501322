FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    DEFAULT_INFER_TIMEOUT_MS=5000 \
    MAX_PENDING_REQUESTS=32

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY examples ./examples

# The API must run as a single process: it is the sole owner of the two-slot
# budget.  A second instance against the same data directory is rejected by a
# startup file lock, so do not add uvicorn/gunicorn workers.
RUN useradd --create-home --uid 10001 qc \
    && mkdir -p /data/models /data/staging /data/workers \
    && chown -R qc:qc /data /app
USER qc

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
