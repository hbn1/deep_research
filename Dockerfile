# ---- Build stage ----
FROM python:3.11-slim AS builder
WORKDIR /app

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PIP_EXTRA_INDEX_URL=
ARG PIP_DEFAULT_TIMEOUT=120
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt pyproject.toml ./
RUN python -m pip install --target /deps -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/app
ENV APP_ENV=production
ENV RAG_UPLOAD_DIR=/data/rag_uploads
ENV EVAL_DATASETS_DIR=/app/eval_datasets
ENV EVAL_RESULTS_DIR=/data/eval_results

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /data/rag_uploads /data/eval_results \
    && chown -R appuser:appuser /data /app

COPY --from=builder /deps /usr/local/lib/python3.11/site-packages/
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser eval_datasets/ ./eval_datasets/
COPY --chown=appuser:appuser main.py pyproject.toml requirements.txt config.json ./

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["python", "-m", "uvicorn", "app_main:app", "--host", "0.0.0.0", "--port", "8000"]
