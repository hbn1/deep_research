# ---- Build stage ----
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --target /deps langgraph langchain langchain-community langchain-core \
    fastapi uvicorn dashscope python-dotenv pydantic-settings

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime
WORKDIR /app

COPY --from=builder /deps /usr/local/lib/python3.11/site-packages/
COPY app/ ./app/
COPY main.py pyproject.toml ./

ENV PYTHONPATH=/app/app
ENV DASHSCOPE_API_KEY=""
ENV BOCHA_API_KEY=""

EXPOSE 8000
CMD ["uvicorn", "app_main:app", "--host", "0.0.0.0", "--port", "8000"]
