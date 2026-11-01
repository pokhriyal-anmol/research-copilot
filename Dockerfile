FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
USER nobody
ENV HF_HOME=/tmp/hf FASTEMBED_CACHE_PATH=/tmp/fastembed
CMD ["/app/.venv/bin/uvicorn", "copilot.app:app", "--host", "0.0.0.0", "--port", "8000"]
