FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
ENV FASTEMBED_CACHE_PATH=/app/models
# Bake models in so a cold start doesn't download them.
RUN .venv/bin/python -c "from copilot.embed import _model, _reranker; _model(); _reranker()"
RUN chmod -R a+rX /app/models
USER nobody
CMD ["/app/.venv/bin/uvicorn", "copilot.app:app", "--host", "0.0.0.0", "--port", "8000"]
