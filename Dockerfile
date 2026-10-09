FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir . \
    && groupadd --system app && useradd --system --gid app app

COPY --chown=app:app models/larger.npz ./models/larger.npz
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)"
CMD ["adb", "serve", "--checkpoint", "models/larger.npz", "--host", "0.0.0.0", "--port", "8000"]
