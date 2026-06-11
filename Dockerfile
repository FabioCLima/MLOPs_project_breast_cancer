# --- Stage 1: builder — resolve e instala dependências com uv ---------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependências primeiro: esta camada só invalida se o lockfile mudar
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Código por último (muda com frequência, não invalida a camada de deps)
COPY src/ src/
COPY app/ app/
COPY params.yaml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# --- Stage 2: runtime — imagem enxuta, non-root ------------------------------
FROM python:3.12-slim

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY src/ src/
COPY app/ app/
COPY params.yaml ./

# Artefatos do modelo: a imagem falha no startup com erro claro se ausentes
COPY artifacts/features_mean_imputer.joblib artifacts/
COPY artifacts/features_scaler.joblib artifacts/
COPY models/model.keras models/
COPY metrics/evaluation.json metrics/

# Logs só em stdout (12-factor): o filesystem do container é do root,
# e a plataforma (docker/k8s) é quem coleta logs
ENV PATH="/app/.venv/bin:$PATH" LOG_TO_FILE=false
USER appuser
EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health', timeout=3)"

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind=0.0.0.0:5001", "--workers=2", "app.api:create_app()"]
