# syntax=docker/dockerfile:1

# ============================================================
# STAGE 1 - BUILDER
# Instala dependencias de producao e prepara a aplicacao.
# ============================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Instala uv somente para preparar o ambiente do projeto.
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /usr/local/bin/

# Copia primeiro os arquivos de dependencias para aproveitar o cache.
COPY pyproject.toml uv.lock ./

# Instala apenas dependencias de producao.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Copia o codigo-fonte.
# O .dockerignore exclui .env, dados, modelos, caches e .venv local.
COPY . .

# Instala o projeto no ambiente virtual de forma nao editavel.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


# ============================================================
# STAGE 2 - RUNTIME
# Imagem final para executar o pipeline DVC.
# ============================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/app/.venv \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_NO_SYNC=1 \
    UV_NO_DEV=1 \
    PATH="/app/.venv/bin:/usr/local/bin:$PATH"

WORKDIR /app

# O dvc.yaml usa "uv run python ...".
# Portanto, o uv precisa existir tambem no runtime.
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /usr/local/bin/

# Cria usuario sem privilegios administrativos.
RUN groupadd --system app \
    && useradd --system --gid app --create-home app

# Copia apenas aplicacao e ambiente virtual prontos do builder.
COPY --from=builder --chown=app:app /app /app

# Cria os diretorios usados pelo pipeline, DVC e MLflow.
RUN mkdir -p /app/data/raw \
             /app/data/processed \
             /app/data/features \
             /app/models \
             /app/mlruns \
             /app/.dvc/cache \
    && chown -R app:app /app

USER app

CMD ["dvc", "repro"]