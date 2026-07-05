# syntax=docker/dockerfile:1

# ============================================================
# STAGE 1 — BUILDER
# Instala dependências de produção e prepara a aplicação.
# ============================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Copia o uv para dentro do estágio de build.
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/

# Copia apenas os arquivos de dependências primeiro.
# Assim o Docker reutiliza o cache quando apenas o código mudar.
COPY pyproject.toml uv.lock ./

# Instala dependências travadas no uv.lock, sem ferramentas de desenvolvimento.
# O projeto ainda não é instalado nesta camada.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Copia o restante do projeto.
# O .dockerignore evita copiar .env, dados, .venv, modelos e caches.
COPY . .

# Instala o projeto no ambiente virtual de forma não editável.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


# ============================================================
# STAGE 2 — RUNTIME
# Recebe somente a aplicação pronta e sua virtualenv.
# Não leva uv, pre-commit, pytest ou Ruff.
# ============================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Usuário sem privilégios administrativos.
RUN groupadd --system app \
    && useradd --system --gid app --create-home app

# Copia somente o resultado do estágio builder.
COPY --from=builder --chown=app:app /app /app

# Cria diretórios que serão usados por DVC, pipeline e MLflow.
RUN mkdir -p /app/data/raw \
             /app/data/processed \
             /app/models \
             /app/mlruns \
             /app/.dvc/cache \
    && chown -R app:app /app

USER app

# O docker-compose poderá substituir este comando quando necessário.
CMD ["dvc", "repro"]