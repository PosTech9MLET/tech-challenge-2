# syntax=docker/dockerfile:1
# =============================================================
# Stage 1 — builder
# Instala uv e todas as dependências de produção.
# Não entra na imagem final, só serve para construir o ambiente.
# =============================================================
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Instala o uv
RUN pip install --no-cache-dir uv

# Copia arquivos de dependências primeiro para aproveitar cache
COPY pyproject.toml uv.lock ./

# Instala só as deps de produção (sem pytest, ruff, pre-commit)
RUN uv sync --no-dev --frozen

# =============================================================
# Stage 2 — runtime
# Imagem final enxuta com apenas o necessário para execução.
# =============================================================
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:/usr/local/bin:$PATH"

WORKDIR /app

# Instala o uv globalmente para ficar disponível em subshells do DVC
RUN pip install --no-cache-dir uv

# Copia o venv construído no builder
COPY --from=builder /app/.venv /app/.venv

# Copia código fonte e configurações
COPY src/ ./src/
COPY configs/ ./configs/
COPY params.yaml dvc.yaml .dvc/ ./
COPY scripts/ ./scripts/
COPY main.py pyproject.toml ./

# Torna o entrypoint executável
RUN chmod +x scripts/entrypoint.sh

# Cria diretórios necessários (dados e modelos montados via volume)
RUN mkdir -p data/raw data/processed data/features models mlruns
    
# Usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /mlflow/artifacts && \
    chown -R appuser:appuser /app /mlflow
USER appuser

ENTRYPOINT ["bash", "scripts/entrypoint.sh"]
