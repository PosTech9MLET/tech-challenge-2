# 🛒 Tech Challenge — Fase 2
### Sistema de Recomendação de Produtos | POSTECH ML Engineering

---

## Visão Geral

Sistema de recomendação de produtos baseado no comportamento de compra de usuários do **Instacart**, utilizando redes neurais (MLP/Embedding-based) treinadas com PyTorch. O pipeline completo é containerizado com Docker, dados versionados com DVC, experimentos rastreados com MLflow e o código segue padrões profissionais de Clean Code.

> **Dataset:** [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)  
> **Grupo:** PosTech9MLET — FIAP POSTECH ML Engineering

---

## Integrantes

| Nome | RM |
|---|---|
| Gabriel Freitas | RM370409 |
| Diego | RM372988 |
| Deyvid Manhães | RM371074 |

---

## Estrutura do Projeto

```
tech-challenge-2/
├── configs/                    # Configurações da aplicação
│   ├── __init__.py
│   └── settings.py             # Pydantic Settings
├── data/
│   ├── raw/                    # CSVs originais (gerenciados pelo DVC)
│   ├── processed/              # Dados processados
│   └── features/               # Features engenheiradas
├── models/                     # Artefatos de modelos treinados
├── notebooks/
│   └── eda_instacart.ipynb     # Análise Exploratória de Dados
├── scripts/
│   ├── setup_dvc.py            # Configura DVC remote
│   └── validate_env.py         # Validação do ambiente
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_builder.py  # Feature engineering
│   │   └── preprocess.py       # Funções de carregamento e limpeza
│   ├── models/
│   │   ├── baseline.py         # PopularityBaseline + métricas
│   │   ├── factory.py          # ModelFactory (design pattern)
│   │   └── mlp.py              # RecommenderMLP (PyTorch)
│   ├── pipelines/              # Stages do pipeline DVC
│   ├── training/               # Lógica de treino com early stopping
│   └── utils/                  # Utilitários compartilhados
├── tests/                      # Testes automatizados (pytest)
├── .dvc/                       # Configuração DVC
├── .env.example                # Template de variáveis de ambiente
├── .env.docker.example         # Template para execução com Docker
├── .gitignore
├── .pre-commit-config.yaml     # Hooks de qualidade de código
├── Dockerfile                  # Multi-stage: builder + runtime
├── MODEL_CARD.md               # Documentação do modelo
├── docker-compose.yml          # Serviços mlflow + train
├── dvc.yaml                    # Pipeline DVC (4 stages)
├── params.yaml                 # Hiperparâmetros externalizados
├── pyproject.toml              # Dependências e configuração do projeto
└── uv.lock                     # Lock file para reprodutibilidade
```

---

## Guia de Início Rápido

### Passo 1 — Instale o `uv`

O projeto usa `uv` como gerenciador de dependências. Instale uma única vez:

```bash
pip install uv
```

> Documentação completa: https://docs.astral.sh/uv/

### Passo 2 — Clone o repositório

```bash
git clone https://github.com/PosTech9MLET/tech-challenge-2.git
cd tech-challenge-2
```

### Passo 3 — Instale as dependências

```bash
uv sync
```

Esse comando lê o `uv.lock` e instala exatamente as mesmas versões usadas pelo grupo, garantindo reprodutibilidade.

### Passo 4 — Configure o ambiente

```bash
cp .env.example .env
```

Para execução **local** (sem nuvem), o `.env` mínimo é:

```dotenv
# MODELO
SEED=42
DATA_INPUT_PATH=""
DATA_OUTPUT_PATH=""
EARLY_STOPPING_PATIENCE=5

# MLFLOW
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
MLFLOW_ARTIFACT_LOCATION=
```

### Passo 5 — Instale os hooks de qualidade de código

```bash
uv run pre-commit install
```

Este comando instala os hooks do `ruff` no Git local. A partir daí, o linting roda automaticamente em todo `git commit`.

### Passo 6 — Baixe os dados do Instacart

O projeto usa DVC para versionar os dados. Rode o script de configuração:

```bash
uv run python scripts/setup_dvc.py
```

> Se as credenciais Azure não estiverem configuradas no `.env`, o script detecta automaticamente e usa o remote local.

Faça o download dos 6 CSVs do Kaggle manualmente e coloque em `data/raw/`:

```
data/raw/
├── aisles.csv
├── departments.csv
├── orders.csv
├── order_products__prior.csv
├── order_products__train.csv
└── products.csv
```

> **Download:** [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)

### Passo 7 — Valide o ambiente

```bash
uv run python scripts/validate_env.py
```

### Passo 8 — Execute o pipeline completo

```bash
uv run dvc repro
```

Isso executa os 4 stages em ordem: `preprocess → feature_eng → train → evaluate`.

> ⚠️ O pipeline processa ~32M de interações — o tempo de execução varia entre 2 e 6 horas dependendo da máquina.

### Passo 9 — Visualize os experimentos no MLflow

```bash
uv run mlflow ui --backend-store-uri mlruns/
```

Acesse `http://127.0.0.1:5000` no navegador.

---

## Testes

O projeto usa `pytest` para testes unitários. Para rodar todos os testes:

```bash
uv run pytest tests/ -v
```

Para rodar com relatório de cobertura:

```bash
uv run pytest tests/ -v --cov=src --cov=configs --cov-report=term-missing
```

Os testes cobrem os seguintes módulos:

| Arquivo | Módulo testado |
|---|---|
| `tests/test_preprocess.py` | `src/features/preprocess.py` |
| `tests/test_baseline.py` | `src/models/baseline.py` |
| `tests/test_factory.py` | `src/models/factory.py` |
| `tests/test_mlp.py` | `src/models/mlp.py` |

---

## Execução com Docker

O projeto disponibiliza uma imagem Docker **multi-stage** e serviços Docker Compose para executar o pipeline DVC em ambiente isolado e acompanhar experimentos no MLflow.

### Pré-requisitos

- Docker Desktop em execução
- Docker Compose v2

```bash
docker version
docker compose version
```

### Estrutura Docker

| Arquivo | Responsabilidade |
|---|---|
| `Dockerfile` | Imagem multi-stage com Python, `uv`, DVC, MLflow e PyTorch |
| `.dockerignore` | Impede que secrets, dados e caches entrem na imagem |
| `docker-compose.yml` | Orquestra os serviços `mlflow` e `train` |
| `.env.docker.example` | Template de variáveis para o container de treino |

### Configuração inicial

```bash
cp .env.docker.example .env.docker
```

### Construir a imagem

```bash
docker compose build
```

### Subir o MLflow

```bash
docker compose up -d mlflow
```

Acesse a interface em `http://localhost:5000`.

### Executar o pipeline no container

Com os dados disponíveis em `data/raw/`:

```bash
docker compose run --rm --no-deps train dvc repro
```

### Encerrar os serviços

```bash
docker compose down
```

---

## Fluxo de Desenvolvimento

Para contribuir com o projeto, siga este fluxo:

```bash
# 1. Crie uma branch para sua feature a partir da dev
git checkout dev
git checkout -b feat/nome-da-feature

# 2. Desenvolva e teste localmente
uv run pytest tests/ -v
uv run python scripts/validate_env.py

# 3. O pre-commit roda automaticamente ao commitar
git add .
git commit -m "feat: descrição da mudança"

# 4. Abra um Pull Request para dev
git push origin feat/nome-da-feature
```

### Convenção de commits

| Prefixo | Uso |
|---|---|
| `feat:` | Nova funcionalidade |
| `fix:` | Correção de bug |
| `docs:` | Documentação |
| `refactor:` | Refatoração sem mudança de comportamento |
| `test:` | Adição ou correção de testes |
| `chore:` | Tarefas de manutenção (deps, config) |

---

## Qualidade de Código

```bash
# Verificar erros
uv run ruff check .

# Corrigir automaticamente
uv run ruff check . --fix

# Rodar pre-commit manualmente
uv run pre-commit run --all-files
```

Regras ativas: `E` (estilo), `F` (lógico), `I` (imports), `N` (naming), `UP` (modernização), `D` (docstrings Google style).

---

## Pipeline DVC

O pipeline é composto por 4 stages executados em sequência via `dvc repro`:

| Stage | Comando | Saída |
|---|---|---|
| `preprocess` | `preprocess_stage.py` | Parquets limpos em `data/processed/` |
| `feature_eng` | `feature_eng_stage.py` | Features em `data/features/` + encoders |
| `train` | `train_stage.py` | Modelos em `models/` + runs no MLflow |
| `evaluate` | `evaluate_stage.py` | Métricas em `models/eval_metrics.json` |

Os hiperparâmetros são externalizados em `params.yaml` e versionados pelo DVC.

---

## Dataset

| Arquivo | Descrição | Tamanho |
|---|---|---|
| `orders.csv` | Pedidos de cada usuário | ~3.4M linhas |
| `order_products__prior.csv` | Produtos dos pedidos históricos | ~32M linhas |
| `order_products__train.csv` | Produtos dos pedidos de treino | ~1.4M linhas |
| `products.csv` | Catálogo de produtos | 49.688 produtos |
| `aisles.csv` | Corredores do supermercado | 134 corredores |
| `departments.csv` | Departamentos | 21 departamentos |

---

## Dependências Principais

| Biblioteca | Versão | Uso |
|---|---|---|
| `torch` | ≥2.12.0 | Rede neural (CPU-only) |
| `scikit-learn` | ≥1.8.0 | Modelos baseline e pré-processamento |
| `mlflow` | ≥3.12.0 | Tracking de experimentos |
| `dvc` | ≥3.67.1 | Versionamento de dados e pipeline |
| `pandas` | ≥2.3.3 | Manipulação de dados |
| `pydantic-settings` | ≥2.14.1 | Configuração via .env |

---

## Status do Projeto

| Etapa | Descrição | Status |
|---|---|---|
| EDA | Análise Exploratória de Dados | ✅ Concluído |
| Etapa 1 | Clean Code e Estrutura | ✅ Concluído |
| Etapa 2 | Ambiente e Dependências | ✅ Concluído |
| Etapa 3 | Feature Engineering + DVC Pipeline | ✅ Concluído |
| Etapa 3 | Docker (multi-stage) | ✅ Concluído |
| Etapa 4 | Modelagem (Baseline + MLP PyTorch) | ✅ Concluído |
| Etapa 4 | MLflow Model Registry | ✅ Concluído |
| Testes | pytest (preprocess, baseline, factory, MLP) | ✅ Concluído |
| Entrega | README + Vídeo STAR | ⏳ Pendente |

---

## Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.