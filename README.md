# 🛒 Tech Challenge — Fase 2
### Sistema de Recomendação de Produtos | POSTECH ML Engineering

---

## Visão Geral

Sistema de recomendação de produtos baseado no comportamento de compra de usuários do **Instacart**, utilizando redes neurais (MLP/Embedding-based) treinadas com PyTorch. O pipeline completo é containerizado com Docker, dados versionados com DVC no Azure Blob Storage, experimentos rastreados com MLflow e o código segue padrões profissionais de Clean Code.

> **Dataset:** [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)  
> **Grupo:** PosTech9MLET — FIAP POSTECH ML Engineering

---

## Estrutura do Projeto

```
tech-challenge-2/
├── configs/                    # Configurações da aplicação
│   ├── __init__.py
│   └── settings.py             # Pydantic Settings + Azure Key Vault
├── data/
│   ├── raw/                    # CSVs originais (gerenciados pelo DVC)
│   ├── processed/              # Dados processados
│   └── features/               # Features engenheiradas
├── models/                     # Artefatos de modelos treinados
├── notebooks/
│   └── eda_instacart.ipynb     # Análise Exploratória de Dados
├── scripts/
│   └── validate_env.py         # Validação do ambiente
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocess.py       # Funções de carregamento e limpeza
│   ├── models/                 # Modelos ML (baseline + MLP)
│   ├── pipelines/              # Pipelines DVC
│   ├── training/               # Lógica de treino
│   └── utils/                  # Utilitários compartilhados
├── tests/                      # Testes automatizados
├── .dvc/                       # Configuração DVC
├── .env.example                # Template de variáveis de ambiente
├── .gitignore
├── .pre-commit-config.yaml     # Hooks de qualidade de código
├── main.py
├── pyproject.toml              # Dependências e configuração do projeto
└── uv.lock                     # Lock file para reprodutibilidade
```

---

## Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — gerenciador de dependências
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) — autenticação com Azure
- Acesso ao Azure Blob Storage e Key Vault (credenciais via `.env`)

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/PosTech9MLET/tech-challenge-2.git
cd tech-challenge-2
```

### 2. Instale as dependências

```bash
uv sync
```

### 3. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com suas credenciais:

```dotenv
# MODELO
SEED=42
DATA_INPUT_PATH=""
DATA_OUTPUT_PATH=""
EARLY_STOPPING_PATIENCE=5

# DVC / AZURE
AZURE_STORAGE_ACCOUNT=stgtechchallenge
AZURE_STORAGE_KEY=             # deixe vazio para buscar do Key Vault
AZURE_CONTAINER_NAME=tech-challenge-f2

# MLFLOW
MLFLOW_TRACKING_URI=""
MLFLOW_ARTIFACT_LOCATION=""
```

### 4. Autentique no Azure

```bash
az login
```

### 5. Baixe os dados via DVC

```bash
uv run dvc pull
```

### 6. Valide o ambiente

```bash
uv run python scripts/validate_env.py
```

---

## Dataset

O dataset do Instacart contém o histórico de compras de mais de 200.000 usuários. São 6 arquivos relacionais:

| Arquivo | Descrição | Tamanho |
|---|---|---|
| `orders.csv` | Pedidos de cada usuário | ~3.4M linhas |
| `order_products__prior.csv` | Produtos dos pedidos históricos | ~32M linhas |
| `order_products__train.csv` | Produtos dos pedidos de treino | ~1.4M linhas |
| `products.csv` | Catálogo de produtos | 49.688 produtos |
| `aisles.csv` | Corredores do supermercado | 134 corredores |
| `departments.csv` | Departamentos | 21 departamentos |

> Os dados são gerenciados pelo **DVC** e armazenados no **Azure Blob Storage** (`stgtechchallenge/tech-challenge-f2`). Nunca são commitados diretamente no Git.

---

## Configuração de Ambiente

### Azure Key Vault

O projeto utiliza o **Azure Key Vault** (`techchallengevaults`) para gerenciar secrets em produção. Localmente, a autenticação é feita via Azure CLI.

Se `AZURE_STORAGE_KEY` estiver vazio no `.env`, o sistema busca automaticamente do Key Vault:

```python
settings.get_azure_storage_key()  # busca do Key Vault se necessário
```

### DVC Remote

O remote DVC está configurado para o Azure Blob Storage:

```
azure://tech-challenge-f2/dvc
```

Para configurar localmente:

```bash
uv run dvc remote modify --local azure_remote account_key SUA_KEY
```

---

## Qualidade de Código

O projeto utiliza **ruff** para linting e formatação, com **pre-commit** hooks que rodam automaticamente em todo commit.

```bash
# Instalar hooks
uv run pre-commit install

# Rodar manualmente
uv run pre-commit run --all-files

# Verificar linting
uv run ruff check .
```

Regras ativas: `E` (estilo), `F` (lógico), `I` (imports), `N` (naming), `UP` (modernização), `D` (docstrings Google style).

---

## Dependências Principais

| Biblioteca | Versão | Uso |
|---|---|---|
| `torch` | ≥2.12.0 | Rede neural (CPU-only) |
| `scikit-learn` | ≥1.8.0 | Modelos baseline e pré-processamento |
| `mlflow` | ≥3.12.0 | Tracking de experimentos |
| `dvc[azure]` | ≥3.67.1 | Versionamento de dados |
| `pandas` | ≥2.3.3 | Manipulação de dados |
| `pydantic-settings` | ≥2.14.1 | Configuração via .env |
| `azure-keyvault-secrets` | — | Gerenciamento de secrets |

---

## Status do Projeto

| Etapa | Descrição | Status |
|---|---|---|
| EDA | Análise Exploratória de Dados | ✅ Concluído |
| Etapa 1 | Clean Code e Estrutura | ✅ Concluído |
| Etapa 2 | Ambiente e Dependências | ✅ Concluído |
| Etapa 3 | Feature Engineering + DVC Pipeline | 🔄 Em progresso |
| Etapa 3 | Docker (multi-stage) | ⏳ Pendente |
| Etapa 4 | Modelagem (Baseline + MLP PyTorch) | ⏳ Pendente |
| Etapa 4 | MLflow Model Registry | ⏳ Pendente |
| Entrega | README + Vídeo STAR | ⏳ Pendente |

---

## Integrantes

| Nome | RM |
|---|---|
| Gabriel Freitas | RM370409 |
| Diego | — |
| Deyvid | — |
| Lucas Molitor | — |

---

## Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.