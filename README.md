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
- Credenciais do Service Principal (fornecidas pelo grupo)

> **Não é necessário ter conta Azure ou instalar o Azure CLI.** A autenticação é feita via Service Principal com credenciais compartilhadas.

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

Edite o `.env` com as credenciais fornecidas pelo grupo:

```dotenv
# MODELO
SEED=42
DATA_INPUT_PATH=""
DATA_OUTPUT_PATH=""
EARLY_STOPPING_PATIENCE=5

# DVC / AZURE
AZURE_STORAGE_ACCOUNT=stgtechchallenge
AZURE_STORAGE_KEY=
AZURE_CONTAINER_NAME=tech-challenge-f2

# MLFLOW
MLFLOW_TRACKING_URI=""
MLFLOW_ARTIFACT_LOCATION=""

# SERVICE PRINCIPAL (solicite ao grupo)
AZURE_CLIENT_ID=
AZURE_CLIENT_SECRET=
AZURE_TENANT_ID=
```

> `AZURE_STORAGE_KEY` deve ficar **vazio** — o sistema busca automaticamente do Key Vault usando o Service Principal.

### 4. Baixe os dados via DVC

```bash
uv run dvc pull
```

### 5. Valide o ambiente

```bash
uv run python scripts/validate_env.py
```

---

## Onboarding de Novos Membros

O projeto usa um **Service Principal** do Azure para autenticação — uma identidade de serviço com permissões fixas que permite qualquer membro rodar o projeto sem precisar de conta Azure própria ou configurações individuais de permissão.

### Como funciona

```
.env com credenciais do Service Principal
        ↓
azure-identity autentica automaticamente
        ↓
acessa Key Vault → busca AZURE-STORAGE-KEY
        ↓
DVC pull baixa os dados do Blob Storage
```

### Passos para novos membros

1. **Clonar o repositório** e instalar dependências com `uv sync`
2. **Solicitar ao grupo** as três credenciais do Service Principal:
   - `AZURE_CLIENT_ID`
   - `AZURE_CLIENT_SECRET`
   - `AZURE_TENANT_ID`
3. **Preencher o `.env`** com as credenciais recebidas
4. **Rodar `uv run dvc pull`** para baixar os dados
5. **Validar com `uv run python scripts/validate_env.py`**

> As credenciais do Service Principal **nunca são commitadas** no repositório. Compartilhe apenas por canal seguro (ex.: mensagem direta, gerenciador de senhas).

---

## Infraestrutura Azure

| Recurso | Nome | Finalidade |
|---|---|---|
| Storage Account | `stgtechchallenge` | Armazenamento dos CSVs via DVC |
| Blob Container | `tech-challenge-f2` | Container dos dados e artefatos |
| Key Vault | `techchallengevaults` | Gerenciamento de secrets |
| Service Principal | `tech-challenge-sp` | Autenticação sem conta pessoal |

### Secrets armazenados no Key Vault

| Secret | Descrição |
|---|---|
| `AZURE-STORAGE-KEY` | Access key do Storage Account |
| `AZURE-STORAGE-ACCOUNT` | Nome do Storage Account |
| `AZURE-CONTAINER-NAME` | Nome do container Blob |

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

> Os dados são gerenciados pelo **DVC** e armazenados no **Azure Blob Storage**. Nunca são commitados diretamente no Git.

---

## DVC Remote

O remote DVC está configurado para o Azure Blob Storage:

```
azure://tech-challenge-f2/dvc
```

Comandos úteis:

```bash
# Baixar dados
uv run dvc pull

# Enviar dados atualizados
uv run dvc push

# Verificar status
uv run dvc status
```

---

## Qualidade de Código

O projeto utiliza **ruff** para linting e formatação, com **pre-commit** hooks que rodam automaticamente em todo commit.

```bash
# Instalar hooks (apenas uma vez após clonar)
uv run pre-commit install

# Rodar manualmente em todos os arquivos
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
| `azure-identity` | — | Autenticação via Service Principal |

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