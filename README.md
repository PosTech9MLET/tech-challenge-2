# 🛒 Tech Challenge — Fase 2
### Sistema de Recomendação de Produtos | POSTECH ML Engineering

---

## Visão Geral

Sistema de recomendação de produtos baseado no comportamento de compra de usuários do **Instacart**, utilizando redes neurais (MLP/Embedding-based) treinadas com PyTorch. O pipeline completo é containerizado com Docker, dados versionados com DVC no Azure Blob Storage, experimentos rastreados com MLflow e o código segue padrões profissionais de Clean Code.

> **Dataset:** [Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)  
> **Grupo:** PosTech9MLET — FIAP POSTECH ML Engineering

---

## Integrantes

| Nome | RM |
|---|---|
| Gabriel Freitas | RM370409 |
| Diego | — |
| Deyvid Manhães | RM371074 |
| Lucas Molitor | — |

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

## Como Funciona a Autenticação

O projeto usa um **Service Principal** do Azure — uma identidade de serviço com permissões controladas que permite qualquer pessoa rodar o projeto **sem precisar de conta Azure própria**.

```
Credenciais do Service Principal no .env
             ↓
   azure-identity autentica
             ↓
   Key Vault → busca AZURE-STORAGE-KEY
             ↓
   DVC pull → baixa os CSVs do Blob Storage
             ↓
        Projeto pronto
```

> Não é necessário instalar o Azure CLI nem ter uma conta Azure pessoal.

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

Abra o `.env` e preencha com as credenciais abaixo:

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

# SERVICE PRINCIPAL
AZURE_CLIENT_ID=<ver seção Avaliação>
AZURE_CLIENT_SECRET=<ver seção Avaliação>
AZURE_TENANT_ID=<ver seção Avaliação>
```

> `AZURE_STORAGE_KEY` deve ficar **vazio** — o sistema busca automaticamente do Key Vault usando o Service Principal.

### Passo 5 — Instale os hooks de qualidade de código

```bash
uv run pre-commit install
```

Este comando instala os hooks do `ruff` no Git local. A partir daí, o linting roda automaticamente em todo `git commit`.

### Passo 6 — Configure o DVC e baixe os dados

Execute o script de configuração do DVC — ele busca automaticamente a access key do Azure Key Vault usando o Service Principal e configura o remote local:

```bash
uv run python scripts/setup_dvc.py
```

Em seguida, baixe os dados:

```bash
uv run dvc pull
```

Os 6 CSVs do Instacart serão baixados do Azure Blob Storage para `data/raw/`.

### Passo 7 — Valide o ambiente

```bash
uv run python scripts/validate_env.py
```

Se tudo estiver correto, a saída será:

```
✓ torch
✓ sklearn
✓ mlflow
...
✓ Ambiente de desenvolvimento validado com sucesso!
```

---
### Passo 8 — Execute o pipeline completo

```bash
uv run dvc repro
```

Isso executa os 4 stages em ordem: `preprocess → feature_eng → train → evaluate`.

Para visualizar os experimentos e o modelo registrado no MLflow:

```bash
uv run mlflow ui --backend-store-uri mlruns/
```

Acesse `http://127.0.0.1:5000` no navegador.

---

## Avaliação

> Esta seção é destinada ao professor avaliador e membros externos que precisam rodar o projeto sem configuração adicional.

O projeto utiliza um Service Principal criado especificamente para avaliação, com permissões de **somente leitura** no Blob Storage e Key Vault. Preencha o `.env` com as credenciais abaixo:

```dotenv
AZURE_CLIENT_ID=1baf74a9-4ad6-4d3d-88db-c3070e970684
AZURE_CLIENT_SECRET=<SUBSTITUIR — fornecido junto com o vídeo>
AZURE_TENANT_ID=11dbbfe2-89b8-4549-be10-cec364e59551
```

> O `AZURE_CLIENT_SECRET` será entregue junto com o vídeo STAR por canal seguro, para evitar exposição pública no repositório.

---

## Fluxo de Desenvolvimento

Para contribuir com o projeto, siga este fluxo:

```bash
# 1. Crie uma branch para sua feature
git checkout -b feat/nome-da-feature

# 2. Desenvolva e teste localmente
uv run python scripts/validate_env.py

# 3. O pre-commit roda automaticamente ao commitar
git add .
git commit -m "feat: descrição da mudança"

# 4. Abra um Pull Request para main
git push origin feat/nome-da-feature
```

### Convenção de commits

O projeto usa **commits semânticos**:

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

O projeto usa **ruff** para linting e formatação automática.

```bash
# Verificar erros
uv run ruff check .

# Corrigir automaticamente
uv run ruff check . --fix

# Rodar pre-commit manualmente em todos os arquivos
uv run pre-commit run --all-files
```

Regras ativas: `E` (estilo), `F` (lógico), `I` (imports), `N` (naming), `UP` (modernização), `D` (docstrings Google style).

---

## Infraestrutura Azure

| Recurso | Nome | Finalidade |
|---|---|---|
| Storage Account | `stgtechchallenge` | Armazenamento dos CSVs via DVC |
| Blob Container | `tech-challenge-f2` | Container dos dados e artefatos |
| Key Vault | `techchallengevaults` | Gerenciamento de secrets |
| Service Principal | `tech-challenge-sp` | Autenticação sem conta pessoal |

### Secrets no Key Vault

| Secret | Descrição |
|---|---|
| `AZURE-STORAGE-KEY` | Access key do Storage Account |
| `AZURE-STORAGE-ACCOUNT` | Nome do Storage Account |
| `AZURE-CONTAINER-NAME` | Nome do container Blob |

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

> Os dados são gerenciados pelo **DVC** e armazenados no Azure Blob Storage. Nunca são commitados diretamente no Git.

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
| Etapa 3 | Feature Engineering + DVC Pipeline | ✅ Concluído |
| Etapa 3 | Docker (multi-stage) | 🔄 Em progresso |
| Etapa 4 | Modelagem (Baseline + MLP PyTorch) | ✅ Concluído  |
| Etapa 4 | MLflow Model Registry | ✅ Concluído |
| Entrega | README + Vídeo STAR | ⏳ Pendente |

---

## Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.