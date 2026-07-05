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

## Execução comDocker

O projeto disponibiliza uma imagem Docker **multi-stage** e serviços Docker
Compose para executar o pipeline DVC em ambiente isolado e acompanhar
experimentos no MLflow.

A arquitetura separa responsabilidades:

- A imagem Docker contém somente código e dependências de execução.
- Dados, modelos, cache DVC e metadados Git são montados no container apenas
  durante a execução.
- Credenciais Azure permanecem no arquivo `.env` local e nunca são inseridas
  na imagem Docker.
- O MLflow utiliza um volume Docker persistente para armazenar experimentos,
  métricas, artefatos e modelos registrados.

### Pré-requisitos

Antes de usar Docker, garanta que os itens abaixo estejam disponíveis:

- Docker Desktop em execução;
- Docker Compose v2;
- Git;
- `uv` instalado para baixar os dados via DVC no host;
- Arquivo `.env` configurado com as credenciais Azure quando for necessário
  executar `dvc pull`.

Verifique a instalação do Docker:

```bash
docker version
docker compose version
```

---

### Estrutura Docker

| Arquivo | Responsabilidade |
|---|---|
| `Dockerfile` | Cria uma imagem multi-stage com Python, `uv`, DVC, MLflow, PyTorch e dependências de produção. |
| `.dockerignore` | Impede que secrets, dados, modelos, ambientes virtuais e caches locais entrem na imagem. |
| `docker-compose.yml` | Orquestra os serviços `mlflow` e `train`. |
| `.env.docker.example` | Template de variáveis de ambiente usadas pelo container de treino. |
| `.env.docker` | Arquivo local usado pelo Docker Compose. Nunca deve ser commitado. |

A imagem final executa com o usuário não privilegiado `app`, evitando que o
pipeline rode como `root`.

---

### Configuração inicial do Docker

Crie o arquivo local usado pelo Docker Compose.

No Windows PowerShell:

```powershell
Copy-Item .env.docker.example .env.docker
```

No Linux ou macOS:

```bash
cp .env.docker.example .env.docker
```

O arquivo `.env.docker` já contém os caminhos e a URL interna do MLflow:

```dotenv
SEED=42
DATA_INPUT_PATH=data/raw
DATA_OUTPUT_PATH=data/processed
EARLY_STOPPING_PATIENCE=5

AZURE_STORAGE_ACCOUNT=stgtechchallenge
AZURE_CONTAINER_NAME=tech-challenge-f2
AZURE_STORAGE_KEY=

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_ARTIFACT_LOCATION=mlflow-artifacts:/
```

> Não adicione `AZURE_CLIENT_SECRET`, `AZURE_STORAGE_KEY` ou outras credenciais
> reais ao `.env.docker`. O download de dados é executado no host usando o
> arquivo `.env` local.

Valide a configuração final do Docker Compose:

```bash
docker compose config
```

O resultado deve listar os serviços `mlflow` e `train`.

---

### Construir a imagem

Construa a imagem Docker:

```bash
docker compose build
```

A primeira execução pode demorar alguns minutos, pois o Docker precisa baixar
dependências grandes, como PyTorch, SciPy, PyArrow, DVC e MLflow.

Em builds posteriores, o Docker reutiliza cache sempre que possível.

Use rebuild sem cache apenas quando houver problema de dependência ou alteração
relevante no `Dockerfile`:

```bash
docker compose build --no-cache
```

---

### Subir o MLflow

Inicie o serviço MLflow em segundo plano:

```bash
docker compose up -d mlflow
```

Verifique o estado do serviço:

```bash
docker compose ps
```

O status esperado é:

```text
healthy
```

Acesse a interface em:

```text
http://localhost:5000
```
---

### Validar o ambiente Docker

```bash
docker compose run --rm --no-deps train python -c "import torch, sklearn, pandas, mlflow, dvc; print('Dependencias Docker OK')"
docker compose run --rm --no-deps train dvc dag
```

### Baixar os dados com DVC

Configure as credenciais Azure no arquivo `.env` local e execute no host:

```bash
uv run python scripts/setup_dvc.py
uv run dvc pull
```

---

### Visualizar o pipeline DVC sem executar treinamento

O comando abaixo mostra o DAG do pipeline sem processar dados ou treinar modelos:

```bash
docker compose run --rm --no-deps train dvc dag
```

O fluxo esperado é:

```text
preprocess → feature_eng → train → evaluate
```

O container recebe `.git` somente para leitura durante a execução. Isso é
necessário porque o DVC utiliza metadados do repositório Git, mas o histórico
Git não é incluído na imagem Docker.

---

### Executar o pipeline no container

Com o serviço MLflow ativo e os dados disponíveis em `data/raw`:

```bash
docker compose run --rm --no-deps train dvc repro
```

Os artefatos do pipeline serão persistidos localmente em `data/` e `models/`.
Os experimentos, métricas e modelos registrados ficam disponíveis no MLflow.

### Encerrar os serviços

```bash
docker compose down
```

> Não use `docker compose down -v` a menos que queira remover também o
> histórico persistido do MLflow.

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
| Etapa 3 | Docker (multi-stage) | ✅ Concluído  |
| Etapa 4 | Modelagem (Baseline + MLP PyTorch) | ✅ Concluído  |
| Etapa 4 | MLflow Model Registry | ✅ Concluído |
| Entrega | README + Vídeo STAR | ⏳ Pendente |

---

## Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.