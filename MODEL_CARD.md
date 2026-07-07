# Model Card — RecommenderMLP

> Documento de transparência do modelo de recomendação de produtos
> desenvolvido para o Tech Challenge Fase 2 — POSTECH ML Engineering.

---

## Visão Geral do Modelo

| Campo | Detalhe |
|---|---|
| **Nome** | RecommenderMLP |
| **Versão** | 3 |
| **Tipo** | MLP Embedding-based (PyTorch) |
| **Tarefa** | Recomendação de produtos — previsão de recompra |
| **Dataset** | Instacart Online Grocery Basket Analysis |
| **Registrado em** | MLflow Model Registry — `recommender-mlp` v3 |
| **Stage** | Production |
| **Data de treino** | Junho 2026 |

---

## Descrição do Modelo

O RecommenderMLP é uma rede neural que aprende representações densas
(embeddings) de usuários e produtos a partir do histórico de compras do
Instacart. Dado um par (usuário, produto), o modelo prevê a probabilidade
de recompra daquele produto no próximo pedido do usuário.

### Arquitetura

```
user_id  → Embedding(n_users, 32)  ─┐
                                      ├─ concat(64) → Linear(256) → ReLU → Dropout(0.3)
product_id → Embedding(n_products, 32) ─┘              → Linear(128) → ReLU → Dropout(0.3)
                                                        → Linear(64)  → ReLU → Dropout(0.3)
                                                        → Linear(1)   → sigmoid → P(recompra)
```

### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| `embedding_dim` | 32 |
| `hidden_layers` | [256, 128, 64] |
| `dropout` | 0.3 |
| `lr` | 0.001 |
| `batch_size` | 1024 |
| `epochs` | 30 (com early stopping) |
| `early_stopping_patience` | 5 |
| `seed` | 42 |
| `negative_ratio` | 20 |

---

## Dados de Treinamento

### Dataset

O Instacart Online Grocery Basket Analysis contém pedidos reais de
usuários de um supermercado online norte-americano.

| Arquivo | Descrição | Volume |
|---|---|---|
| `orders.csv` | Pedidos por usuário | ~3.4M pedidos, 206.209 usuários |
| `order_products__prior.csv` | Histórico de compras | ~32.4M interações |
| `order_products__train.csv` | Pedidos de rótulo | ~1.4M interações |
| `products.csv` | Catálogo de produtos | 49.688 produtos |

### Pré-processamento

- Valores nulos em `days_since_prior_order` preenchidos com a mediana
- Nomes de produtos normalizados para minúsculas
- Flag `is_first_order` criada para identificar primeiros pedidos
- Flag `is_organic` criada a partir do nome do produto

### Feature Engineering

Três grupos de features foram construídos:

**Features de usuário (6 features):**
- `total_orders` — total de pedidos no histórico
- `avg_days_since_prior` — intervalo médio entre pedidos
- `std_days_since_prior` — variabilidade do intervalo
- `avg_hour` — horário médio de compra
- `avg_reorder_rate` — taxa média de recompra

**Features de produto (7 features):**
- `product_order_count` — total de pedidos que incluem o produto
- `product_reorder_rate` — taxa de recompra global do produto
- `avg_cart_position` — posição média no carrinho
- `aisle_id`, `department_id` — localização na loja
- `is_organic` — se o produto é orgânico

**Features do par usuário-produto:**
- `up_order_count` — vezes que esse usuário comprou esse produto
- `up_avg_cart_position` — posição média desse produto para esse usuário
- `up_order_rate` — frequência relativa de compra do par

### Splits

| Conjunto | Amostras | Proporção |
|---|---|---|
| Treino | 4.846.340 | 70% dos usuários |
| Validação | 1.039.230 | 15% dos usuários |
| Teste | 1.037.515 | 15% dos usuários |

A divisão é feita **por usuário** (não por linha) para evitar data leakage.
Para cada interação positiva (recompra real), 20 interações negativas são
amostradas aleatoriamente do catálogo (negative sampling com ratio 1:20).

---

## Performance

### Métricas de Avaliação

As métricas são calculadas no conjunto de teste, comparando o MLP com
um baseline de popularidade global (PopularityBaseline).

| Métrica | Baseline @10 | MLP @10 | Ganho |
|---|---|---|---|
| Precision@K | 7.45% | 58.54% | +51.1pp |
| Recall@K | 7.13% | 68.64% | +61.5pp |
| NDCG@K | 29.18% | 95.36% | +66.2pp |
| Hit Rate@K | 46.41% | 99.99% | +53.6pp |

**Métricas completas por K:**

| K | Precision | Recall | NDCG | Hit Rate |
|---|---|---|---|---|
| 5 | 76.38% | 51.21% | 96.99% | 99.81% |
| 10 | 58.54% | 68.64% | 95.36% | 99.99% |
| 20 | 39.93% | 82.97% | 93.42% | 100.00% |

### Baseline de comparação

| K | Precision | Recall | NDCG | Hit Rate |
|---|---|---|---|---|
| 5 | 9.88% | 4.89% | 26.88% | 37.72% |
| 10 | 7.45% | 7.13% | 29.18% | 46.41% |
| 20 | 5.37% | 9.93% | 30.27% | 53.19% |

### Observação sobre as métricas

As métricas de Hit Rate do MLP são muito elevadas porque o problema
avaliado é de **reordem** — o modelo é testado sobre produtos que o
usuário já comprou antes, em um contexto em que os padrões de recompra
são muito regulares (taxa de recompra global de 59% no dataset). Isso
é uma característica inerente ao dataset Instacart, não um artefato do
modelo.

---

## Limitações

**Cobertura do catálogo:** o modelo só consegue recomendar produtos que
já existiam no histórico de treino. Produtos novos (cold start) não
possuem embedding e não podem ser recomendados sem retreinamento.

**Usuários novos:** da mesma forma, novos usuários sem histórico não
possuem embedding. Para esses casos, o PopularityBaseline é uma
alternativa viável.

**Dados geográficos:** o dataset é exclusivamente de usuários
norte-americanos, e os padrões de compra podem não generalizar para
outros mercados ou culturas alimentares.

**CPU-only:** o modelo foi treinado e otimizado para CPU. Em produção
com alto volume de requisições, seria necessário migrar para GPU ou
otimizar a inferência com técnicas como quantização.

**Atualização dos embeddings:** os embeddings aprendem padrões do
histórico disponível no momento do treino. Mudanças sazonais de
comportamento (ex: pandemia, inflação) requerem retreinamento periódico.

**Avaliação off-policy:** as métricas são calculadas sobre interações
históricas, não sobre experimentos A/B reais. O impacto real em um
sistema de produção pode diferir das métricas reportadas.

---

## Vieses Conhecidos

**Viés de popularidade:** produtos mais populares globalmente tendem a
receber scores mais altos, o que pode prejudicar a descoberta de itens
de nicho relevantes para determinados usuários.

**Viés de frequência:** usuários com mais pedidos no histórico têm
embeddings mais bem calibrados. Usuários ocasionais (poucos pedidos)
podem receber recomendações de menor qualidade.

**Viés de categoria:** o dataset tem concentração em categorias como
Produce e Dairy & Eggs. Produtos de categorias menos frequentes podem
ser sub-representados nas recomendações.

---

## Uso Pretendido

**Casos de uso recomendados:**
- Sugestão de produtos para reposição no carrinho de compras
- Personalização da página inicial de um supermercado online
- Notificações de reabastecimento para usuários recorrentes

**Casos de uso não recomendados:**
- Recomendação de produtos completamente novos para o usuário
- Aplicação em mercados fora do contexto de supermercado online
- Decisões de alto impacto sem supervisão humana

---

## Informações Técnicas

| Campo | Detalhe |
|---|---|
| **Framework** | PyTorch ≥ 2.12.0 (CPU-only) |
| **Serialização** | `state_dict` (.pt) + encoders (.pkl) |
| **Design Pattern** | Factory Pattern (`ModelFactory`) para criação de modelos |
| **Versionamento** | MLflow Model Registry — fluxo challenger → champion (aliases) |
| **Pipeline** | DVC (`dvc repro`) |
| **Reprodutibilidade** | Seed 42 fixada em todas as etapas |

### Como carregar o modelo

```python
import pickle
import torch
from src.models.mlp import RecommenderMLP
from src.training.trainer import MLPTrainer

# Carregar encoders
with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Reconstruir arquitetura (mesmos params do treino)
model = RecommenderMLP(
    n_users=206209,
    n_products=49688,
    embedding_dim=32,
    hidden_layers=[256, 128, 64],
    dropout=0.3,
)

# Carregar pesos
trainer = MLPTrainer.load("models/mlp_best.pt", model=model)
```

---

## Responsáveis

| Nome | Papel |
|---|---|
| Gabriel Freitas |  |
| Deyvid Manhães | |
| Diego | — |
| Lucas Molitor | — |

**Instituição:** FIAP POSTECH ML Engineering — Turma 9MLET
**Contato:** via repositório GitHub `PosTech9MLET/tech-challenge-2`
