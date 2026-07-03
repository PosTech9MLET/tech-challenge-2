#!/bin/bash
# Entrypoint do container de treino.
# Inicializa o DVC sem SCM e executa o pipeline completo.
set -e

echo "Inicializando DVC sem SCM..."
dvc init --no-scm -f

echo "Configurando remote Azure..."
dvc remote add -d azure_remote "azure://${AZURE_CONTAINER_NAME}/dvc"
dvc remote modify azure_remote account_name "${AZURE_STORAGE_ACCOUNT}"

echo "Buscando chave do Key Vault e configurando remote..."
python scripts/setup_dvc.py

echo "Baixando dados brutos do Azure Blob Storage..."
dvc pull data/raw/orders.csv \
         data/raw/order_products__prior.csv \
         data/raw/order_products__train.csv \
         data/raw/products.csv \
         data/raw/aisles.csv \
         data/raw/departments.csv \
         --force

echo "Executando pipeline completo..."
dvc repro
