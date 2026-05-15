# IAA-project2 — Crowd Counting com ML Clássico

Previsão do número de pessoas em imagens de multidão (dataset ShanghaiTech) usando features clássicas (HOG + LBP + histograma de cor) e modelos de regressão.

## Estrutura

```txt
data/ShanghaiTech/       # Dataset original (part_A e part_B)
src/
  preprocessing.py       # Extração de features e PCA
  models.py              # Treino e avaliação dos modelos de regressão
  data_loader.py         # Carregamento das imagens e ground truth
  eda.py                 # Análise exploratória
  models/
    regression_models.py
    metrics.py
    plots.py
processed/               # Datasets processados (uma subpasta por configuração)
models_output/           # Modelos treinados e resultados
outdated/                # Código e resultados de classificação (arquivado)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pré-processamento

Gera features HOG + LBP + histograma de cor, normaliza e aplica PCA. O output é guardado numa subpasta de `processed/` com nome gerado automaticamente a partir dos parâmetros.

```bash
# Configuração default (128x128, HOG 16px, PCA 95%, com augmentation)
python src/preprocessing.py

# HOG com células mais pequenas (mais detalhe)
python src/preprocessing.py --hog_pixels_per_cell 8 8

# PCA mais conservador
python src/preprocessing.py --pca_variance 0.99

# Sem data augmentation
python src/preprocessing.py --no_augment

# Imagem menor (mais rápido)
python src/preprocessing.py --img_size 64 64 --hog_pixels_per_cell 8 8

# Nome explícito para a subpasta de output
python src/preprocessing.py --hog_pixels_per_cell 8 8 --pca_variance 0.99 --name exp_hog8_pca99

# Só part_A
python src/preprocessing.py --part part_A

# Mudar nome do output
python src/preprocessing.py --name output_dir
```

### Parâmetros disponíveis

| Argumento | Default | Descrição |
|---|---|---|
| `--img_size W H` | `128 128` | Dimensão de redimensionamento |
| `--hog_pixels_per_cell W H` | `16 16` | Tamanho das células HOG |
| `--hog_cells_per_block W H` | `2 2` | Células por bloco HOG |
| `--hog_orientations N` | `9` | Nº de orientações HOG |
| `--lbp_radius R` | `3` | Raio LBP |
| `--lbp_n_bins N` | `64` | Bins do histograma LBP |
| `--pca_variance F` | `0.95` | Variância explicada pelo PCA |
| `--no_augment` | — | Desativa data augmentation |
| `--name STR` | auto | Nome da subpasta de output |
| `--part` | `both` | `part_A`, `part_B` ou `both` |
| `--out_dir` | `./processed` | Pasta raiz de output |
| `--data_root` | `./data/ShanghaiTech` | Pasta do dataset |

## Treino e Avaliação

Treina Ridge, Random Forest, XGBoost e SVR. Os resultados ficam em `models_output/<run>/`.

```bash
# Todos os datasets em processed/
python src/models.py

# Um dataset específico
python src/models.py --runs exp_hog8_pca99

# Vários datasets (compara no final)
python src/models.py --runs exp_hog8_pca99 exp_noaug original

# Pasta de output diferente
python src/models.py --out_dir ./resultados_finais
```

### Parâmetros disponíveis

| Argumento | Default | Descrição |
| --- | --- | --- |
| `--processed_dir` | `./processed` | Pasta raiz com os datasets processados |
| `--out_dir` | `./models_output` | Pasta de output dos modelos e resultados |
| `--runs` | todos | Subpastas a usar (nomes separados por espaço) |

## Comparação de Resultados

Após correr `models.py` para vários datasets, compara todos os resultados de uma vez.

```bash
# Comparar todos os runs em models_output/
python src/compare.py

# Comparar runs específicos
python src/compare.py --runs exp1_hog8_pca95 exp2_hog8_pca99 exp3_noaug

# Pasta de output diferente
python src/compare.py --out_dir ./comparacao_final
```

Gera em `models_output/comparacao/`:

- `comparacao_completa.csv` — todas as métricas de todos os modelos e runs
- `comparacao_melhores.csv` — melhor modelo (menor MAPE) por run
- `comparacao_todos_modelos.png` — barras agrupadas por métrica
- `comparacao_melhores.png` — melhor modelo de cada run lado a lado

### Parâmetros disponíveis

| Argumento | Default | Descrição |
| --- | --- | --- |
| `--models_dir` | `./models_output` | Pasta raiz com os resultados dos modelos |
| `--runs` | todos | Runs a comparar (nomes separados por espaço) |
| `--out_dir` | `<models_dir>/comparacao` | Pasta de output dos gráficos e CSVs |

## Métricas reportadas

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error
- **MdAPE** — Median Absolute Percentage Error
- **CV-MAE** — MAE em 5-fold cross-validation no treino
