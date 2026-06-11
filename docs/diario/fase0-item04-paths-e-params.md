# Fase 0 — Item 4: Centralizar caminhos e parâmetros

> **Backlog:** item 4 · **Capítulo relacionado:** 10 (Infraestrutura e ferramentas); cap. 2 (manutenibilidade como requisito de sistema)

## O que foi feito

- `src/config/paths.py` agora é a **única fonte de verdade** para todos os caminhos: dados (raw, preprocessed, processed × train/val/test), artefatos, modelo e métricas. Todos derivam de `PROJECT_ROOT = Path(__file__).parent...`, então funcionam de qualquer working directory.
- Novo `src/config/params.py` com `load_params(section)` — as três cópias de `load_params()` espalhadas pelos módulos viraram um helper único.
- Artefatos renomeados: `[features]_mean_imputer.joblib` → `features_mean_imputer.joblib` (colchetes são metacaracteres de glob — `ls artifacts/[features]*` não faz o que parece — e quebram URLs e shells).
- Cada `save_*` garante `mkdir(parents=True, exist_ok=True)` antes de escrever.

## Por quê

O sintoma clássico do problema anterior: o pipeline só funcionava se executado **exatamente da raiz do projeto**. `pd.read_csv("data/raw/raw.csv")` resolve o caminho relativo ao *current working directory* — que muda entre dev local, cron, container, runner de CI e orquestrador. Cada um desses contextos teria um erro diferente de "arquivo não encontrado".

Strings de caminho duplicadas têm um segundo custo: o caminho `data/preprocessed/train_preprocessed.csv` aparecia escrito em `preprocess_data.py` (escrita) **e** em `engineer_features.py` (leitura). Um typo em um deles cria um bug silencioso de contrato entre estágios. Com a constante compartilhada `TRAIN_PREPROCESSED_PATH`, o contrato é o import — o Python verifica por nós.

### Um bug real apareceu durante a refatoração

Ao trocar o `load_params()` local pelo helper com seção, o `main()` do treino ficou chamando `load_params()` sem argumento — que agora retorna o YAML inteiro, não a seção `train` — e o pipeline quebrou com `KeyError: 'random_seed'`. Foi pego porque **rodamos o pipeline completo como validação do commit**. Lição embutida: refatoração "mecânica" também precisa de validação de execução, não só de lint. (E na Fase 1, o teste de pipeline pegará isso automaticamente.)

## Onde se encaixa no workflow

Este item prepara o terreno para dois passos futuros:

- **DVC (Fase 1):** o `dvc.yaml` declara deps/outs por caminho; com os caminhos centralizados, há um único lugar para manter o espelhamento entre código e pipeline.
- **Docker/K8s (Fase 4):** containers executam com working directories arbitrários; código cwd-independente é pré-requisito para containerizar sem gambiarras de `WORKDIR`.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 2** lista *manutenibilidade* entre os quatro requisitos de um sistema de ML: "é importante estruturar workloads e configurar a infraestrutura de modo que possam ser reproduzidos". Configuração espalhada (caminhos hardcoded em 5 arquivos) é o oposto disso.
- **Cap. 10** discute a separação entre código e configuração como base da camada de desenvolvimento. `params.yaml` (hiperparâmetros) + `paths.py` (layout do projeto) são as duas metades dessa separação aqui.

## Como validar

```bash
cd /tmp && uv run --project <raiz-do-projeto> python -m src.data_preprocessing.preprocess_data
# roda de qualquer diretório
uv run pytest tests/ -q   # 5 passed
uv run ruff check .       # All checks passed!
```

## Lição para levar

Caminho hardcoded é dívida que não dói no notebook e explode no container. A regra prática: **nenhum módulo deve saber onde o projeto está — apenas o `config` sabe.** E quando uma string aparece em dois arquivos, ela está implorando para virar uma constante importada.
