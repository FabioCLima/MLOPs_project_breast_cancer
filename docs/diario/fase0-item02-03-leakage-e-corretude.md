# Fase 0 — Itens 2+3: Data leakage no validation split e corretude do treino

> **Backlog:** itens 2 e 3 · **Capítulo relacionado:** 4 (Dados de treinamento), seções de splitting e data leakage; cap. 5 (leakage em features); cap. 6 (avaliação offline)

## O bug central: um leakage que não aparece em nenhum erro

Antes desta mudança, o fluxo era:

```
split train/test → imputer.fit(train) → scaler.fit(train) → model.fit(train, validation_split=0.2)
```

Parece correto — os transformadores só viram o treino, certo? **Errado.** O `validation_split=0.2` do Keras separa a validação *depois* que o scaler já foi ajustado nos 100% do treino. Ou seja: a média e o desvio usados para escalar os dados de validação **foram calculados incluindo os próprios dados de validação**.

```
ANTES (leakage):
train (100%) ──fit imputer/scaler──► train escalado ──Keras separa──► train (80%) + val (20%)
                     ▲                                                        ▲
                     └────── estatísticas incluem estas linhas ───────────────┘

DEPOIS (correto):
raw ──split estratificado──► train (64%) │ val (16%) │ test (20%)
train ──fit imputer/scaler──► transform(train), transform(val), transform(test)
```

### Por que isso importa, se o efeito é pequeno?

Num dataset de 569 linhas com features bem comportadas, o impacto numérico é pequeno. Mas três coisas tornam o bug grave:

1. **A validação guia decisões.** O `EarlyStopping` monitora `val_loss` — uma `val_loss` otimista para o treino na época errada e seleciona o modelo errado. Leakage na validação não é só um número inflado num relatório; é um critério de decisão corrompido.
2. **O hábito escala mal.** Em datasets maiores, com features de alta variância, séries temporais ou grupos (pacientes com múltiplos exames!), esse mesmo padrão produz modelos que desabam em produção. O custo de fazer certo aqui é zero; o custo de carregar o hábito errado é alto.
3. **É invisível.** Nenhum teste de acurácia pega isso. Só revisão de código — ou um teste automatizado escrito de propósito, que é o que fizemos.

### Segundo bug: métricas que não correspondem ao modelo salvo

`EarlyStopping(restore_best_weights=True)` restaura os pesos do **melhor** epoch, mas o código salvava as métricas do **último** (`history.history[metric][-1]`). Resultado: o `metrics/training.json` descrevia um modelo que não era o que estava em `models/model.keras`. Em tracking de experimentos (Fase 2), isso contaminaria toda comparação entre runs. Correção: `best_epoch = argmin(val_loss)` e métricas desse epoch.

### Terceiro bug: ausência de estratificação

`train_test_split` sem `stratify` num dataset 63/37 e pequeno produz splits com proporções de classe visivelmente diferentes entre si — e as métricas mudam por sorteio, não por mérito do modelo. Agora os três splits (train/val/test) são estratificados pelo target.

## Simplificação: classificação binária como classificação binária

O modelo usava softmax com 2 saídas + `categorical_crossentropy` + `OneHotEncoder` do target. Para 2 classes isso é redundante — e o encoder causava um quarto bug: o app usava `OneHotEncoder.inverse_transform()` **diretamente sobre probabilidades**, um comportamento não documentado do sklearn.

A mudança para `Dense(1, activation="sigmoid")` + `binary_crossentropy`:

- elimina o OneHotEncoder inteiro (um artefato a menos para versionar, carregar e quebrar);
- torna a decisão explícita: `probabilidade >= 0.5` — e esse **threshold visível** é o gancho para a Fase 2, onde vamos calibrá-lo para o contexto clínico (recall da classe maligna);
- a saída passa a ser uma probabilidade interpretável de "benigno", que o app agora retorna junto com o label.

Limpezas incluídas: `params.pop("random_seed")` mutava o dicionário do chamador (efeito colateral clássico — trocado por leitura simples); o parâmetro `encoder` morto em `evaluate_model()` foi removido; `Input(shape=...)` substituiu o `input_shape` deprecado no Keras 3.

## O teste de leakage (`tests/test_no_leakage.py`)

A parte mais "sênior" desta mudança não é a correção — é a **prova automatizada de que a correção não regride**:

- `test_imputer_fitted_on_train_only` / `test_scaler_fitted_on_train_only`: comparam `imputer.statistics_` e `scaler.mean_` com as médias calculadas só no treino (devem bater) e com as médias do dataset completo (devem diferir).
- `test_scaled_train_is_standardized_but_val_test_are_not_exactly`: o treino escalado tem média exatamente 0 *por construção*; se val/teste também tiverem, o scaler os viu — leakage detectado.
- Testes de disjunção e estratificação dos splits.

Usa dados sintéticos com NaNs e classes desbalanceadas — roda em ~1s, sem depender do dataset real.

## Onde se encaixa no workflow

Esta é a fronteira **dados → treino** do pipeline. Tudo que vem depois (tracking na Fase 2, serving na Fase 3, monitoramento na Fase 5) reporta e compara métricas — e métricas só têm valor se o protocolo de avaliação for honesto. Por isso este item vem antes de qualquer ferramenta: **MLflow rastreando métricas vazadas é só leakage com dashboard.**

## Teoria ↔ prática (Chip Huyen)

- **Cap. 4 — Dados de treinamento:** a seção de *splitting* explica por que separar validação/teste antes de qualquer estatística global e por que estratificar com classes desbalanceadas. A lista de causas de leakage do **cap. 5** inclui literalmente o nosso caso: *"scaling before splitting"* — ajustamos a escala antes de separar a validação, só que escondido dentro do `validation_split` do Keras. Lição: leakage raramente está onde o código diz "split"; está onde uma estatística é calculada.
- **Cap. 6 — Avaliação offline:** a discussão sobre baselines e protocolos de avaliação assume que train/val/test são herméticos. E a recomendação de "comece simples" vale também para arquitetura: sigmoid binária é a versão mais simples que resolve o problema.

## Como validar

```bash
uv run pytest tests/ -q          # 5 passed
uv run python -m src.data_loading.load_data
uv run python -m src.data_preprocessing.preprocess_data
uv run python -m src.feature_engineering.engineer_features
uv run python -m src.model_training.train_model      # loga best_epoch
uv run python -m src.model_evaluation.evaluate_model # report com malignant/benign
```

Resultado pós-correção (teste, threshold 0.5): recall(malignant) = 0.98, precision(malignant) = 0.89, accuracy = 0.95.

## Lição para levar

Data leakage não é um tipo de bug — é uma *família* de bugs com o mesmo sintoma (métricas otimistas) e nenhum stack trace. A defesa não é atenção redobrada; é **arquitetura** (separar splits antes de qualquer `fit`) mais **testes que provam a propriedade**. Se você só consegue afirmar "não tem leakage" lendo o código com cuidado, você ainda não tem a garantia — tem uma esperança.
