# Fase 1 — Item 7: Contrato de dados com Pandera

> **Backlog:** item 7 · **Capítulo relacionado:** 4 (Dados de treinamento); cap. 8 (falhas de sistemas de ML)

## O que foi feito

- `src/config/features.py`: lista canônica das 30 features + nome do target + mapa de labels. **Única fonte de verdade** — pipeline, schemas e (futuramente) o serving importam daqui; ninguém mais depende de `sklearn.datasets` em runtime.
- `src/data_validation/schemas.py`: três schemas Pandera, um por fronteira do pipeline:

| Schema | Onde valida | Regras |
|--------|-------------|--------|
| `RAW_SCHEMA` | entrada do preprocess | 30 features float ≥ 0 (NaN permitido), target ∈ {0,1}, **nenhuma coluna extra** (`strict`) |
| `PREPROCESSED_SCHEMA` | entrada do features | idem, mas **NaN proibido** (pós-imputação) |
| `PROCESSED_SCHEMA` | entrada de train/evaluate | floats sem NaN (escalados podem ser negativos), target ∈ {0,1} |

- `validate(data, schema, stage)` com `lazy=True`: acumula **todas** as violações num único relatório com o nome do estágio, em vez de morrer na primeira.
- `tests/test_schemas.py`: 6 testes — coluna faltando, coluna extra, valor negativo, target não-binário, NaN pós-imputação.

## Por quê

Em sistemas de ML, o tipo de bug mais caro não é o que lança exceção — é o que **passa silenciosamente**. Exemplos que o schema agora bloqueia:

- Um CSV regenerado com uma coluna a mais (ou com `target` como string `"0"/"1"`) treinaria sem erro e produziria um modelo lixo.
- Um bug no imputer que deixasse NaNs passaria pelo Keras (que aceita NaN e produz loss NaN 30 épocas depois — boa sorte no debug).
- Uma `mean area` negativa é fisicamente impossível; se aparecer, o problema está na *fonte* dos dados, e é ali que se quer descobrir, não na métrica de produção três semanas depois.

A escolha de `strict=True` (rejeitar colunas extras) é deliberada: colunas inesperadas costumam ser sintoma de join errado ou leakage de target disfarçado.

### A progressão dos schemas conta a história do pipeline

Repare que os três schemas codificam **o que cada estágio promete**: o preprocess promete eliminar NaNs (por isso o estágio seguinte os proíbe); o features promete escalar (por isso o `ge(0)` é relaxado). O contrato de dados é a documentação executável do pipeline.

## Onde se encaixa no workflow

```
dado entra ─► RAW_SCHEMA ─► preprocess ─► PREPROCESSED_SCHEMA ─► features ─► PROCESSED_SCHEMA ─► train/evaluate
```

E olhando adiante: o **serving (Fase 3)** validará os mesmos 30 nomes via Pydantic — importando de `src/config/features.py`. Treino e inferência compartilhando o mesmo contrato é a defesa número 1 contra *training-serving skew*. O **monitoramento (Fase 5)** também usa a mesma lista para comparar distribuições feature a feature.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 8** classifica as falhas de sistemas de ML e mostra que a maioria das falhas reais são "falhas comuns de software" e problemas de dados — não de modelo. A primeira linha de defesa que o capítulo cita para *data distribution shifts* detectáveis de imediato são **validações de schema e de range**: exatamente o que `ge(0)` + `isin([0,1])` + `strict` implementam.
- **Cap. 4**: a discussão de qualidade de dados de treinamento ("garbage in, garbage out" com nuance) justifica validar *antes* de treinar, não depois de avaliar.

## Como validar

```bash
uv run pytest tests/test_schemas.py -q   # 6 passed
uv run dvc repro -f                       # pipeline inteiro com contratos ativos
# quebre de propósito: adicione uma coluna no raw.csv e rode o preprocess —
# o erro nomeia o estágio, a coluna e a regra violada
```

## Lição para levar

Schema de dados é o `assert` que faltava entre estágios: barato de escrever, e transforma "métrica estranha em produção" (dias de investigação) em "erro claro no estágio X, coluna Y" (minutos). Regra prática: **toda fronteira onde um DataFrame troca de dono merece um contrato** — entre estágios do pipeline, e principalmente entre o mundo externo e o seu modelo.
