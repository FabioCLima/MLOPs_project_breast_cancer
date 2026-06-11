# Fase 1 — Item 8: Suite de testes + mypy

> **Backlog:** item 8 · **Capítulo relacionado:** 6 (Desenvolvimento de modelos e avaliação offline); cap. 11 (qualidade de engenharia)

## O que foi feito

A suite saiu de 0 para **19 testes em ~4s**, em quatro arquivos:

| Arquivo | O que protege |
|---------|---------------|
| `test_no_leakage.py` (item 2) | Imputer/scaler nunca veem validação/teste; splits disjuntos e estratificados |
| `test_schemas.py` (item 7) | Contrato de dados: coluna faltando/extra, negativos, target não-binário, NaN |
| `test_pipeline_units.py` | Split reproduzível com seed; proporções 64/16/20; shapes; modelo binário compila e treina 1 época com saída em [0,1] |
| `test_model_service.py` | Lógica de inferência com **artefatos falsos**: colunas da resposta, decisão por threshold, passthrough de probabilidade |

Mudanças estruturais que os testes exigiram (e que melhoram o design):

- **`ModelService` extraído para `app/model_service.py`** — módulo sem efeitos colaterais de import. Antes, `from app.main import ModelService` executava `ModelService()` no nível do módulo e exigia modelo treinado em disco; agora a classe é importável e testável a seco. (É também meio caminho para o application factory da Fase 3.)
- O app deixou de chamar `load_breast_cancer()` para validar colunas — usa `FEATURE_COLUMNS` do contrato canônico (item 7).
- **mypy ativado**: configuração no `pyproject.toml` (`ignore_missing_imports` para libs científicas sem stubs, `check_untyped_defs`) + hook no pre-commit. Encontrou 2 bugs reais no app: `file.filename` pode ser `None` (crash em upload sem nome) e o `exc_info=True` que o loguru ignora silenciosamente (trocado por `logger.exception`).

## Por quê — e por que esses testes e não outros

Teste de ML não é testar se "o modelo acerta" — é testar as **propriedades do sistema que cercam o modelo**:

1. **Propriedades invariantes** (leakage, disjunção de splits): nunca podem quebrar, em nenhuma mudança futura.
2. **Contratos** (schemas, colunas da resposta do serviço): o que o resto do mundo consome.
3. **Mecânica mínima do modelo** (compila, treina 1 época, probabilidade em [0,1]): pega erros de wiring sem custar minutos de treino. Note o truque: não testamos acurácia em testes unitários — acurácia é métrica de experimento (MLflow, Fase 2), não de CI.

O `ModelService` com fakes ilustra o padrão mais útil de teste em ML serving: o modelo é substituído por uma função determinística (`probabilidade = primeira feature`), o que permite verificar a **lógica de decisão** (threshold, labels, formato) com precisão exata, independente de pesos treinados.

## Onde se encaixa no workflow

Os testes são o **gate de qualidade** que o CI (item 9) vai executar em cada push. Sem eles, o CI só verificaria estilo; com eles, verifica as propriedades que fazem o pipeline ser confiável. E cada item futuro (FastAPI, MLflow, drift) chega já com seus testes — o padrão está estabelecido.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 6** discute *model offline evaluation* e os testes de sanidade que vão além de métricas: testes de invariância e expectativas direcionais. Nosso `test_threshold_decision` é um teste direcional puro: probabilidade maior ⇒ classe benigna, sempre.
- **Cap. 11** (e o cap. 1) batem na tecla de que sistemas de ML são software: merecem a mesma engenharia. Uma suite que roda em segundos é o que permite refatorar sem medo — exatamente o que faremos na migração para FastAPI.

## Como validar

```bash
uv run pytest tests/ -q       # 19 passed, ~4s
uv run mypy src/ app/         # Success: no issues found
uv run pre-commit run --all-files
```

## Lição para levar

Em ML, os bugs mais perigosos não derrubam o programa — degradam silenciosamente uma métrica. Por isso a suite não pergunta "o modelo é bom?", e sim "**as propriedades que tornam a avaliação confiável continuam valendo?**". Teste propriedades e contratos; deixe a qualidade do modelo para o experiment tracking.
