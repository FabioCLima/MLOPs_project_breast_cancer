# ADR 0002 — FastAPI + Pydantic v2 no lugar de Flask

**Status:** aceito · **Data:** 2026-06-11

## Contexto

O app Flask original carregava o modelo no import, validava entrada com `if`s manuais, retornava HTML com erro embutido e usava `OneHotEncoder.inverse_transform` sobre probabilidades (comportamento não documentado).

## Decisão

Migrar para FastAPI com application factory (`create_app`), lifespan para carga do modelo, e contrato Pydantic v2 **gerado da lista canônica de features** (`src/config/features.py`) — a mesma usada pelo pipeline e pelo Pandera.

## Racional

- Numa API de ML, validação de entrada é a maior superfície de risco; Pydantic a torna declarativa, com HTTP 422 nomeando o campo inválido.
- OpenAPI/Swagger gerados do código = documentação que não desatualiza.
- Factory + injeção do `ModelService` tornam o serving testável sem artefatos reais (11 testes de contrato rodam a seco).
- Contrato compartilhado com o pipeline elimina training-serving skew de schema por construção.

## Consequências

- (+) 422/400 semânticos; testes rápidos; demo interativa em /docs.
- (−) A UI HTML de upload foi removida — o Swagger cumpre o papel de demo; o endpoint CSV (`/predict/batch`) cobre o caso de uso.
