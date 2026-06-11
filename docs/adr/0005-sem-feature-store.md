# ADR 0005 — Sem feature store (Feast)

**Status:** aceito · **Data:** 2026-06-11

## Contexto

Feature stores (Feast, Tecton) aparecem em toda lista de "stack moderna de MLOps", e a tentação de adicionar uma ao portfólio é real.

## Decisão

Não usar. O contrato de features vive em `src/config/features.py` (única fonte de verdade para pipeline, Pandera e Pydantic) e os transformadores (imputer/scaler) são artefatos versionados na run do MLflow.

## Racional

Uma feature store resolve três problemas específicos:

1. **Consistência online/offline** — a mesma feature computada no treino (batch) e no serving (tempo real) por caminhos de código diferentes;
2. **Reuso entre times/modelos** — catálogo de features compartilhado;
3. **Point-in-time correctness** — joins históricos sem leakage temporal.

Este projeto não tem **nenhum** dos três: as 30 features chegam prontas no request (não são computadas no serving), há um modelo e um time, e não há dimensão temporal. Adotar Feast aqui adicionaria um registry, um online store e sincronização de infra para resolver problemas inexistentes — over-engineering que, numa entrevista, sinaliza pior que a ausência.

O problema real de consistência treino/serving que o projeto TEM (mesmo schema, mesmos transformadores) está resolvido pelo contrato canônico compartilhado + artefatos na run.

## Consequências

- (+) Menos infraestrutura; o mecanismo de consistência é visível e testável.
- (−) Se o sistema evoluísse para features computadas (ex.: agregações do histórico do paciente) ou múltiplos modelos consumindo as mesmas features, esta decisão deve ser revisitada — os critérios objetivos estão listados acima.
