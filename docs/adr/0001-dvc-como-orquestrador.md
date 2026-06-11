# ADR 0001 — DVC como orquestrador do pipeline (não Airflow/Prefect)

**Status:** aceito · **Data:** 2026-06-11

## Contexto

O pipeline tem 6 estágios batch (load → preprocess → features → baselines/train → evaluate/explain) que precisam ser reproduzíveis, com dados e modelos versionados fora do git.

## Decisão

DVC: `dvc.yaml` declara o grafo; `dvc repro` re-executa apenas o que mudou (hash de conteúdo); `dvc.lock` registra a proveniência de cada artefato.

## Racional

- A orquestração necessária é **por conteúdo** (o que mudou?), não **por tempo** (quando rodar?). DVC resolve a primeira; Airflow/Prefect resolvem a segunda — schedules, retries, paralelismo entre times, que este projeto não tem.
- DVC versiona dados/modelos de brinde — Airflow não.
- Custo operacional ~zero (sem scheduler, banco ou workers).

## Consequências

- (+) Reprodutibilidade total com um comando; params.yaml integrado.
- (−) Sem agendamento: o gatilho de retraining (item 25) precisa de um disparador externo (cron/CI) chamando `dvc repro`.
- Migração futura: os estágios são módulos Python independentes — viram tasks de Prefect/Airflow sem reescrita.
