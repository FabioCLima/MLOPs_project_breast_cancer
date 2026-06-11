# ADR 0003 — MLflow (local-first) para tracking e registry; W&B não integrado

**Status:** aceito · **Data:** 2026-06-11

## Contexto

O projeto precisa de experiment tracking comparável e de um model registry de onde o serving carrega o modelo de produção. O backlog listava W&B como opcional.

## Decisão

MLflow com backend SQLite local (registry exige store com banco) e artefatos em `mlruns/`; em compose, MLflow server + MinIO (artifact store S3-compatible) via `MLFLOW_TRACKING_URI`. W&B fica de fora.

## Racional

- Requisito do projeto: funcionar offline, reproduzível por qualquer pessoa com `git clone` — sem conta externa.
- O Model Registry com aliases é o mecanismo central do deploy (gate de promoção, rollback por metadados); W&B Model Registry existiria, mas duplicaria a função.
- MinIO prova "cloud-readiness": migrar para AWS = trocar URL/credenciais.
- W&B brilha em colaboração de equipe e sweeps de hiperparâmetros — necessidades que este projeto não tem.

## Consequências

- (+) Stack 100% local e demonstrável; um único sistema de verdade para runs e versões.
- (−) Sem os relatórios colaborativos do W&B; aceitável para portfólio individual.
