# Fase 1 — Item 6: Pipeline orquestrado com DVC

> **Backlog:** item 6 · **Capítulo relacionado:** 3 (Engenharia de dados); cap. 6 (versionamento de experimentos); cap. 10 (orquestração)

## O que foi feito

- `dvc init` + `dvc.yaml` declarando os 5 estágios (`load_data → preprocess → features → train → evaluate`) com suas dependências (`deps`), parâmetros (`params`), saídas (`outs`) e métricas (`metrics`).
- `metrics/*.json` saíram do `.gitignore`: são pequenos, e mantê-los no git permite `dvc metrics diff` entre commits — comparar a accuracy de hoje com a de qualquer commit passado.
- Dados, artefatos e modelo continuam fora do git — agora versionados pelo cache do DVC.

## Por quê

Antes, "rodar o pipeline" era executar 5 comandos na ordem certa, de cabeça. Três problemas:

1. **Ordem implícita.** A dependência entre estágios existia só no README. Um colega que rodasse `train` sem `features` teria um erro de arquivo — ou pior, treinaria com dados velhos.
2. **Recomputação cega.** Mudou o `learning_rate`? Sem orquestrador, ou você re-roda tudo (lento) ou re-roda só o treino e *torce* para que os dados intermediários estejam atualizados (perigoso).
3. **Sem proveniência.** Qual versão dos dados gerou o `model.keras` atual? Impossível responder.

O `dvc.yaml` resolve os três: o grafo de dependências é **explícito e executável**. `dvc repro` calcula o hash de cada dep e re-executa apenas o que mudou — mudar `train.epochs` no `params.yaml` pula os 3 estágios de dados e re-roda só `train` e `evaluate`. E o `dvc.lock` registra o hash exato de cada entrada/saída de cada execução: proveniência completa, versionada no git.

### Por que DVC e não Airflow/Prefect?

DVC orquestra por **conteúdo** (hash de arquivos), não por **tempo** (schedule). Para um pipeline batch local de experimentação, isso é exatamente o que se quer — e ele já versiona dados de brinde. Airflow/Prefect entram quando há agendamento, retries, paralelismo entre times. Decisão registrada para virar ADR na Fase 6.

## Onde se encaixa no workflow

O `dvc.yaml` é a espinha dorsal de tudo que vem adiante:

- **CI (item 9):** o pipeline inteiro vira um comando auditável.
- **MLflow (Fase 2):** cada run terá o commit do git + `dvc.lock` = experimento 100% reproduzível.
- **Retraining (Fase 5):** o "gatilho de retraining" se materializa como `dvc repro` disparado por uma condição.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 6** insiste em **versionamento de experimentos**: código + dados + hiperparâmetros. Git versiona código; `params.yaml` os hiperparâmetros; faltava a perna dos dados — o `dvc.lock` fecha o triângulo.
- **Cap. 10** descreve a camada de **gerenciamento de recursos e orquestração** (cron → schedulers → orchestrators) e a diferença entre orquestrar *workflows de dados* e *serviços*. DVC cobre o primeiro caso no escopo local.
- **Cap. 3**: a discussão sobre fluxo de dados entre processos ("data passing through databases/files") é literalmente o que os `outs` de um estágio e `deps` do próximo implementam.

## Como validar

```bash
uv run dvc repro          # roda tudo na primeira vez
uv run dvc repro          # "Data and pipelines are up to date" — nada re-executa
# mude train.epochs no params.yaml:
uv run dvc repro          # pula load/preprocess/features, re-roda train+evaluate
uv run dvc metrics show   # métricas de treino e avaliação na mesma tabela
uv run dvc dag            # visualiza o grafo de dependências
```

## Lição para levar

Pipeline que só existe no README não é pipeline — é receita de bolo. O teste decisivo: **um comando único reproduz tudo, e rodá-lo duas vezes seguidas não recomputa nada.** Se qualquer uma das duas propriedades falhar, a reprodutibilidade é manual e vai quebrar no primeiro mês.
