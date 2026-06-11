# Fase 2 — Item 10: MLflow — tracking, registry e gate de promoção

> **Backlog:** item 10 · **Capítulo relacionado:** 6 (Desenvolvimento de modelos — experiment tracking e versionamento); cap. 7 (de onde o serving carrega o modelo)

## O que foi feito

- `src/config/tracking.py`: configuração única do MLflow — backend **SQLite** (`mlflow.db`), experimento `breast-cancer-classifier`, nome do modelo registrado e o alias `production`.
- **Treino** (`train_model.py`): cada execução vira uma run com hiperparâmetros, curvas completas por época (loss/accuracy/val_loss/val_accuracy com `step`), métricas do best epoch, o modelo Keras **e os artefatos de preprocessing (imputer/scaler) na mesma run** — lineage completo. O modelo é registrado no Model Registry a cada treino.
- **Avaliação** (`evaluate_model.py`): reabre a *mesma run* (via `run_id` persistido em `artifacts/mlflow_run_id.txt`, agora um out do DVC) e anexa as métricas de teste. Em seguida aplica o **gate de promoção**: o alias `production` só é movido para a nova versão se `recall(malignant) >= 0.95` (configurável em `params.yaml → evaluate.min_malignant_recall`).

```
train ──registra──► Model Registry v N
evaluate ──teste ok?──► alias "production" → v N      (senão: warning, alias fica onde está)
                                  ▲
serving (Fase 3) carrega "models:/breast-cancer-classifier@production"
```

## Por quê

### Por que SQLite e não o file-store padrão?

O Model Registry (versões + aliases) exige um backend com banco; o file-store puro só faz tracking. SQLite dá o registry completo sem nenhum servidor — e na Fase 4 o mesmo código apontará para um MLflow server com MinIO trocando **uma URI**.

### Por que anexar a avaliação à run de treino, em vez de criar outra run?

Uma run = um modelo candidato. Se treino e teste vivessem em runs separadas, comparar candidatos na UI exigiria juntar tabelas de cabeça. Com o `run_id` passando pelo DVC como artefato, cada linha da UI responde: *quais params → quais curvas → quais métricas de teste → promovido ou não*.

### Por que o gate de promoção é o ponto mais importante deste item

"Registrar modelo" é trivial; **decidir qual versão produção usa** é o problema real. A regra aqui codifica a decisão clínica (falso negativo é o erro caro) como critério executável: nenhum humano precisa lembrar de checar o recall antes de promover — e nenhum modelo ruim é promovido por descuido. Este é o embrião do CD de modelos: na Fase 5, o gatilho de retraining usará exatamente este mecanismo.

### E o W&B? (item 14)

Decisão: **não integrar**. MLflow local cobre tracking + registry sem conta externa nem rede — requisito do backlog ("projeto continua funcionando offline"). W&B brilha em colaboração de equipe e sweeps; para um portfólio reproduzível por qualquer pessoa com `git clone`, seria uma dependência operacional sem ganho. Racional completo virará ADR na Fase 6.

## Onde se encaixa no workflow

O registry desacopla **treinar** de **servir**: o serving não conhece caminhos de arquivo, só o alias. Trocar o modelo de produção passa a ser uma operação de metadados (mover o alias) — com rollback instantâneo (mover de volta). Isso habilita os padrões de deploy da Fase 3/5 (canary, shadow) sem rebuild de imagem.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 6 — experiment tracking e versionamento:** a lista do que rastrear (curva de loss, métricas de avaliação, amostras, velocidade) está implementada nas curvas por época + métricas de teste anexadas. O capítulo nota que "tracking é a parte fácil; versionamento que permite reprodução é a difícil" — aqui cada run aponta para o commit git e o `dvc.lock` cobre os dados.
- **Cap. 7:** o serving que carrega modelo por referência lógica (alias) em vez de arquivo físico é o que o capítulo chama de separar o *model artifact* do *prediction service*.

## Como validar

```bash
uv run dvc repro -f train evaluate
# log: "Model vN promoted to 'production' (malignant recall 0.976 >= 0.95)"
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# UI em http://localhost:5000: runs comparáveis, modelo registrado, alias @production
```

## Lição para levar

Experiment tracking sem registry é diário de bordo; registry sem gate de promoção é prateleira de modelos. O valor aparece quando os três viram um fluxo: **toda run é comparável, todo modelo é versionado, e a promoção é uma regra de negócio executável — não uma decisão de memória.**
