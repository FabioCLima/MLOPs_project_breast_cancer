# Fase 4 — Itens 18–21: Docker multi-stage, Compose, Kubernetes e CD

> **Backlog:** itens 18–21 · **Capítulo relacionado:** 10 (Infraestrutura e ferramentas para MLOps)

## O que foi feito

### Item 18 — Dockerfile multi-stage (validado: build + smoke test)

```
builder (uv + lockfile) ──► runtime (slim, non-root, só o necessário)
```

- **Camada de dependências separada do código:** `COPY pyproject.toml uv.lock` + `uv sync --locked` *antes* do `COPY src/`. Mudar código não re-instala o TensorFlow — o rebuild cai de minutos para segundos. (O Dockerfile antigo fazia `COPY .` primeiro: qualquer mudança invalidava tudo.)
- **Non-root (`appuser`)**, `HEALTHCHECK` nativo, e a imagem de serving **não contém dados** — só imputer, scaler, modelo e o `evaluation.json` (threshold).
- **Bug real encontrado pelo smoke test:** com usuário non-root, o `setup_logger()` quebrou ao criar `/app/logs` (`PermissionError`). A correção não foi `chmod` — foi o padrão **12-factor**: em container, logs vão para stdout (`LOG_TO_FILE=false` na imagem) e a *plataforma* coleta. Filesystem de container deve ser tratado como read-only.
- Smoke test validado: `/health` ok, `whoami` = appuser, threshold 0.88, e o **fallback de modelo funcionou** — sem registry acessível, a API carregou `models/model.keras` com warning e reportou `model_version: "local-file"`. Degradação explícita, não crash.

### Item 19 — Docker Compose (validado: stack completo no ar)

`docker compose up` sobe **API + MLflow server + MinIO** (com job de criação de bucket). MinIO é o detalhe estratégico: é um object store com API S3 — o MLflow grava artefatos via protocolo S3, idêntico ao que faria na AWS. **Migrar para a cloud = trocar URL e credenciais**; nenhuma linha de código muda. O `tracking.py` ganhou suporte a `MLFLOW_TRACKING_URI` por env var pelo mesmo motivo: a configuração muda por ambiente, o código não.

Prometheus e Grafana entram na Fase 5 junto com a instrumentação `/metrics` — adicionar um scraper sem nada para raspar seria teatro de infraestrutura.

### Item 20 — Kubernetes (manifests validados com kubeconform; passo a passo com kind documentado)

`deploy/k8s/`: Deployment (2 réplicas, non-root, resources), Service, HPA (2→6 pods a 70% de CPU). O conceito central para ML está nos **dois probes**:

- **Liveness** — "o processo travou?" → reinicia o pod.
- **Readiness** — "está pronto para tráfego?" → segura o roteamento até o modelo carregar. Modelos demoram para carregar; sem readiness, o K8s enviaria requisições para um pod que ainda está deserializando pesos — erros 500 em todo deploy/scale-up.

### Item 21 — CD no GitHub Actions

`cd.yml`: a cada push na master, o workflow **treina o modelo do zero via `dvc repro`** (o pipeline completo é o teste de integração), builda a imagem, roda smoke test (`/health` com `model_loaded:true` + `/metadata` com threshold) e publica no GHCR; tags `v*` geram releases versionadas. O princípio: **imagem que não respondeu `/health` num container real não vai para o registro.**

## Problemas reais do caminho (e o que ensinam)

1. **Docker daemon com pidfile obsoleto** (WSL2): um dockerd zumbi de outra sessão segurava `/var/run/docker.pid`. Diagnóstico via `journalctl`, correção cirúrgica (kill + rm pidfile + restart). Infra local também é infra.
2. **Conflito de portas** (9000–9009 ocupadas na máquina): MinIO movido para 9100/9101 externas — portas internas da rede do compose não mudam, e o comentário no YAML explica o porquê.
3. **PermissionError do non-root** — o item mais valioso: rodar como root teria "funcionado" e escondido o problema até a produção.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 10:** containers como unidade de reprodução de ambiente; a discussão dev-prod parity é exatamente o pipeline CI→imagem única→compose/K8s. A camada de *resource management* (réplicas, limits, autoscaling) é o que os manifests implementam; e a decisão **build vs buy** aparece em usar MinIO/MLflow prontos em vez de infraestrutura própria.
- **Cap. 7:** o fallback registry→arquivo local é a separação modelo/serviço aplicada — o serviço sobrevive à indisponibilidade do registry degradando com aviso, e a resposta da API sempre diz qual modelo está servindo.

## Como validar

```bash
docker build -t ml-classifier:dev .          # multi-stage, ~3 GB (TF)
docker run -d --rm -p 5001:5001 ml-classifier:dev && curl localhost:5001/health
docker compose up -d                          # API + MLflow (5000) + MinIO (9101)
docker run --rm -v "$PWD/deploy/k8s:/m:ro" ghcr.io/yannh/kubeconform:latest /m
# kind: passo a passo em deploy/k8s/README.md
```

## Lição para levar

A imagem de serving é um **artefato de release**, não um ambiente de desenvolvimento: não contém dados, não roda como root, não escreve no próprio filesystem, e declara como verificar sua saúde. Cada uma dessas propriedades parece pedantismo até o dia em que evita um incidente — e o smoke test desta fase provou duas delas (non-root pegou o bug de logs; fallback segurou a ausência do registry).
