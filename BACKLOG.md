# Backlog de Refatoração e Evolução MLOps

Backlog operacional para evoluir o projeto de formato de curso para um portfólio com sinais claros de engenharia de ML em **nível sênior**, alinhado aos requisitos de vagas de Senior MLOps Engineer (deploy com best practices, workflows automatizados, infraestrutura, monitoramento, compliance, cloud, Kubernetes, Docker).

A sequência e as indicações de leitura estão em [ROADMAP.md](ROADMAP.md). Os itens estão organizados por fase; dentro de cada fase, a ordem é a recomendada.

## Objetivo

Transformar o projeto em uma demonstração reprodutível, testável e observável de um sistema de classificação para breast cancer prediction, cobrindo o ciclo completo: dados, treino, avaliação, tracking, registry, serving, monitoramento e documentação técnica.

---

## Fase 0 — Fundação e Corretude

### 1. Corrigir compatibilidade de runtime Python

**Problema:** o projeto declara `requires-python = ">=3.12"`, mas `tensorflow==2.19.0` não roda em Python 3.13 no ambiente validado.

**Tarefas:**
- Alterar `pyproject.toml` para `requires-python = ">=3.12,<3.13"`.
- Adicionar arquivo `.python-version` com `3.12`.
- Atualizar README com `uv sync --python 3.12`.
- Validar `uv run --python 3.12 ruff check .`.

**Critérios de aceite:**
- Ambiente novo instala sem erro.
- README deixa claro qual versão de Python usar.

### 2. Corrigir data leakage do validation split

**Problema:** o scaler e o imputer são ajustados em 100% do conjunto de treino (`src/feature_engineering/engineer_features.py`), mas `train_model.py` usa `validation_split=0.2` do Keras. As estatísticas dos transformadores "enxergam" os dados de validação, tornando as métricas de validação (que guiam o early stopping) otimistas.

**Tarefas:**
- Criar split train/validation explícito e estratificado ANTES do fit de imputer/scaler.
- Remover `validation_split` do `model.fit()` e passar `validation_data` explicitamente.
- Adicionar teste automatizado que comprove que os transformadores não viram o conjunto de validação.

**Critérios de aceite:**
- Nenhum transformador é ajustado em dados usados para validação ou teste.
- Teste de leakage roda na suite.

### 3. Corrigir corretude do treino e da avaliação

**Problema:** acúmulo de bugs sutis que distorcem os resultados reportados.

**Tarefas:**
- Adicionar `stratify` no `train_test_split` (`src/data_preprocessing/preprocess_data.py`).
- Salvar métricas do melhor epoch (`early_stopping.best_epoch`), não do último (`src/model_training/train_model.py` usa `history.history[metric][-1]` com `restore_best_weights=True`, então modelo salvo e métricas reportadas divergem).
- Simplificar a arquitetura para classificação binária: 1 saída sigmoid + `binary_crossentropy`, eliminando o `OneHotEncoder` do target.
- Remover `params.pop("random_seed")` (mutação do dicionário do chamador).
- Remover o parâmetro `encoder` morto em `evaluate_model()` e corrigir o type hint (`LabelEncoder` vs artefato real).
- Corrigir typo `# type: ignore[assignement]` em `load_data.py`.

**Critérios de aceite:**
- Métricas reportadas correspondem ao modelo salvo.
- Split estratificado verificado por teste.
- Pipeline binário sem one-hot do target.

### 4. Centralizar caminhos e parâmetros

**Problema:** existe `src.config.paths`, mas só `load_data.py` o usa; os demais módulos têm caminhos hardcoded.

**Tarefas:**
- Substituir strings como `data/raw/raw.csv`, `params.yaml`, `models/model.keras` por constantes de `src.config.paths`.
- Criar helper centralizado para carregar `params.yaml`.
- Renomear artefatos removendo colchetes (`[features]_scaler.joblib` → `features_scaler.joblib`); colchetes quebram glob, shell e URLs.
- Garantir que cada etapa roda mesmo quando chamada fora da raiz do projeto.

**Critérios de aceite:**
- Nenhum módulo de pipeline depende do current working directory.
- Diretórios necessários são criados antes de salvar arquivos.

### 5. Padronizar logging

**Problema:** o projeto mistura `loguru` e `logging`; `setup_logger()` existe mas nenhum estágio o chama.

**Tarefas:**
- Adotar loguru como estratégia única.
- Aplicar `setup_logger()` no entrypoint de cada etapa.
- Remover configurações duplicadas.

**Critérios de aceite:**
- Logs consistentes em todas as etapas e no app.
- Execução via `python -m` mostra logs legíveis.

---

## Fase 1 — Reprodutibilidade e Dados como Código

### 6. Orquestrar o pipeline com DVC

**Problema:** os 5 estágios são executados manualmente em sequência; `params.yaml` já está no formato que o DVC consome.

**Tarefas:**
- Criar `dvc.yaml` com os estágios: load_data, preprocess, features, train, evaluate.
- Declarar deps, params e outs de cada estágio.
- Configurar remote local (e depois MinIO/S3 na Fase 4).
- Documentar `dvc repro` e `dvc metrics diff` no README.

**Critérios de aceite:**
- `dvc repro` executa o pipeline completo do zero.
- Mudança em `params.yaml` re-executa apenas os estágios afetados.
- Dados e modelos fora do git, versionados pelo DVC.

### 7. Validação de schema de dados com Pandera

**Tarefas:**
- Definir schema das 30 features: tipos, ranges fisicamente plausíveis, percentual máximo de NaN.
- Validar na entrada de cada estágio do pipeline.
- Falhar com mensagem clara quando o contrato for violado.

**Critérios de aceite:**
- Dado corrompido interrompe o pipeline com erro acionável.
- Schema é único e reutilizado (pipeline e serving).

### 8. Criar suite mínima de testes

**Problema:** `pytest` roda, mas coleta 0 testes.

**Tarefas:**
- Criar `tests/`.
- Testar split de dados com seed fixa e estratificação.
- Testar imputer e scaler sem leakage (treino/val/teste).
- Testar shape de `X_train` e `y_train`.
- Testar criação do modelo Keras sem treinar por muitas épocas.
- Testar `ModelService` com artefatos mockados ou fixture pequena.
- Adicionar `mypy` ao pre-commit (já é dev dependency, nunca roda).

**Critérios de aceite:**
- `uv run --python 3.12 pytest` executa testes reais em menos de 60 segundos.
- Falhas comuns de schema, shape, caminho e leakage são capturadas.

### 9. Adicionar CI

**Tarefas:**
- Criar workflow GitHub Actions: Ruff, mypy, pytest.
- Usar cache de dependências do `uv`.
- Fixar Python 3.12.

**Critérios de aceite:**
- Pull requests mostram checks automáticos.
- README exibe badge de CI.

---

## Fase 2 — Experimentação Madura

### 10. Integrar MLflow (tracking + registry)

**Tarefas:**
- Adicionar `mlflow` como dependência.
- Criar experimento `breast-cancer-classifier`.
- Logar parâmetros de `params.yaml`, métricas de treino/validação/teste, classification report, confusion matrix e curvas como artefatos.
- Logar modelo e preprocessing artifacts.
- Registrar o modelo campeão no **Model Registry** com stage/alias de produção — o serving (Fase 3) carregará daqui.

**Critérios de aceite:**
- `mlflow ui` mostra múltiplas runs comparáveis.
- Modelo de produção identificável no registry, não por caminho de arquivo.
- README inclui screenshot ou instrução clara de uso.

### 11. Criar baselines clássicos antes da rede neural

**Problema:** para um dataset tabular com 569 linhas, uma rede neural não é necessariamente a melhor escolha — e um single split é ruído.

**Tarefas:**
- Treinar `LogisticRegression` e `RandomForestClassifier`.
- Comparar contra o MLP/Keras usando **cross-validation estratificada**.
- Registrar accuracy, precision, recall, F1, ROC-AUC e PR-AUC de todos no MLflow.

**Critérios de aceite:**
- Tabela comparativa de modelos no README.
- README explica por que o modelo final foi escolhido (e se a resposta for "o baseline ganhou", melhor ainda — honestidade técnica conta).

### 12. Avaliação para contexto médico

**Tarefas:**
- Reportar recall da classe maligna como métrica de destaque.
- Adicionar matriz de confusão normalizada, ROC curve e precision-recall curve.
- **Threshold tuning:** otimizar o limiar de decisão para recall da classe maligna com restrição de precision, em vez de argmax/0.5 implícito.
- **Calibração:** curva de calibração e Brier score.

**Critérios de aceite:**
- `metrics/evaluation.json` contém métricas principais e o threshold escolhido.
- README comenta o trade-off entre falso negativo e falso positivo.

### 13. Adicionar explicabilidade

**Tarefas:**
- Permutation importance para os baselines sklearn.
- Opcionalmente SHAP para o modelo final.
- Salvar gráficos como artefatos do MLflow.

**Critérios de aceite:**
- Projeto explica quais features mais influenciam as predições.
- Limitações da interpretabilidade documentadas.

### 14. Opcional: integrar W&B

**Tarefas:**
- Suporte opcional via env var; não obrigatório para rodar localmente.
- Documentar quando usar W&B versus MLflow.

**Critérios de aceite:**
- Projeto continua funcionando offline com MLflow local.

---

## Fase 3 — Serving Moderno

### 15. Migrar Flask → FastAPI com Pydantic v2

**Problema:** `app.main` carrega artefatos no import, não tem health check, valida entrada manualmente e usa `OneHotEncoder.inverse_transform()` direto em probabilidades (comportamento não documentado do sklearn).

**Tarefas:**
- Criar `create_app()` com lifespan para carregar o `ModelService` de forma controlada.
- Endpoints: `/health`, `/predict` (JSON), `/predict/batch` (CSV), `/metadata` (versão do modelo, métricas de treino).
- Contrato de entrada/saída com **Pydantic v2**: 30 features tipadas com ranges; resposta com `{classe, label, probabilidade, versao_do_modelo}`.
- Carregar modelo do **MLflow Registry** por stage/alias, não por caminho.
- Usar argmax/threshold explícito (threshold da Fase 2) na decisão.
- Preservar nomes de features após imputer/scaler para eliminar warnings.

**Critérios de aceite:**
- Importar o módulo do app não exige modelo treinado.
- Request inválida retorna HTTP 422 com mensagem clara.
- Swagger/OpenAPI disponível em `/docs`.
- Inferência não emite warning de feature names.

### 16. Testes do serving

**Tarefas:**
- Testes com `TestClient` e service mockado.
- Teste de contrato: payload válido, payload com feature faltando, tipo errado, NaN.
- Smoke test de latência.

**Critérios de aceite:**
- Suite do app roda sem artefatos reais de modelo.

### 17. Documentar estratégias de deploy

**Tarefas:**
- Seção no README: shadow deployment, canary, blue-green — qual escolheria aqui e por quê.

**Critérios de aceite:**
- Leitor entende as opções e o racional da escolha.

---

## Fase 4 — Containers e Infraestrutura

### 18. Melhorar Dockerfile

**Tarefas:**
- Multi-stage build (builder + runtime slim); instalar dependências antes de copiar o código para aproveitar cache.
- Usuário não-root e `HEALTHCHECK`.
- Não copiar `data/` para a imagem de serving.
- Falhar com erro claro se modelo/artefatos não existirem.

**Critérios de aceite:**
- Build reproduzível e enxuto.
- Container inicia com `gunicorn`/`uvicorn` e `/health` responde.

### 19. Criar Docker Compose

**Tarefas:**
- Serviços: API FastAPI, MLflow server, MinIO (artifact store S3-compatível — "cloud-ready" sem custo), Prometheus, Grafana.
- Volumes para `mlruns/` e dados do MinIO.

**Critérios de aceite:**
- `docker compose up` sobe o stack completo.
- README documenta portas e fluxo.

### 20. Kubernetes local

**Tarefas:**
- Manifests (ou Helm chart simples) para a API: Deployment com liveness/readiness probes, Service, HPA, resource limits.
- Rodar em kind ou minikube; documentar o passo a passo.

**Critérios de aceite:**
- `kubectl apply` sobe a API com probes funcionando.
- HPA demonstrável sob carga sintética.

### 21. CD no GitHub Actions

**Tarefas:**
- Build da imagem, push para GHCR, smoke test do `/health` no container.

**Critérios de aceite:**
- Tag/release gera imagem publicada e testada automaticamente.

---

## Fase 5 — Monitoramento e Pós-Deploy

### 22. Logging estruturado de predições

**Tarefas:**
- Cada request → JSON estruturado: features, predição, probabilidade, latência, versão do modelo, timestamp.

**Critérios de aceite:**
- Logs de predição consumíveis por ferramenta de análise (são a matéria-prima do drift).

### 23. Monitoramento de data drift com Evidently

**Tarefas:**
- Relatório comparando distribuição de produção vs. referência de treino (PSI/KS por feature).
- Simular drift perturbando features para a demo.
- Documentar o que dispararia investigação/retraining.

**Critérios de aceite:**
- Relatório de drift gerado a partir dos logs de predição.
- README mostra um exemplo de drift detectado.

### 24. Métricas de serviço (Prometheus + Grafana)

**Tarefas:**
- Instrumentar a API: latência p50/p95, RPS, taxa de erro, distribuição das probabilidades preditas.
- Dashboard Grafana versionado no repo.

**Critérios de aceite:**
- Dashboard funcional via docker compose.

### 25. Avaliação do modelo em produção

**Tarefas:**
- Documentar como avaliar o modelo sem labels imediatos: proxy metrics, distribuição das probabilidades, taxa de predições malignas.
- Definir gatilho conceitual de retraining.

**Critérios de aceite:**
- README/docs explicam o ciclo de vida pós-deploy.

### 26. Model card e compliance

**Tarefas:**
- Model card: uso pretendido, dados, métricas, limitações, vieses.
- Seção de compliance: privacidade de dados médicos, auditoria, por que o modelo não deve ser usado para decisão clínica.

**Critérios de aceite:**
- Projeto demonstra maturidade e responsabilidade técnica.

---

## Fase 6 — Documentação de Portfólio

### 27. Reescrever README como case técnico

**Tarefas:**
- Arquitetura do projeto com diagrama do pipeline: data → preprocessing → features → training → evaluation → registry/tracking → serving → monitoring.
- Decisões de modelagem, resultados (tabela comparativa), limitações, roadmap.
- Comandos reproduzíveis do zero.
- Seção **"decisões que NÃO tomei e por quê"**: feature store (Feast) — over-engineering para dataset batch de 569 linhas, mas explicar o trade-off de online/offline consistency; Kubeflow; W&B.

**Critérios de aceite:**
- Um recrutador entende o valor técnico em menos de 3 minutos.
- Um engenheiro reproduz o pipeline do zero.

### 28. ADRs (Architecture Decision Records)

**Tarefas:**
- ADRs de 1 página para as decisões principais: DVC vs Airflow/Prefect, FastAPI vs Flask, MLflow vs W&B, threshold tuning, não usar feature store.

**Critérios de aceite:**
- Cada decisão estrutural tem racional registrado.

---

## Definition of Done Geral

- `uv run --python 3.12 ruff check .` e `mypy` passam.
- `uv run --python 3.12 pytest` passa com testes reais (incluindo testes de leakage).
- `dvc repro` roda o pipeline completo do zero.
- MLflow UI mostra runs comparáveis e o registry aponta o modelo de produção.
- API FastAPI sobe sem efeitos colaterais no import; `/health` e `/docs` respondem.
- `docker compose up` sobe API + MLflow + monitoramento.
- Relatório de drift gerado a partir de logs de predição.
- README reproduz o projeto do zero; repositório não versiona dados/modelos/artefatos grandes.
