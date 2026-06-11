# Roadmap — Evolução para Nível Sênior em MLOps

Este roadmap organiza o [BACKLOG.md](BACKLOG.md) em fases sequenciais, cada uma fechando com algo demonstrável. Para cada fase há uma indicação de leitura do livro **"Projetando Sistemas de Machine Learning" — Chip Huyen (O'Reilly)**, com os conceitos-chave a extrair antes de codar.

## Visão geral

| Fase | Tema | Itens do backlog | Capítulos (Chip Huyen) |
|------|------|------------------|------------------------|
| 0 | Fundação e corretude | 1–5 | 1, 2 e 4 |
| 1 | Reprodutibilidade e dados como código | 6–9 | 3 e 4 |
| 2 | Experimentação madura | 10–14 | 5 e 6 |
| 3 | Serving moderno (FastAPI + Pydantic) | 15–17 | 7 |
| 4 | Containers e infraestrutura (Docker, K8s) | 18–21 | 10 |
| 5 | Monitoramento e pós-deploy | 22–26 | 8, 9 e 11 |
| 6 | Documentação de portfólio | 27–28 | 2 e 11 |

**Sequência recomendada:** Fase 0 inteira primeiro — sem fundação honesta, todo o resto herda os bugs. Depois 1 → 2 → 3 na ordem. As fases 4 e 5 podem intercalar. Não pular para Kubernetes antes de ter testes e CI.

---

## Fase 0 — Fundação e Corretude

**Objetivo:** pipeline rodando do zero com resultados honestos — sem leakage no validation split, com estratificação, métricas correspondendo ao modelo salvo e arquitetura binária adequada.

**Entregável demonstrável:** `python -m` de cada estágio roda limpo em ambiente novo; métricas confiáveis.

📖 **Leitura prévia:**
- **Cap. 1 — Visão geral dos sistemas de machine learning:** diferença entre ML em pesquisa e em produção; por que "acurácia alta" não é o objetivo do sistema.
- **Cap. 2 — Introdução ao design de sistemas de ML:** requisitos (confiabilidade, escalabilidade, manutenibilidade, adaptabilidade) — use-os como lente para julgar cada decisão deste projeto.
- **Cap. 4 — Dados de treinamento** (seções de splitting e leakage): é a base teórica para entender por que o `validation_split` do Keras combinado com scaler ajustado no treino inteiro é leakage, e por que estratificar.

---

## Fase 1 — Reprodutibilidade e Dados como Código

**Objetivo:** pipeline orquestrado por DVC, dados validados por schema (Pandera), suite de testes (incluindo testes de leakage) e CI no GitHub Actions.

**Entregável demonstrável:** `dvc repro` reproduz tudo; PR mostra checks verdes; dado corrompido derruba o pipeline com erro claro.

📖 **Leitura prévia:**
- **Cap. 3 — Fundamentos de engenharia de dados:** formatos de dados, modelos de dados, batch vs streaming — contexto para justificar as escolhas de armazenamento e o desenho dos estágios.
- **Cap. 4 — Dados de treinamento** (releitura completa): sampling, rotulagem, desbalanceamento de classes — fundamenta a estratificação e os testes de split.

---

## Fase 2 — Experimentação Madura

**Objetivo:** MLflow tracking + Model Registry; baselines clássicos (LogisticRegression, RandomForest) comparados ao MLP com cross-validation estratificada; avaliação clínica com threshold tuning e calibração; explicabilidade.

**Entregável demonstrável:** `mlflow ui` com runs comparáveis; tabela de modelos no README; modelo campeão no registry.

📖 **Leitura prévia:**
- **Cap. 5 — Engenharia de features:** operações de features, **data leakage em features** (lista de causas comuns — confronte cada uma com este projeto), importância de features.
- **Cap. 6 — Desenvolvimento de modelos e avaliação offline:** o argumento "comece com o modelo mais simples" (justifica os baselines), ensembles, rastreamento de experimentos e versionamento, métodos de avaliação (calibração, slice-based evaluation, testes de invariância).

---

## Fase 3 — Serving Moderno

**Objetivo:** migrar Flask → FastAPI com contrato Pydantic v2, lifespan para carga do modelo, modelo vindo do MLflow Registry, endpoints `/health`, `/predict`, `/predict/batch`, `/metadata`, testes com TestClient.

**Entregável demonstrável:** Swagger em `/docs`; request inválida retorna 422 com mensagem clara; app importável sem modelo treinado.

📖 **Leitura prévia:**
- **Cap. 7 — Implantação do modelo e serviço de predição:** batch prediction vs online prediction, os mitos de deployment, compressão de modelo, ML na cloud vs edge. As estratégias de deploy (shadow, canary, blue-green) que o item 17 do backlog documenta vêm daqui e do cap. 9.

---

## Fase 4 — Containers e Infraestrutura

**Objetivo:** Dockerfile multi-stage non-root com healthcheck; docker-compose com API + MLflow + MinIO + Prometheus + Grafana; deploy em Kubernetes local (kind) com probes e HPA; CD publicando imagem no GHCR.

**Entregável demonstrável:** `docker compose up` sobe o stack; `kubectl apply` sobe a API com probes; release gera imagem testada.

📖 **Leitura prévia:**
- **Cap. 10 — Infraestrutura e ferramentas para MLOps:** camadas de infraestrutura (storage/compute, desenvolvimento, gerenciamento de recursos, plataforma de ML), containers e orquestração de workloads, decisão build vs buy — material direto para os ADRs da Fase 6.

---

## Fase 5 — Monitoramento e Pós-Deploy

**Objetivo:** logging estruturado de predições; relatórios de data drift com Evidently; métricas de serviço no Prometheus/Grafana; documentação de avaliação em produção sem labels imediatos; model card e seção de compliance.

**Entregável demonstrável:** relatório de drift gerado a partir de logs de predição reais (com drift simulado); dashboard Grafana versionado.

📖 **Leitura prévia (o coração da senioridade — leia com calma):**
- **Cap. 8 — Mudanças de distribuição de dados e monitoramento:** causas de falha de sistemas de ML, tipos de shift (covariate, label, concept drift), métodos de detecção (testes estatísticos, janelas temporais) — fundamenta diretamente o item 23.
- **Cap. 9 — Aprendizado contínuo e teste em produção:** retraining (stateless vs stateful), gatilhos de atualização, teste em produção (shadow, A/B, canary, interleaving, bandits) — fundamenta os itens 17 e 25.
- **Cap. 11 — O lado humano do machine learning:** IA responsável, vieses, transparência — fundamenta o model card e a seção de compliance (item 26).

---

## Fase 6 — Documentação de Portfólio

**Objetivo:** README como case técnico (arquitetura com diagrama, resultados, limitações, decisões que NÃO foram tomadas e por quê — feature store, Kubeflow, W&B) e ADRs para as decisões estruturais.

**Entregável demonstrável:** recrutador entende o valor em < 3 minutos; engenheiro reproduz do zero.

📖 **Leitura de apoio:**
- **Cap. 2** (releitura): enquadrar o README nos requisitos de sistema — confiabilidade, escalabilidade, manutenibilidade, adaptabilidade.
- **Cap. 5** (seção de feature stores) e **cap. 10**: argumentos para o ADR "por que NÃO usar feature store aqui" — online/offline consistency é o problema que ela resolve, e este projeto (batch, 569 linhas, features computadas no pipeline) não o tem.
- **Cap. 11:** limitações e responsabilidade técnica na documentação.

---

## Mapa reverso: capítulo → onde ele aparece no projeto

| Capítulo | Aplicação concreta neste repositório |
|----------|--------------------------------------|
| 1. Visão geral dos sistemas de ML | Enquadramento do projeto: produção ≠ pesquisa |
| 2. Introdução ao design de sistemas de ML | Requisitos do sistema; estrutura do README final |
| 3. Fundamentos de engenharia de dados | DVC, formatos, desenho dos estágios do pipeline |
| 4. Dados de treinamento | Correção do leakage, estratificação, testes de split |
| 5. Engenharia de features | Imputer/scaler sem leakage, importância de features, ADR de feature store |
| 6. Desenvolvimento de modelos e avaliação offline | Baselines, cross-validation, MLflow, calibração, threshold |
| 7. Implantação do modelo e serviço de predição | FastAPI, batch vs online, estratégias de deploy |
| 8. Mudanças de distribuição e monitoramento | Evidently, PSI/KS, logging de predições |
| 9. Aprendizado contínuo e teste em produção | Gatilho de retraining, shadow/canary/A-B |
| 10. Infraestrutura e ferramentas para MLOps | Docker, compose, Kubernetes, build vs buy |
| 11. O lado humano do ML | Model card, compliance, limitações documentadas |
