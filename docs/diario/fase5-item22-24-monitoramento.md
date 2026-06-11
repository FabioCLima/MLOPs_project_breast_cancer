# Fase 5 — Itens 22–24: Logs de predição, drift (Evidently) e métricas (Prometheus/Grafana)

> **Backlog:** itens 22, 23, 24 · **Capítulo relacionado:** 8 (Mudanças de distribuição e monitoramento)

## O que foi feito

### Item 22 — Logging estruturado de predições (`app/prediction_logger.py`)

Cada predição vira uma linha JSONL: timestamp, endpoint, **versão do modelo**, threshold, latência, **as 30 features de entrada** e a saída (classe, label, probabilidade). Destino configurável por `PREDICTION_LOG_DIR` (em container: volume montado — validado no compose com 20 predições persistidas).

Este arquivo é a **matéria-prima de todo o resto**: sem registrar o que o modelo viu e respondeu, não existe análise de drift, nem comparação shadow, nem auditoria. É a peça mais barata e mais negligenciada do MLOps.

### Item 23 — Data drift com Evidently (`src/monitoring/drift_report.py`)

- **Referência:** features do treino na escala original. **Produção:** features dos logs JSONL.
- `DataDriftPreset` aplica K-S por feature (p-value < 0.05 ⇒ drift); saída: HTML interativo + `drift_summary.json` com a lista de features drifted.
- Modo `--simulate` para demo: desloca as 3 features mais importantes (item 13) em +1.5σ. **Resultado da validação: detecção cirúrgica — exatamente `worst texture`, `worst area`, `worst concave points` flagradas; zero falsos positivos nas outras 27.** A simulação importa: ninguém deveria esperar o primeiro incidente para descobrir se o detector funciona.

### Item 24 — Prometheus + Grafana (instrumentação + stack no compose)

Três famílias de métricas em `/metrics` (`app/metrics.py`), cada uma respondendo uma pergunta:

| Métrica | Pergunta |
|---------|----------|
| `prediction_request_latency_seconds` (histograma por endpoint) | O serviço está degradando? (p95) |
| `predictions_total{label, model_version}` | A taxa de malignas mudou? (proxy nº 1 de drift, segmentável por versão → canary) |
| `predicted_probability_benign` (histograma) | A distribuição de confiança deslocou? (sinal precoce, antes da classe virar) |

Compose ganhou Prometheus (scrape 10s) e Grafana com **datasource e dashboard provisionados em código** (`deploy/monitoring/`) — dashboard clicável a zero configuração manual, versionado no git como tudo mais. Validação end-to-end: tráfego real → `/metrics` populado → target Prometheus `up` → Grafana 200.

## O bug repetido que virou padrão

O logger de predições quebrou no container pelo mesmo motivo do item 18: non-root sem permissão em `/app/logs`. A correção consolidou a distinção conceitual: **logs de diagnóstico → stdout (12-factor, a plataforma coleta); logs de predição → são DADOS, vão para volume persistente** (`mkdir + chown` no Dockerfile, volume no compose). Mesma palavra "log", ciclos de vida completamente diferentes.

## Por que três camadas e não uma?

Porque cada uma tem latência e custo diferentes: Prometheus dá o alarme em segundos mas não diz *qual feature* mudou; Evidently diz qual feature mas roda em batch; o JSONL guarda tudo para qualquer análise futura (incluindo comparação shadow entre versões). Alarme barato → diagnóstico rico → arquivo completo. Remover qualquer camada quebra a cadeia.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 8** está implementado quase seção por seção: logs como base da observabilidade de ML; monitorar **predições como proxy** da performance (histograma de probabilidades); testes estatísticos (K-S) para detectar shift; e a advertência sobre janelas/sazonalidade — nosso relatório compara batches, e a escolha do tamanho da janela fica documentada como decisão do operador.
- A distinção do capítulo entre *monitoring* (ver) e *observability* (poder perguntar) é a diferença entre o dashboard (item 24) e o JSONL (item 22).

## Como validar

```bash
uv run python -m src.monitoring.drift_report --simulate   # 3/30 drifted, as corretas
docker compose up -d --build
# tráfego → http://localhost:3000 (admin/admin) → dashboard "Model Serving"
# Prometheus: http://localhost:9090/targets (api: UP)
uv run pytest tests/test_observability.py -q              # logs JSONL + /metrics
```

## Lição para levar

Monitoramento de ML não é um produto que se instala — é uma cadeia de dados que se projeta: **registre o que o modelo viu (JSONL), resuma o que importa (Prometheus), explique o que mudou (Evidently)**. E teste o detector com drift sintético antes de confiar nele: um alarme que nunca disparou na sua frente é um alarme em que você ainda não tem por que confiar.
