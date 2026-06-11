# Fase 5 — Item 25: Avaliar o modelo em produção (sem labels imediatos)

> **Backlog:** item 25 · **Capítulo relacionado:** 8 (monitoramento) e 9 (continual learning)

## O problema que ninguém resolve com accuracy

Em produção, a pergunta "o modelo continua bom?" não pode ser respondida com a métrica de teste — **os labels verdadeiros não chegam junto com a predição**. No nosso domínio, a confirmação (biópsia/acompanhamento) leva semanas e pode nunca ser associada de volta ao request. Avaliar em produção é, na prática, **monitorar proxies** e saber o que cada um consegue (e não consegue) detectar.

## A pirâmide de sinais (do mais barato ao mais informativo)

| Sinal | Onde está implementado | O que detecta | Latência de detecção |
|-------|------------------------|---------------|---------------------|
| 1. Saúde do serviço (p95, RPS, erros) | Prometheus/Grafana (item 24) | Problemas operacionais | segundos |
| 2. Schema e ranges das entradas | Pydantic 422 (item 15) | Mudança de contrato upstream | imediata |
| 3. **Taxa de predições malignas** | `predictions_total{label}` (item 24) | Mudança grosseira no mundo ou no upstream | minutos |
| 4. **Distribuição de P(benign)** | histograma `predicted_probability_benign` | Drift que ainda não muda a classe — modelo "menos confiante" | horas |
| 5. **Drift por feature (K-S/PSI)** | Evidently sobre os logs JSONL (item 23) | Qual feature mudou, e quanto | um batch de análise |
| 6. Labels atrasados (ground truth) | processo externo (não implementado) | Performance real | semanas |

A lógica da pirâmide: os sinais 1–5 são **proxies** — nenhum prova que o recall caiu, mas cada anomalia neles é um motivo para investigar *antes* que o sinal 6 (o único definitivo) chegue. O item 13 (importância de features) prioriza o sinal 5: drift em `worst area` pesa mais que drift numa feature irrelevante.

## Gatilho de retraining — a política deste projeto

Retraining aqui é **dirigido por evidência, não por calendário**:

```
SE  drift_summary.n_drifted >= 3 features (entre as top-10 de importância)
OU  taxa de malignas sair da banda histórica (ex.: 2x em 24h)
OU  novos labels rotulados acumularem >= N casos
ENTÃO:
  1. dvc repro                      # re-treina com dados atualizados
  2. gate offline (item 12)         # recall maligno >= 0.95 no teste
  3. registry: nova versão SEM alias
  4. shadow ou canary (item 17)     # avaliação online controlada
  5. mover alias @production        # promoção = operação de metadados
```

Os passos 1–3 e 5 **já estão implementados**; o que falta num cenário real é a automação do disparo (um cron/scheduler avaliando o `drift_summary.json`) e o passo 4 com tráfego real. A decisão *stateless retraining* (re-treinar do zero, simples e suficiente para 569 linhas) vs *stateful* (fine-tuning incremental) está documentada como não-decisão consciente: stateless, pelo tamanho do dataset.

## O que este desenho NÃO detecta (limitações documentadas)

- **Concept drift puro** — P(y|x) mudar com P(x) constante: nenhum proxy sem label detecta isso. Mitigação: ciclo de labels atrasados (sinal 6) com auditoria periódica por amostragem.
- **Drift multivariado sutil** — correlações entre features mudando sem mudar marginais: o K-S univariado do item 23 não vê. Mitigação possível: detecção por reconstrução/domínio (não implementada; custo > benefício aqui).
- **Feedback loop** — se as predições influenciarem quais pacientes voltam, os dados futuros ficam enviesados pela própria predição. Relevante em crédito/recomendação; baixo risco neste domínio.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 8:** a taxonomia de shifts (covariate, label, concept) mapeia 1:1 para a tabela acima — covariate shift é o que os sinais 3–5 capturam; concept shift é a limitação declarada. A recomendação de monitorar *predições* como proxy de *performance* ("predictions are cheap, labels are expensive") é literalmente o histograma do item 24.
- **Cap. 9:** os quatro estágios de continual learning (manual → automated retraining → ...) — este projeto está no estágio "automated pipeline com gatilho manual informado por monitoramento", e o documento diz o que falta para o próximo estágio. Saber *onde se está* na escada é mais valioso que fingir estar no topo.

## Lição para levar

Em produção, "o modelo está bom?" vira "os proxies estão estáveis e o processo de reação está pronto?". O sistema maduro não é o que nunca degrada — é o que **detecta cedo (proxies), reage rápido (pipeline + gate + registry) e promove com segurança (shadow/canary + alias reversível)**.
