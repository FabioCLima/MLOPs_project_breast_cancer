# Model Card — Breast Cancer Classifier

> Formato inspirado em [Mitchell et al., 2019 — "Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993).

## Detalhes do modelo

| Campo | Valor |
|-------|-------|
| Modelo em produção | MLP Keras (2 camadas ocultas, saída sigmoid), registrado como `breast-cancer-classifier@production` no MLflow |
| Versão / proveniência | Alias no MLflow Model Registry; cada versão aponta para a run com params, métricas, curvas e artefatos de preprocessing |
| Entrada | 30 features numéricas (medidas de núcleos celulares de imagem FNA digitalizada) |
| Saída | P(benign) ∈ [0,1] + decisão binária com threshold 0.88 (ajustado na validação para recall maligno com precision ≥ 0.90) |
| Alternativa avaliada | LogisticRegression empata em recall maligno (0.976) com ROC-AUC de CV 0.994 ± 0.006 — ver `metrics/baselines.json` |

## Uso pretendido

- **Pretendido:** demonstração didática e de portfólio de práticas de MLOps (pipeline, tracking, serving, monitoramento) sobre um problema de classificação binária.
- **Fora de escopo — proibido:** uso em decisão clínica, triagem de pacientes ou qualquer contexto assistencial. O modelo foi treinado em um dataset didático de 569 casos de um único centro (Wisconsin, anos 1990) e **não passou por validação clínica, regulatória ou prospectiva de nenhum tipo.**

## Dados

- **Fonte:** Breast Cancer Wisconsin (Diagnostic), via `sklearn.datasets` (569 amostras; 63% benignas / 37% malignas).
- **Particularidades:** NaNs são injetados sinteticamente (~5%) para fins didáticos de imputação; população de um único centro e época — sem representatividade demográfica documentada (idade, etnia, equipamento), o que por si só inviabiliza uso clínico.
- **Splits:** 64/16/20 (treino/validação/teste), estratificados; transformadores ajustados somente no treino (testes automatizados de leakage em `tests/test_no_leakage.py`).

## Métricas (conjunto de teste, n=114, threshold 0.88)

| Métrica | Valor |
|---------|-------|
| Recall — maligna (métrica de decisão) | **0.976** |
| Precision — maligna | 0.837 |
| ROC-AUC (maligna) | 0.992 |
| PR-AUC (maligna) | 0.989 |
| Brier score (calibração) | 0.040 |

O threshold foi escolhido para minimizar falsos negativos (câncer não detectado) ao custo de mais falsos positivos (exames adicionais) — trade-off documentado em `docs/diario/fase2-item12-avaliacao-clinica.md`.

## Fatores relevantes (explicabilidade)

Permutation importance (consistente entre MLP e regressão logística): **worst area, worst concave points, worst texture** dominam as predições — alinhado à literatura do dataset. Gráficos em `metrics/explainability/`. Limitação: features altamente correlacionadas (radius/perimeter/area) dividem importância.

## Limitações conhecidas

1. Dataset pequeno (569) e antigo; métricas têm incerteza amostral relevante (teste n=114).
2. Sem validação externa (outro centro/equipamento) — generalização desconhecida.
3. K-S univariado no monitoramento não captura drift multivariado; concept drift só é detectável com labels atrasados.
4. O threshold foi ajustado em ~90 amostras de validação — sensível a re-amostragem (mitigado pelo tie-break conservador).

## Monitoramento em produção

- Logs de predição estruturados (JSONL) com features, saída, versão e latência.
- Drift por feature (Evidently, K-S) contra a referência de treino; relatório em `metrics/drift/`.
- Métricas Prometheus: latência p95, taxa de predições malignas, distribuição de P(benign).
- Gate de promoção: nenhuma versão recebe o alias de produção com recall maligno < 0.95 no teste.

## Considerações éticas e de compliance

- **Privacidade:** o dataset público é anonimizado. Num cenário real (dados de pacientes), seriam mandatórios: base legal (LGPD/HIPAA), minimização de dados nos logs de predição (hoje as 30 features são logadas em claro — num sistema real, pseudonimização e retenção limitada), criptografia em trânsito/repouso e controle de acesso por papel.
- **Auditoria:** toda predição carrega versão do modelo e threshold; o registry + git + DVC permitem reconstruir exatamente o artefato que gerou qualquer decisão.
- **Vieses:** a composição demográfica do dataset não é documentada; qualquer uso além do didático exigiria análise de performance por subgrupo (slice-based evaluation).
- **Supervisão humana:** mesmo num cenário hipotético de uso assistencial, o desenho correto seria *apoio à decisão* com revisão humana obrigatória — nunca decisão automatizada (LGPD art. 20).
