# ADR 0004 — Threshold de decisão como parâmetro de negócio versionado

**Status:** aceito · **Data:** 2026-06-11

## Contexto

Classificadores binários costumam decidir por `argmax`/0.5 implícito. Num contexto clínico, falso negativo (câncer não detectado) e falso positivo (exame extra) têm custos radicalmente assimétricos.

## Decisão

O threshold é **tunado na validação** (maximiza recall maligno sujeito a precision ≥ 0.90, com tie-break para o threshold mais alto), gravado em `evaluation.json`, logado no MLflow e **carregado pelo serving** junto com o modelo. A promoção no registry exige recall maligno ≥ 0.95 no teste (gate executável).

## Racional

- O threshold escolhe *quais erros preferimos cometer* — decisão de negócio, não de otimização. Como tal: explícita, versionada e auditável.
- Tunar no teste seria leakage de decisão; tunar na validação e congelar é o protocolo correto.
- O tie-break conservador nasceu de um caso real: a versão por max-precision escolheu t=0.18, o recall de teste caiu a 0.929 e **o gate barrou a promoção** — evidência de que threshold também overfitta em validações pequenas.

## Consequências

- (+) Produção opera no ponto da curva PR que a avaliação aprovou; sem skew avaliação/serving.
- (−) Precision maligna cai a 0.837 no teste (mais falsos positivos) — trade-off documentado no MODEL_CARD.
