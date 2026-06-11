"""Métricas Prometheus do serviço de predição.

Expostas em /metrics (formato OpenMetrics). Três famílias:

- prediction_request_latency_seconds: histograma de latência por endpoint —
  responde "qual o p95?" e "o serviço está degradando?".
- predictions_total{label, model_version}: contador por classe predita —
  a TAXA de predições malignas é a métrica proxy mais barata de saúde do
  modelo (mudança brusca = investigar drift).
- predicted_probability_benign: histograma das probabilidades emitidas —
  deslocamento na distribuição é sinal precoce de drift, antes de qualquer
  label chegar.
"""

from prometheus_client import Counter, Histogram

REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "Latência dos requests de predição",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total de predições por classe",
    ["label", "model_version"],
)

PREDICTED_PROBABILITY = Histogram(
    "predicted_probability_benign",
    "Distribuição das probabilidades P(benign) emitidas",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
