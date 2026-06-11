# Fase 2 — Item 13: Explicabilidade com permutation importance

> **Backlog:** item 13 · **Capítulo relacionado:** 5 (importância de features); cap. 11 (transparência e IA responsável)

## O que foi feito

- `src/model_evaluation/explain_model.py` + estágio `explain` no DVC:
  - **LogisticRegression:** `sklearn.inspection.permutation_importance` (10 repetições, scoring ROC-AUC) no teste.
  - **Keras MLP:** permutation importance **manual** — permuta cada coluna 10×, mede a queda de ROC-AUC. O loop manual existe porque o sklearn exige um estimator com API sklearn; escrever as ~15 linhas mostra que a técnica não tem mágica: *embaralhe a coluna, meça o estrago*.
- Saídas: gráficos top-15 com barras de erro + `importance.json`, anexados à run do MLflow em `explainability/`.
- Resultado: top-3 do MLP = **worst area, worst concave points, worst texture** — as mesmas features que dominam a literatura deste dataset, e coerente com o baseline LR. Dois modelos diferentes apontando para as mesmas features é evidência de que o sinal está nos dados, não no acaso do treino.

## Por quê permutation importance (e por que não SHAP)

Permutation importance é **modelo-agnóstica, global e honesta com a métrica**: mede exatamente "quanto ROC-AUC eu perco se esta feature virar ruído", na métrica que o projeto já usa. Limitações documentadas:

1. **Features correlacionadas dividem importância** — e este dataset tem correlações fortes (radius/perimeter/area são quase a mesma medida). Uma feature "pouco importante" pode ser apenas redundante com outra. Por isso reportamos a importância com desvio-padrão e não tiramos conclusões causais.
2. **É global, não local:** explica o modelo em média, não uma predição individual.

**SHAP ficou de fora de propósito** (o backlog o marcava como opcional): para 30 features tabulares, permutation + coeficientes da LR cobrem a necessidade de transparência; SHAP adicionaria uma dependência pesada e tempo de computação para ganho marginal aqui. Em um caso com features de alta cardinalidade, interações complexas ou necessidade de explicação por paciente (predição local), a resposta seria outra — o critério é a necessidade, não a moda.

## Onde se encaixa no workflow

- **Confiança clínica:** um modelo que aponta para as mesmas features que a prática médica considera relevantes é defensável diante de um domain expert; um que apontasse para "smoothness error" como dominante mereceria investigação (possível leakage ou artefato).
- **Monitoramento (Fase 5):** as features mais importantes são as que mais merecem alarme de drift — drift em "worst area" importa mais que em uma feature de importância ~0. O ranking daqui prioriza os alertas de lá.
- **Model card (item 26):** a seção "fatores relevantes" do model card consome diretamente este JSON.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 5** fecha com *feature importance* como prática de engenharia (não só de interpretação): saber quais features carregam o modelo orienta o que monitorar, o que simplificar e onde procurar leakage. A observação do capítulo de que "umas poucas features dominam" aparece aqui com clareza — o top-5 concentra a maior parte da queda de AUC.
- **Cap. 11:** transparência como requisito de IA responsável — em contexto médico, "o modelo funciona" sem "por quê" não passa de auditoria.

## Como validar

```bash
uv run dvc repro explain
ls metrics/explainability/   # lr_importance.png, keras_importance.png, importance.json
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db  # artifacts da run
```

## Lição para levar

Explicabilidade não é um slide no final — é um teste de sanidade do pipeline inteiro: se as features importantes não fizerem sentido para quem entende do domínio, o problema raramente é do domínio. E a escolha da técnica segue a mesma regra dos modelos: **a mais simples que responde à pergunta que você realmente tem.**
