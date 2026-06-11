# Fase 2 — Item 11: Baselines clássicos com cross-validation estratificada

> **Backlog:** item 11 · **Capítulo relacionado:** 6 (Desenvolvimento de modelos — "comece com o modelo mais simples")

## O que foi feito

- `src/model_training/train_baselines.py` + estágio `baselines` no DVC: `LogisticRegression` e `RandomForestClassifier` avaliados com **StratifiedKFold (5 folds)** sobre train+val, depois ajustados no dev set inteiro e medidos no teste.
- Métricas (CV média±desvio e teste): accuracy, F1 macro, ROC-AUC, PR-AUC e — em destaque — **recall e precision da classe maligna** (`pos_label=0`).
- Cada baseline é uma run no MLflow, comparável lado a lado com o MLP; tabela completa em `metrics/baselines.json`.

## O resultado — e por que ele é a lição mais valiosa do projeto

| Modelo | ROC-AUC (CV, 5 folds) | Recall maligno (teste) |
|--------|----------------------|------------------------|
| **Logistic Regression** | **0.994 ± 0.006** | **0.976** |
| Random Forest | 0.988 ± 0.007 | 0.929 |
| MLP Keras (item 10) | — (single split) | 0.976 |

**Uma regressão logística empata com a rede neural** neste dataset — com ~100× menos parâmetros, treino em milissegundos, probabilidades naturalmente bem calibradas e coeficientes interpretáveis (relevante em contexto clínico). Esse resultado não é um fracasso do MLP; é a estatística esperada para 569 linhas × 30 features tabulares bem comportadas.

Num projeto de portfólio, *admitir isso por escrito* vale mais que qualquer arquitetura sofisticada: demonstra que a escolha de modelo é guiada por evidência, não por hype. O README (Fase 6) trará essa tabela e a discussão.

## Decisões metodológicas (os detalhes que entrevistador sênior pergunta)

1. **Por que CV e não o split train/val?** Com ~450 linhas de dev, a variância de um único split domina diferenças entre modelos. A CV de 5 folds estima média ± desvio — e o desvio (±0.006) mostra que a diferença LR vs RF (0.006) está *no limite do ruído*. Reportar incerteza é o que separa comparação de chute.
2. **Por que estratificada?** Mesma razão do item 3: classes 63/37 em folds pequenos derrapam sem estratificação.
3. **Uma sutileza assumida:** os dados chegam escalados por um scaler ajustado só no treino original (64%). Dentro da CV, cada fold "vê" estatísticas de escala fixas — o rigor máximo colocaria o scaler dentro de um `Pipeline` sklearn re-ajustado por fold. Para scaling com 450 amostras o efeito é desprezível (e para RF, nulo), mas a limitação fica registrada — saber *onde* se está sendo aproximado é parte do rigor.
4. **O teste só entra uma vez**, no fim, para a tabela comparativa — nunca para escolher hiperparâmetros.

## Onde se encaixa no workflow

Os baselines mudam o status do MLP: de "o modelo do projeto" para "um candidato entre três, com evidência". O gate de promoção (item 10) e o threshold tuning (item 12) se aplicam a qualquer candidato — a infraestrutura é agnóstica ao modelo, como deve ser.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 6** é explícito: *"comece com o modelo mais simples possível"* e *"avalie trade-offs, não só performance"* — latência, interpretabilidade e custo de manutenção contam. A seção de *model selection* também alerta contra comparar modelos em um único split: é literalmente o que a CV daqui corrige.
- O capítulo ainda discute **baselines** como pré-requisito de avaliação: uma métrica só significa algo *relativo a* uma referência (aleatória, heurística, simples). Sem a LR, o "0.95 de accuracy" do MLP parecia ótimo; com ela, vira "igual ao baseline".

## Como validar

```bash
uv run dvc repro baselines
cat metrics/baselines.json
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # 3 runs comparáveis
```

## Lição para levar

Sem baseline, toda métrica impressiona; com baseline, métrica vira informação. A pergunta "seu modelo é melhor do que uma regressão logística?" é a primeira de qualquer revisão séria de ML — chegue com a resposta medida, com desvio padrão, antes que alguém pergunte.
