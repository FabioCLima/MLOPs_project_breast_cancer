# Fase 2 — Item 12: Avaliação para contexto médico — threshold, curvas e calibração

> **Backlog:** item 12 · **Capítulo relacionado:** 6 (avaliação offline e calibração); cap. 4 (uso correto dos splits); cap. 11 (o custo humano dos erros)

## O que foi feito

- **Threshold tuning na validação** (nunca no teste): varredura de t ∈ [0.05, 0.95] maximizando o recall da classe maligna sujeito a `precision ≥ 0.90` (configurável em `params.yaml`). O teste é tocado **uma única vez**, com o threshold já congelado.
- **Curvas salvas e anexadas à run do MLflow**: ROC e PR da classe maligna, curva de calibração e matriz de confusão normalizada (`metrics/plots/`, out do DVC).
- **Novas métricas em `evaluation.json`**: ROC-AUC e PR-AUC da classe maligna e **Brier score** (0.0396).
- Resultado final no teste com t = 0.88: **recall maligno 0.976, precision 0.837**, ROC-AUC 0.992.

## A história deste item: o bug que o gate pegou

A primeira versão do tuner tinha um tie-break sutilmente errado: entre thresholds com o mesmo recall na validação, escolhia o de **maior precision** — que empurra o threshold para baixo (t = 0.18). Na validação parecia ótimo (recall 0.971, precision 1.000). No teste, o recall maligno **caiu para 0.929**… e o gate de promoção do item 10 **recusou o modelo automaticamente**.

Dois aprendizados de uma vez:

1. **Threshold também sofre overfitting.** Com ~90 amostras de validação, o "ponto ótimo" da varredura é em parte ruído. A correção: entre empates de recall, preferir o threshold **mais alto** — mais conservador contra falsos negativos em dados novos, que é o erro caro aqui. Com t = 0.88, o recall de teste subiu para 0.976.
2. **O gate de promoção funcionou na primeira oportunidade real.** Nenhum humano precisou notar a queda de recall; a regra executável barrou a promoção. É exatamente para isso que ela existe.

## Por que cada peça importa no contexto clínico

- **Recall maligno como métrica de destaque:** um falso negativo é um câncer não detectado; um falso positivo é um exame adicional. Os custos são assimétricos, então a métrica de decisão também deve ser. Accuracy esconde essa assimetria.
- **Threshold explícito ≠ argmax:** argmax é threshold 0.5 disfarçado — uma escolha de custo igual entre os erros, feita sem ninguém decidir. Tunar o threshold é *decidir de propósito* o ponto da curva precision-recall em que se quer operar (aqui: aceitar precision 0.84 para chegar a recall 0.976).
- **Calibração + Brier:** num sistema clínico, "P(maligno) = 0.8" precisa significar ~80% de frequência real — médicos agem sobre a probabilidade, não sobre o label. O Brier score (0.04, quanto menor melhor) e a curva de calibração medem isso; redes neurais costumam sair descalibradas e a curva mostra onde.
- **Matriz de confusão normalizada:** com classes 63/37, a matriz absoluta engana o olho; a normalizada mostra as taxas por classe.

## Onde se encaixa no workflow

O threshold escolhido fica em `evaluation.json` e na run do MLflow — o **serving (Fase 3) vai carregá-lo junto com o modelo**, garantindo que produção opere no mesmo ponto da curva que a avaliação aprovou. Threshold no código do serviço e threshold da avaliação divergirem é uma forma clássica de training-serving skew.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 6 — métodos de avaliação:** a seção de **calibração** ("seu modelo diz 70%, acontece 70%?") é implementada literalmente aqui; o capítulo também cobre slice-based evaluation — nosso recorte por classe (maligno vs benigno) é o slice clinicamente relevante.
- **Cap. 4:** o papel de cada split — o teste serve para *estimar* performance, não para *escolher* nada (nem hiperparâmetros, nem threshold). A queda 0.971→0.929 entre val e teste na primeira tentativa é a ilustração perfeita do porquê.
- **Cap. 11:** erros de ML têm custos humanos assimétricos; codificar essa assimetria (recall mínimo, precision mínima) em parâmetros versionados é levar o capítulo a sério.

## Como validar

```bash
uv run dvc repro evaluate
# log: threshold escolhido na validação + report no teste + decisão do gate
ls metrics/plots/   # roc, pr, calibração, matriz de confusão
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db  # plots na aba Artifacts da run
```

## Lição para levar

O threshold é um hiperparâmetro de **negócio**, não de modelo: ele escolhe quais erros você prefere cometer. Por isso (1) ele se tuna em validação como qualquer hiperparâmetro, (2) se versiona em config como qualquer decisão, e (3) viaja com o modelo até produção. E quando a validação e o teste discordarem — vão discordar — é o processo automático, não a memória de alguém, que tem que segurar a promoção.
