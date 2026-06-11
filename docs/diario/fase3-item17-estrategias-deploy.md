# Fase 3 — Item 17: Estratégias de deploy de modelos

> **Backlog:** item 17 · **Capítulo relacionado:** 7 (implantação) e 9 (teste em produção)

Este documento responde: **quando uma nova versão do modelo é promovida no registry, como ela chega aos usuários?** As estratégias abaixo são padrões da indústria; para cada uma, o custo, o que ela detecta e como se aplicaria *neste* projeto.

## As quatro estratégias

### 1. Recreate (substituição direta)
Derruba a versão antiga, sobe a nova. **Custo:** zero infra extra. **Risco:** todo o tráfego migra de uma vez; um modelo ruim afeta 100% dos usuários até o rollback. Aceitável apenas com gates offline fortes (que temos: item 12) e baixo tráfego.

### 2. Blue-Green
Duas infraestruturas completas (blue = atual, green = nova); o load balancer vira a chave de uma vez — e pode virar de volta em segundos. **Detecta:** falhas operacionais (latência, memória, crash). **Não detecta:** degradação de qualidade de predição, que só aparece com tráfego real e tempo. **Aqui:** dois Deployments no K8s (Fase 4) com o Service apontando para um deles; rollback = trocar o selector.

### 3. Canary (liberação gradual)
A nova versão recebe uma fração do tráfego (1% → 10% → 50% → 100%), com métricas comparadas a cada degrau. **Detecta:** problemas operacionais *e* mudanças na distribuição das predições, com raio de dano limitado. **Custo:** precisa de roteamento ponderado e métricas por versão (Prometheus, Fase 5). **Aqui:** o `model_version` que a API retorna em cada resposta é o que permite segmentar as métricas por versão — o pré-requisito já está construído.

### 4. Shadow (espelhamento)
A nova versão recebe **cópia** de todo o tráfego real, mas suas respostas são descartadas — só registradas. Zero risco ao usuário; compara-se offline as predições das duas versões. **Custo:** 2× a computação de inferência; e não testa o efeito das predições no mundo (as respostas não são usadas). **Aqui:** é a estratégia ideal para validar um candidato a substituir o MLP (ex.: a LogisticRegression do item 11) — os logs de predição da Fase 5 são exatamente o insumo da comparação.

## Dimensão ortogonal: como avaliar durante o rollout

Estratégia de *tráfego* (acima) ≠ estratégia de *avaliação*. O cap. 9 cobre **testes A/B** (atribuição aleatória + teste de hipótese sobre uma métrica de negócio) e **bandits** (alocação adaptativa). Para um classificador clínico, A/B sobre desfecho real seria ética e operacionalmente complexo — mais um motivo para shadow + métricas de proxy (distribuição de probabilidades, taxa de predições malignas, item 25).

## Recomendação para este projeto

| Cenário | Estratégia |
|---------|-----------|
| Trocar MLP por outro modelo candidato | **Shadow** (comparação offline com logs, sem risco) |
| Atualizar a mesma arquitetura re-treinada | **Canary 10% → 100%** vigiando latência e distribuição das predições |
| Hotfix de infraestrutura (não do modelo) | **Blue-green** (rollback instantâneo) |

O mecanismo comum aos três: **o alias `production` do registry + o `model_version` em cada resposta**. Promover/reverter é mover um alias; medir por versão é agrupar pelos logs. As Fases 4 e 5 constroem o restante (K8s, Prometheus, logs de predição).

## Teoria ↔ prática (Chip Huyen)

- **Cap. 7** apresenta as estratégias de release e o princípio de separar *deployment* (código no ar) de *release* (tráfego de verdade) — canary e shadow são exatamente essa separação.
- **Cap. 9** cobre shadow, A/B, canary, interleaving e bandits como *test in production*, com o argumento central: avaliação offline nunca é suficiente, porque a distribuição de produção é outra. O gate offline (item 12) é o **primeiro** filtro, não o último.

## Lição para levar

Não existe estratégia "certa" — existe o trade-off entre custo de infra, velocidade de detecção e raio de dano. O que não é negociável: predição carregar a versão do modelo, promoção ser reversível em segundos, e a decisão de "subir mais tráfego" ser baseada em métrica, não em otimismo.
