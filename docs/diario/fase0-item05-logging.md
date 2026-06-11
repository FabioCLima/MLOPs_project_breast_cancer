# Fase 0 — Item 5: Logging padronizado com loguru

> **Backlog:** item 5 · **Capítulo relacionado:** 8 (Monitoramento — logs como observabilidade); cap. 10 (ferramentas)

## O que foi feito

- Todos os estágios do pipeline e o app passaram a usar **loguru** via `from loguru import logger`, eliminando o `logging` da stdlib.
- Cada entrypoint (`main()`) chama `setup_logger()` — que configura console colorido + arquivo com rotação diária e retenção (30 dias geral, 90 dias para erros).
- `setup_logger()` agora usa `LOGS_DIR` de `src.config.paths` (era um `"logs"` relativo ao cwd — mesma classe de bug corrigida no item 4).

## Por quê

O projeto tinha **duas configurações de logging coexistindo**: `load_data.py` usava loguru; os outros quatro estágios usavam `logging.getLogger(...)` sem nenhum handler configurado — ou seja, o formato dependia do que o último import tivesse feito, e o `setup_logger()` cuidadosamente escrito em `logging_config.py` **nunca era chamado por ninguém**.

Esse é um anti-padrão comum: a infraestrutura existe, mas não está ligada. Em revisão de portfólio, código morto de configuração é pior que ausência — sinaliza que o autor não executou o próprio sistema observando a saída.

A escolha por loguru (e não stdlib) é pragmática: API menor, formatação rica sem boilerplate de handlers, rotação/retenção/compressão declarativas em uma linha. Para uma aplicação (não biblioteca), não há razão para a cerimônia da stdlib.

## Onde se encaixa no workflow

Logging é a **camada zero da observabilidade**. A cadeia completa que este projeto vai construir:

```
logs estruturados (agora) → logs de predição (Fase 5, item 22)
→ métricas de serviço Prometheus (item 24) → detecção de drift Evidently (item 23)
```

Sem logs consistentes, nenhum dos passos seguintes tem matéria-prima. Quando o serving (Fase 3) registrar cada predição com latência e versão do modelo, será sobre esta fundação.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 8** abre a discussão de monitoramento com os três pilares de observabilidade — **logs, traces e métricas** — e nota que em ML os logs são frequentemente a única evidência do que o modelo viu em produção. A disciplina de logar *em todos os estágios com o mesmo formato* é o que torna os logs pesquisáveis depois.
- **Cap. 10**: ferramenta pronta e bem mantida (loguru) versus manter configuração caseira de stdlib — de novo a decisão *build vs buy* em escala pequena.

## Como validar

```bash
uv run python -m src.data_preprocessing.preprocess_data
# saída colorida uniforme: timestamp | LEVEL | módulo:linha - mensagem
ls logs/   # mlops_YYYY-MM-DD.log com os mesmos eventos
```

## Lição para levar

Configuração de logging que nenhum entrypoint chama é igual a não ter logging. A regra: **quem configura é o entrypoint; quem usa é todo mundo** — módulos importam o logger e logam; só o `main()` (ou o factory do app) decide formato, destino e nível. Misturar os dois papéis é o que gera os projetos com 3 formatos de log diferentes na mesma execução.
