# Diário de Bordo — Teoria e Prática em MLOps

Esta pasta documenta, mudança a mudança, a evolução deste projeto de um formato de curso para um sistema de ML com práticas de produção. Cada documento corresponde a um item do [BACKLOG.md](../../BACKLOG.md) e a um commit, e responde sempre a quatro perguntas:

1. **O que foi feito** — a mudança concreta, com arquivos e trechos.
2. **Por quê** — o problema que existia e o risco de não corrigir.
3. **Onde se encaixa no workflow** — em que ponto do ciclo de vida do sistema de ML essa peça entra.
4. **Teoria** — o capítulo de *"Projetando Sistemas de Machine Learning"* (Chip Huyen) que fundamenta a decisão, e como a teoria se materializou aqui.

O objetivo é que um(a) cientista de dados ou ML engineer júnior consiga ler o livro e este diário lado a lado, vendo cada conceito virar código de verdade.

## Índice

| Doc | Item do backlog | Tema | Capítulo (Chip Huyen) |
|-----|----------------|------|----------------------|
| [01](fase0-item01-runtime-python.md) | 1 | Runtime Python reprodutível | 10 |
| [02](fase0-item02-03-leakage-e-corretude.md) | 2, 3 | Data leakage no validation split; estratificação; best epoch; sigmoid binária; teste de leakage | 4, 5, 6 |
| [03](fase0-item04-paths-e-params.md) | 4 | Paths e params centralizados; independência de cwd | 2, 10 |
| [04](fase0-item05-logging.md) | 5 | Logging único com loguru; observabilidade camada zero | 8, 10 |
| [05](fase1-item06-dvc.md) | 6 | Pipeline DVC: grafo explícito, repro incremental, proveniência | 3, 6, 10 |
| [06](fase1-item07-pandera.md) | 7 | Contrato de dados Pandera nas fronteiras do pipeline | 4, 8 |
| [07](fase1-item08-testes.md) | 8 | Suite de testes (19) + mypy; ModelService testável | 6, 11 |
| [08](fase1-item09-ci.md) | 9 | CI GitHub Actions: lint, tipos e testes a cada push | 9, 10 |
| [09](fase2-item10-mlflow.md) | 10, 14 | MLflow: tracking, registry e gate de promoção por recall | 6, 7 |
| [10](fase2-item11-baselines.md) | 11 | Baselines com CV estratificada: LR empata com o MLP | 6 |

*(o índice cresce conforme os itens são entregues — veja a sequência completa no [ROADMAP.md](../../ROADMAP.md))*
