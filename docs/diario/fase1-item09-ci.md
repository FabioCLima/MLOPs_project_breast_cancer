# Fase 1 — Item 9: CI com GitHub Actions

> **Backlog:** item 9 · **Capítulo relacionado:** 10 (Infraestrutura); cap. 9 (continuous delivery para ML)

## O que foi feito

- `.github/workflows/ci.yml`: a cada push/PR roda, em ordem de custo crescente, **ruff check → ruff format --check → mypy → pytest**.
- `uv sync --locked`: o CI instala exatamente o `uv.lock` — se alguém esquecer de commitar o lockfile atualizado, o CI falha em vez de resolver versões diferentes silenciosamente.
- Python 3.12 vem do `.python-version` (item 1) — uma única fonte de verdade para dev, CI e Docker.
- Cache de dependências do uv habilitado (`enable-cache`): a primeira run baixa TensorFlow (~600 MB); as seguintes restauram do cache.
- Badge de status no README.

## Por quê

Sem CI, "os testes passam" significa "passavam na máquina de quem commitou, quando rodou". O CI transforma as garantias construídas nos itens anteriores (leakage, contratos, tipos) em **propriedades verificadas a cada mudança, num ambiente neutro**.

A ordem dos passos importa: lint (segundos) antes de mypy (dezenas de segundos) antes de pytest (com import de TF). Falhas baratas aparecem rápido — o feedback loop curto é o que faz dev usar o CI em vez de ignorá-lo.

### O detalhe `--locked`

`uv sync` normal re-resolveria dependências se o `pyproject.toml` e o lock divergissem — e o CI testaria **outra coisa** que não o que está versionado. `--locked` falha nessa situação. Reprodutibilidade de ambiente não é só ter lockfile; é *recusar-se a rodar sem ele*.

## Onde se encaixa no workflow

Este é o primeiro "A" de **automated workflows for model training, testing, and deployment** (texto literal da vaga-alvo). A progressão planejada:

```
CI de código (agora) → CD de imagem Docker (Fase 4, item 21) → gatilho de retraining (Fase 5, item 25)
```

O mesmo workflow ganhará, na Fase 4, o build da imagem e o smoke test do `/health` — deploy automatizado começa com teste automatizado.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 9** discute *continual learning* e nota que a infraestrutura de atualização frequente de modelos pressupõe pipelines de teste automatizados — sem CI, cada retrain é um risco manual.
- **Cap. 10**: o CI é parte da camada de desenvolvimento da infra de ML; a recomendação de padronizar ambientes (item 1) é o que torna o resultado do CI representativo do que rodará em produção.

## Como validar

```bash
# o mesmo gate, localmente:
uv run ruff check . && uv run ruff format --check . && uv run mypy src/ app/ && uv run pytest tests/ -q
# no GitHub: aba Actions mostra o workflow verde; o README exibe o badge
```

## Lição para levar

CI de ML começa igual a CI de software comum — lint, tipos, testes — e isso já elimina a maior parte dos incidentes. As partes específicas de ML (retraining, avaliação de modelo em PR, data tests pesados) se penduram nesse esqueleto depois. **Não existe MLOps maduro sobre CI inexistente.**
