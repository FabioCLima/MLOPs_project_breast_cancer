# Fase 0 — Item 1: Runtime Python reprodutível

> **Backlog:** item 1 · **Capítulo relacionado:** 10 (Infraestrutura e ferramentas para MLOps)

## O que foi feito

- `pyproject.toml`: `requires-python` mudou de `">=3.12"` para `">=3.12,<3.13"`.
- Criado `.python-version` com `3.12` — o `uv` (e o pyenv) leem este arquivo e selecionam o interpretador correto automaticamente.
- README atualizado: instalação via `uv sync --python 3.12` em vez de `pip install -e .`.

## Por quê

O projeto declarava suportar "qualquer Python a partir do 3.12", mas `tensorflow==2.19.0` **não tem wheels para Python 3.13**. Na prática isso significa que um colega (ou um runner de CI) com Python 3.13 instalado teria uma falha de instalação confusa — ou pior, um resolver que silenciosamente troca versões de dependências.

A regra geral: **declare o que você realmente testou, não o que você espera que funcione.** Um limite superior em `requires-python` é controverso para bibliotecas, mas para *aplicações* (como este projeto) é a escolha certa: a aplicação controla seu próprio runtime.

## Onde se encaixa no workflow

Este é o alicerce de tudo: **reprodutibilidade do ambiente**. Antes de discutir leakage, tracking ou deploy, o pipeline precisa instalar de forma idêntica em três lugares:

```
máquina do dev  →  runner de CI  →  imagem Docker de produção
```

Os três passam a apontar para o mesmo interpretador (3.12) e o mesmo lockfile (`uv.lock`). Quando chegarmos ao Dockerfile (Fase 4) e ao CI (Fase 1), eles herdam esta decisão sem precisar repensá-la.

## Teoria ↔ prática (Chip Huyen, cap. 10)

O capítulo 10 descreve a **camada de desenvolvimento** (dev environment) da infraestrutura de ML e insiste num ponto: o ambiente de desenvolvimento deve ser padronizado — mesmas versões, mesmas ferramentas — porque a maior parte dos "funciona na minha máquina" de ML vem de divergência de ambiente, não de código.

Conexões diretas:

- *"Standardize dev environments"* → `.python-version` + `uv.lock` versionados no repo.
- A discussão de **build vs buy** do capítulo também se aplica em miniatura: usamos `uv` (ferramenta pronta, rápida, com lockfile) em vez de scripts caseiros de setup.

## Como validar

```bash
uv sync --python 3.12
uv run --python 3.12 python -c "import tensorflow as tf; print(tf.__version__)"  # 2.19.0
uv run --python 3.12 ruff check .                                               # All checks passed!
```

## Lição para levar

Reprodutibilidade não começa no modelo — começa no interpretador. Pin do runtime + lockfile é a versão de "ambiente" do `random_seed`: sem isso, qualquer comparação de experimentos fica contaminada por uma variável que ninguém está olhando.
