# Prompt — Automatizar o gatilho de retraining dirigido por drift

> Cole o texto abaixo numa sessão do Claude Code (modelo Sonnet/Haiku) na raiz do projeto.

---

Implemente a automação do gatilho de retraining descrito em `docs/diario/fase5-item25-avaliacao-em-producao.md` (seção "Gatilho de retraining"). Leia esse doc ANTES de escrever código — a política já está decidida; seu trabalho é só materializá-la.

**Contexto:** o projeto já tem todas as peças: `src/monitoring/drift_report.py` gera `metrics/drift/drift_summary.json` (campos `n_drifted` e `drifted_features`); `uv run dvc repro` re-treina; a avaliação aplica o gate de promoção no MLflow automaticamente. Falta apenas o decisor que conecta drift → retraining.

**Tarefas:**
1. Crie `src/monitoring/retraining_trigger.py`:
   - Lê `metrics/drift/drift_summary.json` (erro claro se ausente).
   - Decide retraining se `n_drifted >= 3` OU se alguma feature de `TOP_FEATURES` (importe de `src.monitoring.drift_report`) está em `drifted_features`.
   - Com `--dry-run` (default): só loga a decisão e sai com exit code 0 (sem retraining) ou 42 (retraining recomendado). Com `--execute`: roda `dvc repro` via subprocess e loga o resultado.
   - Siga os padrões do projeto: loguru com `setup_logger()`, paths de `src.config.paths`, docstrings no estilo dos módulos vizinhos.
2. Crie `.github/workflows/retraining-check.yml`: agendado (cron semanal) + `workflow_dispatch`; roda o drift report em modo `--simulate` seguido do trigger em `--dry-run`, e **falha o job (vermelho) se exit code 42** — o vermelho É o alerta. Use os workflows existentes em `.github/workflows/` como referência de estilo (uv, cache, Python 3.12).
3. Testes em `tests/test_retraining_trigger.py`: decisão com summary sintético (3 cenários: sem drift, drift em features irrelevantes < 3, drift em top feature). Use `tmp_path` + monkeypatch como em `tests/test_observability.py`.
4. Doc `docs/diario/fase5-anexo-gatilho-retraining.md` no formato do diário (inclua a seção "Teoria ↔ prática" citando o cap. 9 de Chip Huyen — continual learning, estágio "automated retraining") + linha na tabela do `docs/diario/README.md`.

**Critérios de aceite:**
- `uv run python -m src.monitoring.drift_report --simulate && uv run python -m src.monitoring.retraining_trigger --dry-run; echo $?` imprime 42 (drift simulado nas top features ⇒ retraining recomendado).
- Sem o arquivo de summary: mensagem de erro clara, não traceback cru.
- `uv run pytest tests/ -q` passa (32 testes existentes + os novos).
- `uv run ruff check .` e `uv run mypy src/ app/` limpos.
- Um commit no padrão do repo (veja `git log --oneline -5`, com `Co-Authored-By` ao final). NÃO faça push.

**Guard-rails:** não altere `params.yaml`, o gate de promoção em `src/model_evaluation/evaluate_model.py`, os schemas, nem `drift_report.py` (apenas importe dele). O workflow NÃO deve fazer push de modelo nem mexer no registry — ele só detecta e alerta.
