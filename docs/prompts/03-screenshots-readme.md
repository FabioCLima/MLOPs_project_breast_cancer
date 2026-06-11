# Prompt — Capturar screenshots das UIs para o README

> Cole o texto abaixo numa sessão do Claude Code (modelo Sonnet/Haiku) na raiz do projeto. Requer Docker rodando.

---

Capture screenshots das interfaces do projeto e adicione-as ao README, na seção "Quickstart" (logo após a tabela de serviços do compose).

**Contexto:** projeto MLOps com stack completo no docker-compose (API com Swagger, MLflow, Grafana). O README é em inglês; o objetivo é que um recrutador VEJA o sistema funcionando sem rodar nada.

**Tarefas:**
1. Garanta artefatos e histórico de runs: `uv run dvc repro` (se `mlflow.db` não existir, isso o popula).
2. Suba o stack: `docker compose up -d --build`. Gere tráfego real: ~30 requests no `/predict` usando linhas de `data/raw/raw.csv` (30 features de `src/config/features.py`).
3. Capture 4 screenshots (use uma ferramenta headless — ex.: `npx playwright screenshot` ou um script python com playwright; instale o que precisar localmente, sem sudo):
   - `docs/img/swagger.png` — http://localhost:5001/docs
   - `docs/img/mlflow-runs.png` — http://localhost:5000 com o experimento `breast-cancer-classifier` aberto (lista de runs)
   - `docs/img/mlflow-registry.png` — página do modelo registrado mostrando o alias `production`
   - `docs/img/grafana.png` — http://localhost:3000/d/model-serving (login admin/admin) com os painéis populados pelo tráfego do passo 2 (aguarde ~1 min de scrape antes de capturar)
4. Adicione as imagens ao README em inglês com legendas curtas (uma sub-seção "Screenshots", colapsável com `<details>` para não inflar a página).
5. Derrube o stack (`docker compose down`).

**Critérios de aceite:**
- 4 PNGs em `docs/img/`, cada um < 500 KB (reduza resolução/qualidade se preciso — há um hook de pre-commit que bloqueia arquivos > 500 KB).
- O painel do Grafana aparece COM dados (não vazio) e o MLflow mostra múltiplas runs.
- README renderiza as imagens (confira os paths relativos).
- `uv run ruff check .` limpo; um commit no padrão do repo (veja `git log --oneline -5`, com `Co-Authored-By`). NÃO faça push.

**Guard-rails:** não altere nada além do README e `docs/img/`. Não commite `mlflow.db`, `mlruns/` nem dados (já estão no .gitignore — não os remova de lá). Se uma UI não carregar, diagnostique com `docker compose logs <serviço>` antes de tentar de novo.
