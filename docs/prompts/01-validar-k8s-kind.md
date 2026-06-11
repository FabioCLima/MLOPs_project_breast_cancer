# Prompt — Validar deploy Kubernetes local com kind (+ demo do HPA)

> Cole o texto abaixo numa sessão do Claude Code (modelo Sonnet/Haiku) na raiz do projeto.

---

Valide o deploy Kubernetes deste projeto usando kind, seguindo o passo a passo já documentado em `deploy/k8s/README.md`. Não altere os manifests em `deploy/k8s/` exceto se um deles falhar por erro objetivo (nesse caso, explique o erro antes de corrigir).

**Contexto:** projeto MLOps com API FastAPI containerizada (imagem `ml-classifier:dev`, build local). Os manifests (Deployment com probes, Service, HPA) já foram validados com kubeconform, mas nunca aplicados num cluster real. Docker funciona na máquina (WSL2); kind pode precisar ser instalado.

**Tarefas, em ordem:**
1. Instale o kind se ausente (binário oficial em ~/.local/bin; não use sudo).
2. Garanta que os artefatos do modelo existem (`uv run dvc repro` se necessário) e builde a imagem: `docker build -t ml-classifier:dev .`
3. Siga o `deploy/k8s/README.md`: criar cluster, carregar imagem, `kubectl apply`, acompanhar rollout.
4. Instale o metrics-server no kind (instruções no mesmo README, incluindo o ajuste `--kubelet-insecure-tls`).
5. Gere carga sintética no endpoint `/predict` (port-forward + loop de curl com payload válido — use as 30 features de `src/config/features.py` com valores do `data/raw/raw.csv`) e capture o HPA escalando: `kubectl get hpa -w`.
6. Registre as evidências num novo doc `docs/diario/fase4-anexo-validacao-kind.md` seguindo o formato dos outros docs do diário (seções: O que foi feito / Como validar / saídas de comando relevantes coladas). Não invente seção de teoria — anexo de validação não precisa.
7. Destrua o cluster ao final (`kind delete cluster --name mlops`).

**Critérios de aceite (todos verificáveis por comando):**
- `kubectl rollout status deployment/breast-cancer-api` conclui com sucesso, 2/2 pods Ready.
- `curl localhost:8080/health` via port-forward retorna `{"status":"ok","model_loaded":true}`.
- Saída do `kubectl get hpa` mostra réplicas > 2 durante a carga (cole a evidência no doc).
- Doc novo no diário + linha nova na tabela do `docs/diario/README.md`.
- `uv run ruff check .` limpo; commit único no padrão do repo (veja `git log --oneline -5` como referência de estilo, com `Co-Authored-By` ao final). NÃO faça push.

**Guard-rails:** não modifique `params.yaml`, `dvc.yaml`, `src/`, `app/`, nem os workflows de CI/CD. Se algo do WSL2/kind travar por mais de ~3 tentativas com a mesma falha, pare e relate o erro em vez de improvisar workarounds.
