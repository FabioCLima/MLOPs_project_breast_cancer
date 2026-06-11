# Deploy local com kind

Pré-requisitos: [kind](https://kind.sigs.k8s.io/) e kubectl. Os artefatos do
modelo precisam existir (`uv run dvc repro`) antes do build da imagem.

```bash
# 1. Build da imagem e cluster local
docker build -t ml-classifier:dev .
kind create cluster --name mlops
kind load docker-image ml-classifier:dev --name mlops

# 2. Deploy
kubectl apply -f deploy/k8s/

# 3. Acompanhar probes e rollout
kubectl rollout status deployment/breast-cancer-api
kubectl get pods -w

# 4. Testar a API
kubectl port-forward svc/breast-cancer-api 8080:80 &
curl localhost:8080/health

# 5. Ver o HPA (requer metrics-server; no kind:
#    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
#    e adicionar --kubelet-insecure-tls ao deployment do metrics-server)
kubectl get hpa breast-cancer-api -w

# Limpeza
kind delete cluster --name mlops
```

## O que cada manifesto demonstra

- **deployment.yaml** — 2 réplicas, non-root, resource requests/limits,
  **liveness** (reinicia pod travado) vs **readiness** (segura tráfego até o
  modelo carregar — modelos grandes demoram, e é o readiness que evita servir
  erro durante o warm-up).
- **service.yaml** — endpoint estável na frente dos pods; o load balancing
  entre réplicas é nativo.
- **hpa.yaml** — escala horizontal por CPU (2→6 réplicas a 70%); inferência
  de ML é CPU-bound, então CPU é um proxy razoável de carga aqui.
