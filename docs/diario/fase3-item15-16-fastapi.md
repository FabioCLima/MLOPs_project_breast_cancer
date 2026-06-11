# Fase 3 — Itens 15+16: Serving moderno com FastAPI + Pydantic v2

> **Backlog:** itens 15 e 16 · **Capítulo relacionado:** 7 (Implantação do modelo e serviço de predição)

## O que foi feito

Flask saiu; entrou **FastAPI com application factory**:

| Endpoint | Função |
|----------|--------|
| `GET /health` | liveness/readiness (app de pé + modelo carregado) |
| `GET /metadata` | versão do modelo, threshold, features, métricas de teste |
| `POST /predict` | lote JSON validado pelo Pydantic |
| `POST /predict/batch` | upload de CSV |
| `GET /docs` | Swagger/OpenAPI gerado automaticamente do contrato |

Decisões estruturais:

- **`create_app(model_service=None)` + lifespan:** o modelo carrega na inicialização do *servidor*, nunca no import. Importar `app.api` não exige modelo treinado — e os testes injetam um serviço fake. (O Flask antigo carregava o modelo no import do módulo: impossível de testar sem artefatos, e qualquer ferramenta que importasse o módulo pagava o custo do TensorFlow.)
- **Contrato Pydantic gerado do contrato canônico:** as 30 features viram um modelo Pydantic via `create_model()` a partir de `FEATURE_COLUMNS` — a *mesma lista* que o pipeline e o Pandera usam. Schema do treino e schema do serving não podem divergir porque são o mesmo objeto. `extra="forbid"` rejeita campos desconhecidos; `ge=0` rejeita valores impossíveis; nomes com espaço viram aliases (`mean radius` ⇄ `mean_radius`).
- **Modelo vem do MLflow Registry** (`models:/breast-cancer-classifier@production`), com fallback para o arquivo local (e warning) se o registry não estiver disponível. A resposta da API carrega `model_version` — toda predição é auditável até a run que gerou o modelo.
- **Threshold vem da avaliação** (`evaluation.json`): produção opera no mesmo ponto da curva precision-recall que o item 12 aprovou. Smoke test real: `/metadata` reporta `v2`, threshold `0.88`.

## Os testes (item 16) — 11 testes de contrato sem nenhum artefato real

`tests/test_api.py` usa `TestClient` com o serviço fake injetado:

- caminho feliz JSON e CSV (labels e probabilidade conferidos exatamente, graças ao modelo fake determinístico);
- **422** para: feature faltando, tipo errado, valor negativo, campo extra, lista vazia — com a feature problemática nomeada no corpo do erro;
- **400** para: CSV sem coluna obrigatória, arquivo não-CSV.

Diferença para o Flask antigo: erro de schema retornava HTML 200 com mensagem embutida; agora é HTTP semântico com JSON estruturado — integrável por qualquer cliente.

## Por que FastAPI (e não manter Flask)?

1. **Validação é o produto:** numa API de ML, 80% do risco está na entrada. Pydantic transforma o contrato em código declarativo com erros automáticos e precisos — no Flask isso era um `if` manual por regra.
2. **OpenAPI de graça:** `/docs` é demo interativa e documentação sempre atualizada (gerada do código, não escrita à parte).
3. **Lifespan bem definido** para carregar recursos pesados — o padrão que o Flask só alcança com extensões e disciplina.
4. ASGI/async preparado para I/O concorrente (chamadas a feature store, model server remoto) sem reescrever.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 7:** *online prediction* com API síncrona é exatamente este desenho; a discussão dos "mitos de deployment" (mito: deploy é só colocar atrás de um endpoint) aparece aqui como tudo que cerca o endpoint — contrato, versão, threshold, health, testes.
- O capítulo insiste na diferença entre o **artefato do modelo** e o **serviço de predição**: nosso serviço conhece o modelo só pelo alias do registry — eles têm ciclos de vida independentes (re-treinar não exige redeploy; redeploy não exige re-treinar).

## Como validar

```bash
uv run pytest tests/test_api.py -q       # 11 passed, sem modelo real
uv run uvicorn --factory app.api:create_app --port 5001
curl localhost:5001/health               # {"status":"ok","model_loaded":true}
# http://localhost:5001/docs — Swagger interativo
```

## Lição para levar

Serving de ML maduro se reconhece por três perguntas: *De onde veio este modelo?* (registry, versão na resposta), *Que decisão ele aplica?* (threshold explícito e auditável), *O que acontece com entrada ruim?* (contrato que rejeita com erro claro). O framework importa menos — mas o FastAPI torna as três respostas quase inevitáveis, e o Flask as deixava todas por conta da disciplina do autor.
