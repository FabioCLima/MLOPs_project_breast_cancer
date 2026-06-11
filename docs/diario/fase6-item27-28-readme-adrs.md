# Fase 6 — Itens 27+28: README como case técnico e ADRs

> **Backlog:** itens 27 e 28 · **Capítulo relacionado:** 2 (requisitos de sistema); 11 (comunicação e responsabilidade)

## O que foi feito

- **README reescrito como case técnico** (em inglês — a audiência inclui recrutadores internacionais): pitch em duas frases, diagrama Mermaid do sistema completo, **tabela de resultados com o achado honesto** (LR empata com o MLP), quickstart reproduzível, destaques de engenharia, e a seção de decisões — incluindo as que NÃO foram tomadas.
- **5 ADRs** (`docs/adr/`), uma página cada, formato Status/Contexto/Decisão/Racional/Consequências:
  1. DVC como orquestrador (orquestração por conteúdo vs. por tempo)
  2. FastAPI no lugar de Flask (validação como produto)
  3. MLflow local-first; W&B fora (offline reproduzível)
  4. Threshold como parâmetro de negócio versionado (com o caso real do gate)
  5. **Sem feature store** — com os três problemas que Feast resolve e a demonstração de que este projeto não tem nenhum

## Por que o README tem essa forma

Um README de portfólio tem dois leitores com orçamentos de tempo diferentes:

- **Recrutador (3 minutos):** badges → pitch → diagrama → tabela de resultados. Precisa sair entendendo *o que é* e *que nível de engenharia demonstra*.
- **Engenheiro (30 minutos):** quickstart que funciona de primeira, highlights com links para o código e os testes, ADRs para as decisões. Precisa conseguir *reproduzir e auditar*.

Duas escolhas deliberadas: a tabela de resultados **lidera com o baseline vencendo** — credibilidade técnica vale mais que glamour de deep learning; e a seção "o que NÃO usei" existe porque, em nível sênior, justificar ausências pesa tanto quanto justificar presenças.

## Por que ADRs (e não só o diário)

O diário conta a *história* (didática, cronológica); o ADR registra a *decisão* (atemporal, consultável). Daqui a um ano, "por que não tem Airflow?" se responde em uma página, com os critérios que mudariam a resposta. ADRs também são um hábito de equipe sênior que entrevistadores reconhecem imediatamente.

## Teoria ↔ prática (Chip Huyen)

- **Cap. 2:** os quatro requisitos (confiabilidade, escalabilidade, manutenibilidade, adaptabilidade) estruturam os "engineering highlights" do README — cada highlight responde a um deles.
- **Cap. 11:** comunicação responsável — limitações em destaque, model card linkado, e o aviso "não usar clinicamente" no topo, não no rodapé.

## Lição para levar

Documentação de portfólio não é descrição do código — é **argumento de engenharia**: o que foi construído, por que essas escolhas, contra quais alternativas, com que evidência. Se o leitor consegue discordar de uma decisão sua *usando os critérios que você mesmo documentou*, a documentação cumpriu o papel.
