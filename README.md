# Agentic AI + LiteLLM Gateway

Production-grade agentic AI running in **Docker Desktop** — no Kubernetes.

---

## What This Project Does (Plain English)

You ask a question. Two AI agents work together to research and answer it,
with safety checks, cost tracking, quality monitoring, and full observability.

| Problem | Solution |
|---|---|
| Every question costs money | LiteLLM semantic cache — same question = free |
| OpenAI goes down | Auto-fallback: gpt-4o-mini → claude-haiku → groq |
| Malicious prompts / PII | Guardrails block/redact before AI ever sees it |
| No idea what it costs | Per-call cost tracked in Langfuse + Prometheus |
| Which prompt works best | Langfuse prompt registry — change without deploy |
| AI quality unknown | RAGAS offline eval + CI quality gate |
| No debugging visibility | OTel traces → Tempo, logs → Loki, metrics → Grafana |
| Research uses fake/hardcoded data | **Tavily real-time web search** with Redis caching |

---

## Pipeline (what the code actually does)

```
User Request
     │
     ▼
[1] Input Guardrails              app/guardrails/validator.py
     │  PII redacted, injection blocked, empty rejected
     │  Blocked → return error immediately
     │
     ▼
[2] Orchestrator                  app/agents/orchestrator.py
     │  (quick gateway health check ping — not routing)
     │
     ├──► Research Agent          app/agents/research_agent.py
     │      │
     │      ├── LLM call: decompose query
     │      └──► LiteLLM Proxy ──► LLM Provider
     │             cache check, fallback, cost logged
     │
     ├── search_tool (Tavily API, Redis cached)
     │     └──► Tavily Web Search ──► real-time results
     │            OTel span + Prometheus metrics emitted
     │
     └──► LLM call: synthesise research brief
            └──► LiteLLM Proxy ──► LLM Provider
     │
     └──► Synthesis Agent         app/agents/synthesis_agent.py
            │
            ├── LLM call: draft response
            │     └──► LiteLLM Proxy ──► LLM Provider
            │
            └── LLM call: self-critique
                  └──► LiteLLM Proxy ──► LLM Provider
     │
     ▼
[3] Output Guardrails             app/guardrails/validator.py
     │  toxicity, PII leakage, length check
     │
     ▼
Response: { response, cost_breakdown, llm_cache_hits, trace_id }
```

## Project Structure

```
agentic-llm-gateway/
├── app/
│   ├── main.py                    FastAPI + all endpoints
│   ├── agents/
│   │   ├── orchestrator.py        pipeline phases 1-4
│   │   ├── research_agent.py      Agent 1 + Langfuse spans + prompt fetch
│   │   └── synthesis_agent.py     Agent 2 + Langfuse spans + prompt fetch
│   ├── gateway/
│   │   ├── litellm_client.py      all LLM calls go here -> proxy
│   │   └── cache.py               Redis cache for tool results
│   ├── guardrails/validator.py    input + output safety checks
│   ├── ragas_eval/evaluator.py    RAGAS metrics (offline use only)
│   ├── observability/
│   │   ├── metrics.py             Prometheus metrics
│   │   ├── tracing.py             OpenTelemetry -> Tempo
│   │   ├── langfuse_tracker.py    prompt versioning, traces, cost, datasets
│   │   └── logging_config.py      structlog JSON -> direct Loki push
│   ├── tools/search_tool.py       Tavily real-time web search (Redis cached)
│   └── config/settings.py         all config via .env
├── prompts/                       Git source of truth for prompts
│   ├── prompts.yaml               registry + quality gate thresholds
│   └── *.txt                      6 prompt template files
├── docker/
│   ├── litellm_config.yaml        gateway: cache, routing, fallbacks
│   ├── docker-compose.yml         8 services, Docker Desktop only
│   ├── prometheus.yml
│   ├── loki-config.yaml
│   ├── tempo.yaml
│   └── grafana/provisioning/      auto-wired datasources + pre-built dashboard
├── .github/workflows/
│   ├── prompt-push.yml            merge -> push prompts as staging
│   ├── ragas-eval.yml             PR opened -> eval + score comment
│   └── prompt-promote.yml         git tag -> staging to production
├── scripts/
│   ├── demo.py                    end-to-end demo
│   ├── run_ragas_eval.py          offline RAGAS batch eval + CI gate
│   ├── setup_langfuse.py          first-time prompt registration
│   └── manage_prompts.py          push/pull/promote/diff/status CLI
├── tests/                         84 tests, all mocked, no API keys needed
├── Makefile
├── pyproject.toml
└── .env.example                   pre-wired Langfuse keys, add OPENAI_API_KEY + TAVILY_API_KEY
```

**Where LiteLLM proxy actually sits in the code:**
NOT between guardrails and orchestrator.
It sits between each **agent LLM call** and the LLM provider.
Every `self.gateway.chat_completion()` inside an agent hits the proxy.
4 calls per pipeline (decompose, synthesis, draft, critique).
Cache, routing, fallbacks, cost tracking happen transparently on each call.

RAGAS runs OFFLINE only — not in this pipeline:
- `scripts/run_ragas_eval.py` — batch eval, CSV report, CI quality gate
- `POST /api/v1/eval/ragas`   — on-demand in dev/staging only
---

## Why No Promtail?

Promtail ships logs from files/container stdout — it exists for Kubernetes
where pods write to disk and something needs to collect them. Here we have a
Python app that pushes logs **directly** to Loki over HTTP using
`python-logging-loki`. One fewer container, no Docker socket needed.

---

## How LiteLLM Tracks Cost

Every agent LLM call (4 per pipeline) goes through the same proxy:

```
agent.chat_completion() → LiteLLM proxy
  1. Cache check (Redis cosine sim >= 0.85)
       HIT  → return cached, cost = $0.000000
       MISS → forward to LLM provider
  2. LLM returns: prompt_tokens=310, completion_tokens=180
  3. Proxy calculates:
       (310/1000 * $0.00015) + (180/1000 * $0.00060) = $0.0001545
  4. Logs to Langfuse: { agent_name, model, tokens, cost_usd }
  5. litellm_client.py also emits to Prometheus for Grafana
```

Total request cost = sum of all 4 calls. Tracked per-agent in both places.

---

## Langfuse Integration

Open: **http://localhost:3001** — login: `admin@example.com` / `password`

Keys are pre-wired in docker-compose and `.env.example` — no manual setup.

**1. Prompt Versioning** — all 6 prompts live in Langfuse, not hardcoded:
```bash
make setup-langfuse    # register prompts once after 'make up'
# Edit any prompt in the UI → next request uses new version, no restart
```

**2. Trace Hierarchy** — one per user request:
```
Trace: agentic_pipeline  [cost: $0.000583]  [3.2s]
  Span: research_agent
    Generation: decompose   [310+180 tokens, $0.000155, prompt v2]
    Generation: synthesize  [820+420 tokens, $0.000375, prompt v1]
  Span: synthesis_agent
    Generation: draft       [cache HIT, $0.000000, prompt v3]
    Generation: critique    [580+120 tokens, $0.000159, prompt v2]
```

**3. Cost filtering** — filter by agent_name, user_id, date, model in UI

**4. RAGAS scores** — offline eval attaches faithfulness/relevancy to traces,
letting you correlate quality with specific prompt versions

**5. Dataset** — every request auto-logged to `production_qa_pairs` for eval

---

## Prompt CI/CD with GitHub Actions

Git is source of truth. Langfuse is the runtime store.

```
prompts/
  prompts.yaml                  registry + quality gate thresholds
  research_system_prompt.txt
  research_query_prompt.txt
  research_synthesis_prompt.txt
  synthesis_system_prompt.txt
  synthesis_draft_prompt.txt
  synthesis_critique_prompt.txt
```

Release flow:
```
Edit prompts in a branch
        ↓
Open PR → ragas-eval.yml runs
  diffs local vs Langfuse, runs RAGAS, comments score on PR
  fails if avg score < 0.60
        ↓
PR merged → prompt-push.yml runs
  pushes to Langfuse as label="staging"
  production agents still use label="production" (safe)
        ↓
git tag v1.2.0 && git push --tags → prompt-promote.yml runs
  promotes staging → production
  agents use new prompts on next request, no restart
```

Manual commands:
```bash
make prompt-diff      # diff local vs Langfuse production
make prompt-push      # push local → Langfuse as staging
make prompt-promote   # staging → production
make prompt-status    # show all versions and labels in Langfuse
```

GitHub secrets needed: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
`LANGFUSE_HOST`, `OPENAI_API_KEY`, `TAVILY_API_KEY`

---

## Tavily Search Tool

`app/tools/search_tool.py` is the Research Agent's web-search tool.

- Uses `tavily-python` / `AsyncTavilyClient` with `TAVILY_API_KEY`
- Runs Tavily `search_depth="advanced"` and returns title, snippet, source URL, and relevance score
- Caches each `{ query, max_results }` result in Redis for 1 hour
- Emits `tool.search` OpenTelemetry spans plus Prometheus tool latency/call metrics
- Falls back to an honest empty `source="stub"` result when `TAVILY_API_KEY` is missing, `tavily-python` is not installed, or Tavily errors

Local setup:
```bash
cp .env.example .env
# Set TAVILY_API_KEY=tvly-...
make setup
```

Without a Tavily key, the pipeline still runs, but research context will say no
search results were found instead of using fake or hardcoded data.

---

## Observability

```
What                      Tool           Where
─────────────────────────────────────────────────────────────────
Request latency p95       Prometheus     Grafana -> LLM Latency panel
Token usage by agent      Prometheus     Grafana -> Token Usage panel
Cost per agent (USD/min)  Prometheus     Grafana -> Cost panel
Cache hit rate            Prometheus     Grafana -> Cache Hit Rate panel
Guardrail violations      Prometheus     Grafana -> Violations panel
Tavily search latency     Prometheus     Grafana -> Tool Latency panel
Tavily call count/errors  Prometheus     Grafana -> Tool Calls panel
Full request trace        OTel -> Tempo  Grafana -> Explore -> Tempo
  (includes tool.search span per Tavily call)
Per-call LLM cost/detail  Langfuse       :3001 -> Traces
Prompt version history    Langfuse       :3001 -> Prompts
App logs (JSON)           structlog      Grafana -> Explore -> Loki
                          direct HTTP push to Loki (no Promtail needed)
```

Grafana datasources (Prometheus, Loki, Tempo) auto-provisioned on startup.
Pre-built 14-panel dashboard loads automatically. Logs linked to traces via
trace_id field injected into every log line.

---

## Quick Start

```bash
cp .env.example .env        # add OPENAI_API_KEY and TAVILY_API_KEY at minimum
make setup                  # install Python deps with uv
make up                     # start 8 Docker services
# wait ~30s for Langfuse to initialise
make setup-langfuse         # register all prompts (once)
make app                    # FastAPI on :8000

# Try it
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the benefits of LiteLLM?", "user_id": "chandu"}'

make demo    # rich end-to-end demo with cost breakdown
make ragas   # offline RAGAS batch eval -> CSV
make test    # 84 tests
```

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| App + API Docs | http://localhost:8000/docs | — |
| LiteLLM Proxy | http://localhost:4000 | — |
| Grafana | http://localhost:3000 | admin / admin |
| Langfuse | http://localhost:3001 | admin@example.com / password |
| Prometheus | http://localhost:9090 | — |

---

## Docker Services (Docker Desktop only, no K8s)

| Container | Port | Role |
|---|---|---|
| agentic-redis | 6379 | LiteLLM semantic cache backend |
| agentic-litellm | 4000 | LLM gateway proxy |
| agentic-postgres | 5432 | Langfuse database |
| agentic-langfuse | 3001 | LLM observability UI |
| agentic-prometheus | 9090 | Metrics storage |
| agentic-loki | 3100 | Log storage |
| agentic-tempo | 3200/4317 | Trace storage |
| agentic-grafana | 3000 | Unified dashboards |

FastAPI app runs locally with `make app` (port 8000).

---

## LiteLLM Config Reference

```yam
# docker/litellm_config.yaml

cache:
  type: redis
  similarity_threshold: 0.85   # cosine sim threshold for cache hit
  ttl: 3600

router_settings:
  routing_strategy: latency-based-routing

model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      rpm: 100
      tpm: 100000

litellm_settings:
  fallbacks:
    - gpt-4o-mini:
      - claude-3-haiku-20240307
      - groq/llama3-8b-8192
  success_callback: ["langfuse"]   # every call logged with cost
  failure_callback: ["langfuse"]
```
