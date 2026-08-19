# Zammad LLM Customer Service Agent

## What this is

An LLM-powered agent that plugs into **Zammad** (open-source helpdesk) to read
incoming support tickets, classify intent, retrieve relevant context via RAG,
generate a response, and decide whether to auto-reply or escalate to a human.
Zammad stays the single source of truth for ticket data — the agent reads
from and writes back to it via the REST API, it doesn't maintain its own
parallel ticket store.

## How I want to work with Claude Code on this

I'm intentionally moving away from vibe coding. I don't want a working app
handed to me that I can't explain. Follow these rules on every task in this
project, no exceptions unless I explicitly say "just do it quickly":

1. **Explain before you generate.** Before writing code for a new piece,
   give me a short plain-English explanation of the approach and *why* it's
   the right one here (vs. obvious alternatives) — 3-6 sentences is enough.
2. **Small diffs, not big drops.** Build one function/endpoint/module at a
   time. Don't scaffold the whole service in one shot. I want to review and
   understand each piece before the next one lands on top of it.
3. **Comment the "why," not the "what."** Skip comments like `# loop over
   tickets`. Add comments where a decision was made that isn't obvious from
   the code itself (e.g. why polling was rejected in favor of webhooks).
4. **Ask me to restate it back.** After a non-trivial piece (auth flow, RAG
   retrieval, confidence-scoring logic), ask me a quick question that checks
   I understand what the code does — not a quiz, just a sanity check.
5. **Flag the risky parts.** Anywhere secrets, auth tokens, or user input
   touch the code, call it out explicitly rather than quietly handling it
   "safely" in a way I never see.
6. **No unexplained dependencies.** If you pull in a library I haven't used
   before, tell me in one line what it's for and what the lightweight
   alternative would have been.

## Architecture

### 1. Integration Layer (Zammad connection)
- Zammad REST API, wrapped in a small Python client (`requests`-based)
- Auth via Zammad API Token
- Core endpoints:
  - `GET /api/v1/tickets` — list tickets
  - `GET /api/v1/ticket_articles/by_ticket/{id}` — get a ticket's message thread
  - `POST /api/v1/ticket_articles` — post a reply into a ticket
  - `PUT /api/v1/tickets/{id}` — update ticket state/tags

### 2. LLM Core Pipeline
Four stages per incoming ticket message:
1. **Intent Classification** — what does the customer actually want?
2. **Context Retrieval (RAG)** — pull relevant knowledge-base chunks
3. **Response Generation** — system prompt + retrieved context → LLM reply
4. **Confidence Decision** — auto-respond vs. escalate to a human

### 3. Structured Output
LLM must return strict JSON, e.g.:
```json
{
  "intent": "registration_issue",
  "confidence": 0.87,
  "action": "respond_with_video",
  "response_text": "...",
  "video_link": "https://...",
  "escalate": false
}
```

### 4. Model routing: DeepSeek vs. Gemini
- **DeepSeek**: primary model for text reasoning (intent classification,
  response generation)
- **Gemini**: used specifically for **multimodal** input — customers who
  attach screenshots

### 5. Knowledge Base / RAG
- Source docs: Markdown files and/or Google Sheets
- Embedded into a vector DB (Chroma or Qdrant — pick one, don't run both)
- Semantic search retrieves top-k relevant chunks → injected into the prompt
- Purpose: ground responses in real docs to reduce hallucination

### 6. Zammad → Agent trigger (webhook, not polling)
- Zammad **Trigger** (Manage → Triggers) fires a **Webhook** on new
  ticket/message events
- Webhook POSTs to the agent's FastAPI endpoint
- Event-driven by design — no polling loop

### 7. Escalation logic
Escalate to a human when:
- Confidence score is below threshold
- Sentiment analysis flags frustration/negative tone
- Repeated back-and-forth without resolution

Escalated tickets get tagged `needs_human` in Zammad for a human agent to pick up.

### 8. Logging & Monitoring
- Every interaction (ticket id, intent, confidence, action taken) logged to Postgres
- Streamlit dashboard to review prompts, outputs, and confidence trends over time

### 9. Deployment
- FastAPI app, containerized with Docker, deployed as a microservice
- Secrets (API keys, Zammad token) via Kubernetes Secrets — never hardcoded,
  never committed to git
- HTTPS only
- Horizontal Pod Autoscaling (HPA) for load

### 10. Security notes
- Secrets excluded from git via `.gitignore` + env vars / K8s secrets
- Rate limiting on the webhook endpoint to prevent abuse

## Proposed phased build order

> The original roadmap doc was only partially legible — this ordering is my
> proposed sequence based on dependency order (each phase only needs what
> came before it). Adjust if you had a different sequence in mind.

**Phase 1 — Zammad connection (read-only)**
Get `GET /api/v1/tickets` working with the API token. Confirm you can read
ticket lists and thread content before building anything else. This is the
foundation everything else depends on.

**Phase 2 — Intent classification + structured output**
Feed a ticket thread to the LLM, get back the strict JSON shape. No RAG, no
auto-reply yet — just prove the classification + structured output loop works.

**Phase 3 — RAG / knowledge base**
Stand up the vector DB, embed your docs, wire retrieval into the prompt.
Test retrieval quality on its own before connecting it to response generation.

**Phase 4 — Response generation + write-back**
Generate real replies and post them via `POST /ticket_articles`. Start with
a dry-run mode (log the reply, don't post it) before letting it write to
live tickets.

**Phase 5 — Confidence & escalation logic**
Add the confidence threshold and sentiment check, wire up the `needs_human`
tag path.

**Phase 6 — Webhook trigger (event-driven)**
Replace manual test-runs with the real Zammad Trigger → Webhook → FastAPI flow.

**Phase 7 — Logging, dashboard, deployment**
Postgres logging, Streamlit dashboard, then Docker/K8s deployment with
secrets and rate limiting.

## Next immediate step

Test the Zammad connection: `GET /api/v1/tickets` with the API token, to
confirm read access before writing any pipeline code.
