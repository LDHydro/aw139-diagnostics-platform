# Enterprise LLM Platform

A self-hosted LLM deployment for an aviation maintenance organisation, built to
run on a single workstation with an RTX 5090. Nothing leaves the building: the
models, the documents and the fleet data all stay on your hardware.

It does six things:

| Capability | Endpoint |
|---|---|
| Answers plain-language questions from your governing documents, **with citations** | `POST /v1/ask` |
| Consults your **other internal LLM/AI systems** and attributes what they say | same call, `[A#]` references |
| **Predictive maintenance scheduling** from your standard maintenance programme | `/v1/maintenance/*` |
| **MEL dispatch decision support** — may we dispatch, under what conditions, until when | `/v1/mel/*` |
| **LaTeX** authoring and PDF compilation | `POST /v1/latex` |
| **Application development** assistance over your own repositories | `POST /v1/dev` |
| A drop-in **OpenAI-compatible API** so existing in-house apps integrate unchanged | `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` |

Access is governed by **Active Directory group membership** through your
existing SSO, down to the individual document.

> **New here?** Follow [`GUIDE.md`](GUIDE.md) instead — it walks you from an
> empty machine to a working department system in order, with checkpoints at
> each step, a troubleshooting table, and a roadmap of further use cases. This
> README is the reference: what everything is and how it is configured.

---

## 1. What runs where

```
                    ┌──────────────────── your workstation ────────────────────┐
  in-house apps     │                                                          │
  IDE assistants ──►│  ELP gateway  :8080   (FastAPI: auth, RAG, scheduling)    │
  browsers          │        │                                                 │
                    │        ├──► vLLM chat        :8101  ─┐                    │
  Entra ID / ADFS ─►│        ├──► vLLM embeddings   :8102  ├─ one RTX 5090      │
  (token validation)│        ├──► vLLM reranker     :8103  ─┘   32 GB           │
                    │        └──► PostgreSQL + pgvector :5432                   │
                    └──────────┬───────────────────────────────────────────────┘
                               │
                               ▼  outbound only, on your LAN
                    other internal LLM/AI systems
```

The gateway is stateless apart from Postgres, so you can put it behind a
reverse proxy and run more than one if you outgrow a single host.

### Hardware notes

The install script reads your actual CPU rather than assuming one — core count
and NUMA layout differ between Threadripper and Ryzen 9, and the thread
settings should follow what is really fitted.

**The external GPU matters in exactly one way.** An eGPU sits behind a
Thunderbolt or OCuLink link that is far narrower than a desktop x16 slot.
Weights cross it once at start-up, so the first load of a 32B model takes
minutes rather than seconds. After that everything runs from VRAM and
inference speed is unaffected. The one rule: **never enable CPU offload or
vLLM swap space on an eGPU.** Offloaded tensors cross that narrow link on
every single token and throughput collapses.

The RTX 5090 is Blackwell (`sm_120`). It needs a **570-series or newer driver**
and a **CUDA 12.8** build of PyTorch. Wheels built for older architectures
install cleanly and then fail at runtime with *"no kernel image is available
for execution on the device"*, so `deploy/install_host.sh` checks the arch list
and runs a real matmul on the GPU before it goes any further.

### VRAM budget (32 GB, default `balanced` profile)

| Process | Model | VRAM |
|---|---|---|
| Chat / Q&A / LaTeX / code | Qwen3-32B-AWQ (4-bit), 32k context, fp8 KV cache | ~25.0 GB |
| Embeddings | BAAI/bge-m3 (1024 dims, multilingual) | ~2.2 GB |
| Reranker | BAAI/bge-reranker-v2-m3 | ~2.2 GB |
| Headroom | driver, fragmentation | ~2.6 GB |

Two other profiles ship in `config/models.yaml`: `throughput` (a
mixture-of-experts model — faster, more concurrent users, slightly weaker
reasoning) and `long_context` (a smaller model with a 128k window).

> **Start order matters.** vLLM sizes its KV cache from the free memory it sees
> at start-up. The two small servers must start *first* so the large model
> measures last. `deploy/start_models.sh` handles this.

---

## 2. Quick start

```bash
cd enterprise_llm

# 1. Host preparation: driver/CUDA checks, venv, PyTorch cu128, vLLM
deploy/install_host.sh

# 2. Configuration
cp .env.example .env && $EDITOR .env      # database URL + your SSO settings

# 3. Database
docker compose up -d postgres             # or point at your own PostgreSQL 16+

# 4. Models (first load crosses the eGPU link — several minutes is normal)
deploy/start_models.sh balanced

# 5. Schema + an administrator key (printed once)
python scripts/bootstrap.py

# 6. Serve
uvicorn elp.main:app --host 0.0.0.0 --port 8080

# 7. Verify everything end to end
python scripts/smoke_test.py --api-key elp_...
```

For a permanent install, copy the tree to `/opt/elp` and use the units in
`deploy/systemd/` (`elp-models`, `elp-gateway`, and the `elp-scheduler` timer
for the nightly re-forecast).

---

## 3. Access control

### Microsoft Entra ID (recommended)

1. Register an application. Expose an API scope, e.g. `api://<client-id>/access`.
2. Under **Token configuration**, add the **groups claim** and choose
   *Security groups*. Emit `sAMAccountName` if you want readable group names;
   object IDs work equally well — map whatever arrives.
3. Set in `.env`:

```bash
ELP_AUTH__MODE=oidc
ELP_AUTH__OIDC_ISSUER=https://login.microsoftonline.com/<tenant-id>/v2.0
ELP_AUTH__OIDC_CLIENT_ID=<application-client-id>
ELP_AUTH__OIDC_AUDIENCE=api://<application-client-id>
ELP_AUTH__GROUP_ROLE_MAP={"AW139-Maint-Admins":"admin","AW139-Engineering":"engineer","AW139-Planning":"planner"}
ELP_AUTH__ALLOWED_GROUPS=["AW139-Engineering","AW139-Planning","AW139-Maint-Admins"]
```

The platform never sees a password. Your app gets a token from Entra and
presents it as `Authorization: Bearer <token>`; the gateway validates the
signature against the published JWKS and reads group membership from the token.

> **Groups overage.** A user in more than ~200 groups causes Entra to omit the
> groups claim entirely and point at Graph instead. Rather than adding a Graph
> round-trip to every request, the platform falls back to **app roles**, which
> are never truncated. If you have users in that many groups, assign app roles
> in the app registration and map those instead.

**ADFS 2019+** works the same way with `ELP_AUTH__OIDC_ISSUER=https://adfs.corp.internal/adfs`
and `ELP_AUTH__GROUPS_CLAIM=group`. **Keycloak** likewise.

### No identity provider

Set `ELP_AUTH__MODE=ldap` and fill in the LDAP block. The platform binds
directly against a domain controller and issues its own signed session tokens.
Prefer OIDC where you can — it keeps passwords out of this system entirely and
supports MFA.

### Roles

| Role | Can do |
|---|---|
| `reader` | ask questions, read documents and the maintenance forecast |
| `planner` | + record utilisation, build and commit plans, cancel visits |
| `maintenance_manager` | + **approve deferrals** |
| `engineer` | + upload and retire documents, code assistance |
| `developer` | ask, read, code assistance, LaTeX |
| `admin` | everything, including keys, peers and the audit log |

Approving a deferral is a separate permission from planning, because it is an
airworthiness decision rather than routine scheduling.

**Platform admins do not bypass document ACLs by default.** Administering the
system is not the same as being cleared to read every department's governing
documents. Override with `ELP_RAG__ADMIN_BYPASS_ACL=true` if your policy differs.

### In-house applications

Applications authenticate with a service key that carries its own AD groups, so
document-level access control applies to apps exactly as it does to people:

```bash
curl -X POST http://elp:8080/v1/api-keys \
  -H "X-API-Key: $ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"aw139-diagnostics-ui",
       "scopes":["ask","chat","docs:read","maint:read"],
       "groups":["AW139-Engineering"]}'
```

The plaintext key is returned once; only a SHA-256 hash is stored.

---

## 4. Loading the governing documents

Describe the corpus in a manifest so revisions and access groups are
version-controlled rather than typed into a form:

```yaml
# docs/manifest.yaml
defaults:
  department: Maintenance
  allowed_groups: [AW139-Engineering, AW139-Maint-Admins]
documents:
  - file: MOE-Rev-C.pdf
    doc_key: MOE-001
    title: Maintenance Organisation Exposition
    revision: C
    effective_date: 2026-01-15
  - file: quality-manual.docx
    doc_key: QM-001
    title: Quality Manual
    revision: "4"
    allowed_groups: [AW139-Maint-Admins]     # narrower than the default
```

```bash
python scripts/ingest_docs.py --manifest docs/manifest.yaml
```

PDF, DOCX, Markdown, HTML, XML and plain text are supported. Each document is
indexed in its own transaction, so one bad file cannot roll back the rest.
Re-running skips anything whose content hash is unchanged.

**Why the citations are trustworthy.** The chunker tracks section hierarchy as
it walks the document and never lets a passage straddle a clause boundary, so
every chunk knows exactly which section it came from. It also distinguishes a
heading (`4.2.3 Deferral of scheduled maintenance`) from a numbered procedure
step (`1. Remove the access panel.`) — getting that wrong shreds procedures and
produces citations to sections that do not exist.

Retrieval is hybrid: vector search for meaning, Postgres full-text for the
exact identifiers vector search is bad at (part numbers, ATA codes, clause
numbers), fused with reciprocal rank fusion, then reranked by a cross-encoder.
Access control is applied **inside the SQL**, so a passage your groups do not
cover is never loaded into memory and cannot leak through a summary.

### Asking

```bash
curl -X POST http://elp:8080/v1/ask \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"question":"Who can approve deferring a scheduled inspection, and for how long?"}'
```

```json
{
  "answer": "A scheduled inspection may only be deferred with the written approval of the Maintenance Manager [D1]. The deferral may not exceed 30 days or 25 flight hours, whichever comes first [D2]. Airworthiness limitations may not be deferred at all [D2].",
  "references": [
    {"marker":"D1","type":"document","citation":"MOE-001, Rev C, §4.2.3, p. 51",
     "document_title":"Maintenance Organisation Exposition","score":0.94},
    {"marker":"D2","type":"document","citation":"QM-001, Rev 4, §7.1, pp. 22-23"}
  ],
  "confidence": 0.91,
  "grounded": true
}
```

Markers the model invents are **stripped before the answer is returned**, and
`confidence` reflects what retrieval actually found rather than how assured the
prose sounds. When nothing clears the threshold the platform says so and lists
what came closest, instead of guessing on a maintenance question.

---

## 5. Predictive maintenance scheduling

Import your standard maintenance programme. Column names vary between
operators, so the importer matches on aliases (`Task Code` / `TASKNO` /
`Card No.` / `Reference` all work) and reports what it rejected rather than
dropping rows silently:

```bash
curl -X POST http://elp:8080/v1/maintenance/schedule/import \
  -H "X-API-Key: $KEY" \
  -F file=@maintenance-programme.xlsx \
  -F default_models=AW139 \
  -F source_document_key=MOE-001 \
  -F dry_run=true          # check what it makes of your file first
```

Then feed it reality:

```bash
curl -X POST http://elp:8080/v1/maintenance/utilization \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '[{"tail_number":"PP-ABC","day":"2026-08-20","flight_hours":3.2,"cycles":5,"landings":7}]'
```

**The forecast.** A task card may carry several intervals at once — 600 flight
hours *or* 12 months *or* 1000 landings. Each is projected onto the calendar
independently and the earliest one drives the schedule. Two details make the
projection trustworthy:

- **Non-flying days count.** Logbook feeds usually emit rows only for days
  flown. Averaging just those rows says 2.5 FH/day for an aircraft that really
  averages 1.79 — over a 600-hour interval that is a two-month error. Gaps are
  filled with zeros.
- **Smoothing is weekly, not daily.** Flying is strongly weekly, and a
  day-level average lands wherever the last few days fell, so the same fleet
  would forecast differently depending on whether you asked on a Friday or a
  Sunday. Bucketing into whole weeks removes that.

Calendar-driven tasks report full confidence because they need no forecast at
all; hours-driven tasks inherit the confidence of the utilisation estimate.

**The plan.** Tasks due near each other are bundled into one visit, but only
when doing them early does not throw away too much interval — pulling a
20-day task forward by 13 days discards 65% of its life, so it gets its own
slot. Downtime is the greater of the labour requirement and the longest single
task's elapsed time, because a sealant cure does not go faster with more
technicians. Every event carries a rationale; a planner nobody can interrogate
is a planner nobody uses.

```bash
curl -X POST http://elp:8080/v1/maintenance/plan \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"horizon_days":120,"commit":true,"explain":true}'
```

**Cancellations.** This is where schedules usually go wrong in practice: the
slot is dropped, the work quietly disappears from the plan, and a limit is
exceeded weeks later. Cancelling re-forecasts every affected task and builds a
replacement plan **in the same transaction**, and reports anything that can no
longer be completed before its hard limit:

```bash
curl -X POST http://elp:8080/v1/maintenance/events/$ID/cancel \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"reason":"hangar flooded","reschedule":true}'
```

**Deferrals** require `maint:approve`, a named approver and an expiry date, and
are refused outright for airworthiness limitations or beyond the limit the task
card itself allows.

The `elp-scheduler` timer re-forecasts the fleet nightly and exits non-zero if
any aircraft is past a hard limit, so a systemd failure alert means something
real.

---

## 5a. MEL dispatch decision support

Load the approved Minimum Equipment List into the item catalogue:

```bash
curl -X POST http://elp:8080/v1/mel/import \
  -H "X-API-Key: $KEY" \
  -F file=@mel-rev6.xlsx -F source_document_key=MEL-001 \
  -F source_revision=6 -F default_models=AW139 -F dry_run=true
```

Then ask the question a line engineer asks several times a week:

```bash
curl -X POST http://elp:8080/v1/mel/check \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"tail_number":"PP-ABC","item_number":"24-11-01",
       "discovered_on":"2026-08-25","intended_operation":["IFR","night"]}'
```

```json
{
  "decision": {
    "verdict": "go_with_conditions",
    "dispatchable": true,
    "summary": "Dispatch permitted under 24-11-01 (AC Generator), Category C. Rectify by 2026-09-04 (10 day(s) from today). 3 condition(s) must be satisfied first.",
    "expires_on": "2026-09-04",
    "conditions": ["(o) operational procedure: ...", "(m) maintenance procedure: ...", "placard: GEN 2 INOP"],
    "citation": "MEL-001, Rev 6, item 24-11-01"
  }
}
```

Three rules the engine enforces, each a place real operations go wrong:

- **If it is not in the MEL, it must work.** The MEL grants relief, it does not
  restrict it. An unlisted item returns `not_in_mel` and a plain no-go, never a
  cautious pass.
- **The interval excludes the day of discovery.** A Category C item found on the
  1st runs to the end of the 11th. Off-by-one here is an audit finding.
- **Relief is per quantity.** "Two installed, one required" permits exactly one
  inoperative; a second failure of the same item is a no-go even though the MEL
  lists it.

Recording a deferral (`POST /v1/mel/deferrals`) re-runs the full evaluation and
**refuses anything the MEL does not permit** — including a deferral whose (o)/(m)
procedures and placarding have not been confirmed as carried out, because a
deferral recorded before its conditions are met was never valid. Extensions
require `maint:approve`, are one-time, and are refused for Category A and after
the interval has already run out.

`GET /v1/mel/status` gives fleet dispatch status; a single expired item makes
that aircraft undispatchable regardless of everything else. The nightly job
expires overdue deferrals and exits non-zero if any aircraft cannot be
dispatched.

> Index the MEL itself as a document with `doc_type: mel` as well as importing
> the catalogue. The catalogue drives the decisions; the indexed document lets
> `/v1/mel/check` resolve a free-text description to candidate items.

---

## 6. Other internal AI systems

Register peers in `config/peers.yaml` (version-controlled) or at runtime via
`POST /v1/peers`. Three wire formats cover essentially anything internal:
OpenAI-compatible, Anthropic-style `/messages`, and plain REST described by a
small request/response template.

```yaml
peers:
  - name: reliability-analytics
    display_name: Fleet Reliability Analytics
    description: Component removal rates, MTBUR and failure trends.
    protocol: openai
    base_url: https://reliability.corp.internal/v1
    auth_type: bearer
    auth_env_var: PEER_RELIABILITY_TOKEN     # the NAME, never the secret
    capabilities: [reliability, mtbur, failure rate, component removal]
    allowed_groups: [AW139-Engineering]
```

Routing is cheap-first: a capability-tag match handles the common case with no
model call, and the local model only chooses when tags are ambiguous. Peers are
consulted in parallel; **a peer that is down degrades the answer rather than
failing it**. Their contributions are cited as `[A1]`, `[A2]` and are explicitly
marked as secondary to document sources, so a reader can always tell which
claims rest on a manual and which on another system's opinion.

---

## 7. Integrating in-house apps

Point anything that speaks the OpenAI API at `http://elp:8080/v1` with a
service key. Three model names are published:

| Model | Behaviour |
|---|---|
| `Qwen/Qwen3-32B-AWQ` | the raw local model |
| `elp-grounded` | retrieves from your governing documents and appends citations — existing chat UIs get references without knowing anything about RAG |
| `elp-code` | code-tuned sampling for application development |

```python
from openai import OpenAI

client = OpenAI(base_url="http://elp:8080/v1", api_key="elp_...")
answer = client.chat.completions.create(
    model="elp-grounded",
    messages=[{"role": "user", "content": "What is the 600-hour inspection scope?"}],
)
```

Streaming, `/v1/models` and `/v1/embeddings` all work. Every call is
authenticated, scoped and audited on the way through.

For IDE assistants (Continue, Cline, Aider and similar), use the same base URL
with `elp-code`, and set `ELP_DEV__WORKSPACE_ROOTS` to the repositories the
assistant may read. Every requested path is resolved — symlinks included — and
checked against those roots, so a crafted path cannot walk out of the workspace.

---

## 8. Operations

```bash
curl http://elp:8080/health                              # liveness, unauthenticated
curl -H "X-API-Key: $ADMIN" .../v1/health/deep           # every dependency probed
curl -H "X-API-Key: $ADMIN" .../v1/audit?limit=50        # who asked what, and which sources answered
curl -H "X-API-Key: $KEY"   .../v1/whoami                # what the platform believes about a caller
```

`/v1/whoami` is the first thing to check when someone reports *"it says I don't
have access"* — it shows exactly which AD groups arrived on the token and what
they mapped to.

Every answered question records its sources in the audit log, so a past answer
can be reconstructed even after the document is revised. In a maintenance
organisation that is an airworthiness question, not an IT one.

**Backups.** Postgres holds everything that matters — documents, embeddings,
fleet state, compliance history and the audit trail. `pg_dump` covers all of
it. The models are re-downloadable; the LaTeX artifacts are disposable.

**Re-indexing.** Changing the embedding model changes the vector dimension,
which is baked into the schema:

```bash
ELP_INFERENCE__EMBED_DIM=<new> python scripts/reindex.py --confirm --new-dim <new>
```

**GPU reset.** Stop the model servers cleanly (`deploy/stop_models.sh`) — a hard
kill can leave the card holding memory until the driver resets.

---

## 9. What to expect

With the `balanced` profile on one RTX 5090:

- 5–8 concurrent users at 32k context comfortably; more with `throughput`.
- Retrieval (embed + hybrid search + rerank) typically well under a second.
- A cited answer is dominated by generation time, so it scales with answer length.
- Indexing 10–15 large manuals is a one-off job of minutes, not hours.
- First model load after a reboot takes minutes over the eGPU link. This is
  normal and does not affect steady-state speed.

## 10. Known limits

- **One GPU, one large model.** Chat, LaTeX and code share a single generalist
  because 32 GB will not hold a specialist for each. Set
  `ELP_INFERENCE__CODE_BASE_URL` to route code work to a second endpoint if you
  add a card.
- **Scanned PDFs need OCR first.** The parsers extract embedded text; a scanned
  manual with no text layer yields nothing. Run OCR before ingesting.
- **Full-text search uses the `simple` dictionary**, which does not stem. That
  is deliberate for a mixed English/Portuguese corpus, where English stemming
  would hurt more than it helps. Vector search covers morphology.
- **The planner is greedy, not optimal.** Deliberately: its output is meant to
  be predictable and explainable rather than mathematically minimal.
- **Forecasts are only as good as the utilisation data.** An aircraft with no
  recent history falls back to a configured default rate and says so, in the
  API response and in the nightly log.

## 11. Development

```bash
pip install -e ".[dev,ldap,excel]"
pytest tests/ -q        # 75 tests, no database or GPU required
ruff check elp scripts tests
```

The forecasting, planning, chunking and validation logic is written as pure
functions over plain dataclasses precisely so it can be tested without
infrastructure.
