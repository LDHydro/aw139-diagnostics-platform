# Running Guide

A living, task-oriented guide to the Enterprise LLM Platform. `README.md` is the
reference — what everything is and how it is configured. **This** is the guide
you follow in order, from an empty machine to a system your department uses
daily.

> **This document is meant to grow.** As you commission the platform, hit
> problems and add capabilities, append what you learned. Section 7 is a dated
> journal for exactly that. A runbook written by the people who ran it beats
> any document written in advance.

**Contents**

1. [Before you start](#1-before-you-start)
2. [Day 1 — get it running](#2-day-1--get-it-running)
3. [Day 2 — load your documents](#3-day-2--load-your-documents)
4. [Day 3 — load the maintenance programme](#4-day-3--load-the-maintenance-programme)
5. [Day 4 — connect people and applications](#5-day-4--connect-people-and-applications)
6. [Running it day to day](#6-running-it-day-to-day)
7. [Troubleshooting](#7-troubleshooting)
8. [Roadmap and capability backlog](#8-roadmap-and-capability-backlog)
9. [Change journal](#9-change-journal)

---

## 1. Before you start

Collect these before you touch the machine. Chasing them mid-install is what
turns a one-day commissioning into a one-week one.

| Item | Who has it | Needed for |
|---|---|---|
| Entra ID tenant ID, application (client) ID, API scope | IT / identity team | SSO |
| Exact AD group names or object IDs for each role | IT / department head | Access control |
| Governing documents as PDF/DOCX, with revision letters | Quality / tech pubs | The knowledge base |
| Which AD group may read each document | Department head | Document ACLs |
| Standard maintenance programme export (CSV/XLSX) | Planning | Scheduling |
| Current fleet counters (hours, cycles, landings) per tail | Planning / records | Forecasting |
| At least 90 days of utilisation history | Flight ops / records | Forecast accuracy |
| Base URLs + credentials for other internal AI systems | System owners | Federation |

**A note on the utilisation history.** This is the item people skip, and it is
the one that decides whether the forecasts are worth anything. Without it the
platform falls back to a configured default flying rate and says so in every
response. With 90 days of real data it will tell you when an aircraft reaches a
limit to within a few days.

### Hardware check

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.width.current --format=csv
```

You want a 570-series or newer driver and ~32 GB. If `pcie.link.width.current`
comes back as 4 or less, that is your external GPU enclosure — expected, and
only affects how long models take to load.

---

## 2. Day 1 — get it running

### 2.1 Install

```bash
cd enterprise_llm
deploy/install_host.sh
```

The script checks the driver, installs a CUDA 12.8 build of PyTorch, and then
runs a real matrix multiply on the GPU. That last step matters: PyTorch wheels
built for older architectures install without complaint and only fail when the
first user asks a question.

**Checkpoint** — you should see:

```
  device: NVIDIA GeForce RTX 5090  (sm_120)  torch 2.x.x
  wheel supports: ..., sm_120
  matmul on GPU: OK
```

If `sm_120` is missing from the supported list, stop. Nothing downstream will
work. Reinstall from the cu128 index.

### 2.2 Configure

```bash
cp .env.example .env
```

Fill in, at minimum:

```bash
ELP_DATABASE_URL=postgresql+asyncpg://elp:<password>@127.0.0.1:5432/elp
ELP_AUTH__MODE=oidc
ELP_AUTH__OIDC_ISSUER=https://login.microsoftonline.com/<tenant-id>/v2.0
ELP_AUTH__OIDC_CLIENT_ID=<client-id>
ELP_AUTH__OIDC_AUDIENCE=api://<client-id>
ELP_AUTH__GROUP_ROLE_MAP={"<your-admin-group>":"admin"}
```

> **Commissioning shortcut.** If SSO is not ready yet, set
> `ELP_AUTH__MODE=disabled` **and** `ELP_ENVIRONMENT=development` to work
> unauthenticated on a local machine. The gateway refuses to start with
> authentication disabled in production, deliberately. Do not leave it this way.

### 2.3 Database

```bash
echo "POSTGRES_PASSWORD=<password>" >> .env
docker compose up -d postgres
docker compose ps          # wait for "healthy"
```

### 2.4 Models

```bash
deploy/start_models.sh balanced
```

The small servers start first so the large model sizes its KV cache from what
is actually left. **First load takes several minutes** while ~19 GB of weights
cross the eGPU link. This is normal and happens only on a cold start.

**Checkpoint** — at the end you should see roughly 29–30 GB of 32 GB in use.
Substantially less means a server failed; check `logs/chat.log`.

### 2.5 Schema and first credential

```bash
python scripts/bootstrap.py --admin-key-name platform-admin
```

Copy the printed key into your password manager now. It is shown once.

### 2.6 Serve and verify

```bash
uvicorn elp.main:app --host 0.0.0.0 --port 8080
```

In another terminal:

```bash
python scripts/smoke_test.py --api-key elp_...
```

**Checkpoint** — on a fresh system expect `PASS` for liveness, authentication,
models and generation, and `WARN` for the document corpus and retrieval,
because you have not loaded anything yet. Any `FAIL` is worth stopping for.

### 2.7 Make it permanent

```bash
sudo cp -r . /opt/elp
sudo useradd --system --home /opt/elp elp && sudo chown -R elp:elp /opt/elp
sudo cp deploy/systemd/* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now elp-models elp-gateway elp-scheduler.timer
```

---

## 3. Day 2 — load your documents

### 3.1 Decide the access model first

Before uploading anything, write down which AD group may read each document.
Doing this after the fact means a period where everything is readable by
everyone, and you will not reliably remember to go back.

The rule: **an empty `allowed_groups` list means every authenticated user can
read it.** That is the right setting for a company handbook and the wrong one
for anything commercially or personally sensitive.

### 3.2 Write a manifest

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

  - file: maintenance-programme.pdf
    doc_key: AMP-001
    title: Approved Maintenance Programme
    revision: "7"
    effective_date: 2026-03-01

  - file: quality-manual.docx
    doc_key: QM-001
    title: Quality Manual
    revision: "4"
    allowed_groups: [AW139-Maint-Admins]     # narrower than the default
```

Keep this file in version control. It is the record of what is indexed, at
which revision, and who may read it — which is exactly what an auditor asks for.

### 3.3 Ingest

```bash
python scripts/ingest_docs.py --manifest docs/manifest.yaml
```

**Checkpoint** — a large manual should produce hundreds to a few thousand
chunks. A 400-page PDF that yields 12 chunks is a scanned document with no text
layer. Run OCR on it and re-ingest; nothing downstream can read an image.

### 3.4 Test retrieval honestly

Do not test with questions you already know the platform will get right. Write
ten questions your department actually argues about, and check the citations
against the paper:

```bash
curl -s -X POST http://localhost:8080/v1/ask \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"question":"Who may approve deferring a scheduled inspection, and for how long?"}' \
  | python -m json.tool
```

For each answer, check three things:

1. **Is the citation real?** Open the manual at the cited section and page.
2. **Does the answer say what the section says?** Not something adjacent.
3. **Is `confidence` honest?** A confident-sounding answer with confidence
   below 0.3 means retrieval found little and the model wrote around it.

If citations point at the right document but the wrong section, the chunker is
not finding your heading style. Note the document and heading format in the
journal — that is a fixable parsing problem, not a fundamental limit.

---

## 4. Day 3 — load the maintenance programme

### 4.1 Dry run first, always

```bash
curl -X POST http://localhost:8080/v1/maintenance/schedule/import \
  -H "X-API-Key: $KEY" \
  -F file=@maintenance-programme.xlsx \
  -F default_models=AW139 \
  -F source_document_key=AMP-001 \
  -F dry_run=true
```

Read the `issues` array in the response before you commit anything. Every
rejected row is a limit that would not be tracked. The two common causes:

- **"no interval on any basis"** — the row has no hours, cycles, landings or
  calendar interval the importer could find. Usually a column name it does not
  recognise; check `unmapped_columns` in the response.
- **"duplicate task code"** — two rows share a task code. Decide which is right
  before importing, not after.

Re-run with `dry_run=false` once the issues list is empty or understood.

### 4.2 Load the fleet and its history

Create each aircraft with its current counters, then backfill utilisation. The
importer accepts up to 2000 days per call:

```bash
curl -X POST http://localhost:8080/v1/maintenance/utilization \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '[{"tail_number":"PP-ABC","day":"2026-05-01","flight_hours":3.2,"cycles":5,"landings":7},
       {"tail_number":"PP-ABC","day":"2026-05-02","flight_hours":0,"cycles":0,"landings":0}]'
```

**Include the zero days.** An aircraft that sat on the ground on Sunday needs a
row saying so, or the forecast will assume it flies seven days a week. If your
source system only emits flying days, the platform fills the gaps with zeros
automatically — but only *between* the first and last row it sees, so make sure
the range covers the full window.

### 4.3 Sanity-check the forecast against a planner

```bash
curl -s "http://localhost:8080/v1/maintenance/aircraft/PP-ABC/forecast" \
  -H "X-API-Key: $KEY" | python -m json.tool
```

Show the output to whoever currently does the planning by hand and ask one
question: *"is anything here wrong?"*

The `utilization.description` field tells you whether to trust it at all:

- `source=history, confidence 0.85` — good, act on it.
- `source=history, confidence 0.35` — flying is erratic; treat dates as
  approximate and re-check weekly.
- `source=default` — **no usable history.** These dates are arithmetic on a
  guess. Fix the data before anyone plans against them.

### 4.4 First plan

```bash
curl -X POST http://localhost:8080/v1/maintenance/plan \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"horizon_days":120,"explain":true}'      # note: commit is false
```

Read `rationale` on each event and the fleet-level `warnings`. Only add
`"commit": true` once the plan looks like something you would actually do.

### 4.5 Load the MEL

Do this twice, deliberately — the catalogue and the document serve different
purposes:

```bash
# The catalogue: drives dispatch decisions, intervals and expiry tracking.
curl -X POST http://localhost:8080/v1/mel/import \
  -H "X-API-Key: $KEY" \
  -F file=@mel-rev6.xlsx \
  -F source_document_key=MEL-001 -F source_revision=6 \
  -F default_models=AW139 -F dry_run=true
```

```yaml
# The document: lets /v1/mel/check resolve a free-text defect description
# to candidate item numbers. Add to docs/manifest.yaml:
  - file: mel-rev6.pdf
    doc_key: MEL-001
    doc_type: mel                      # required for description lookup
    title: Minimum Equipment List
    revision: "6"
```

Read the dry-run response before committing. Three fields matter:

- **`issues`** — every rejected row. A row whose category could not be read is
  rejected rather than guessed, because inventing a category puts a wrong
  airworthiness limit into the system.
- **`category_a_parsed`** — Category A items whose interval was read out of
  the remarks prose. **Check every one of these against the paper MEL.** The
  parser is good, but a Category A interval is item-specific and the remarks
  column is free text.
- **`category_a_unresolved`** — Category A items where no day-based interval
  could be read at all, usually because the limit is stated in flight hours.
  These need the interval entered manually; the platform will not convert
  hours to days, because that would mean assuming a utilisation rate for an
  airworthiness limit.

**Checkpoint** — spot-check three items against the paper MEL:

```bash
curl -s "http://localhost:8080/v1/mel/check" -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"tail_number":"PP-ABC","item_number":"24-11-01"}' | python -m json.tool
```

Verify the category, the expiry date, the installed/required counts and the
(o)/(m) procedures. Pay particular attention to the expiry: the interval
excludes the day of discovery, so a Category C item found on the 1st should
come back expiring on the **11th**, not the 10th.

### 4.6 Backfill anything currently deferred

Whatever is carried on the fleet today needs to be in the system, or the
dispatch status will be wrong from day one:

```bash
curl -X POST http://localhost:8080/v1/mel/deferrals \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"tail_number":"PP-ABC","item_number":"24-11-01",
       "defect_description":"Gen 2 low voltage",
       "discovered_on":"2026-08-20",
       "accepted_by":"J. Silva, LIC 12345",
       "placard_fitted":true,
       "operational_procedure_applied":true,
       "maintenance_procedure_applied":true}'
```

Note that the platform **refuses** the deferral if the MEL does not permit it,
or if the conditions have not been confirmed. If a backfill is rejected, that
is worth investigating rather than working around — it means what is on the
aircraft does not match what the MEL allows.

### 4.7 Connect NAMIS and commission reporting

**Before anything else, get a read-only NAMIS account.** Every other control
in the reporting path is there to catch mistakes before they reach the
database; this is the one that actually holds. Ask your DBA for an account
with `SELECT` and nothing else, scoped to the tables operations needs.

```bash
# /etc/elp/namis.env  — root-owned, chmod 600, NOT in .env
NAMIS_PASSWORD=<the read-only account password>
```

```bash
# Host prerequisite: NAMIS is SQL Server, so the box needs the Microsoft
# ODBC Driver 18 and unixODBC. Then:
pip install -e ".[mssql]"
```

```bash
# .env
ELP_REPORTS__NAMIS_KIND=sql
ELP_REPORTS__SQL_DIALECT=mssql
ELP_REPORTS__NAMIS_DSN=mssql+aioodbc://elp_reports_ro:${PASSWORD}@24pwnmsap001.nts.ops:1433/NAMISNNSS?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes
ELP_REPORTS__CATALOG_PATH=/opt/elp/config/namis-catalog.json
ELP_REPORTS__ALLOWED_TABLES=[]        # empty: the catalogue is the allowlist
```

Two things to get right before this works:

- **A dedicated read-only login, not Integrated Security.** The report
  generator's "run as yourself" model is correct for a desktop tool and
  impossible for a scheduled job — a 03:00 report has no signed-in user. Ask
  your DBA for `elp_reports_ro` with `db_datareader` and nothing else.
- **`TrustServerCertificate=yes`** is needed against the on-prem self-signed
  certificate, matching what the production ops apps already use. Drop it
  once a trusted certificate is installed.

**Bring the existing reports over.** Copy
`%LOCALAPPDATA%\NamisReports\saved-reports\*.json` off the machine that has
them and import, without saving, to see what maps:

```bash
curl -X POST "http://localhost:8080/v1/reports/import" -H "X-API-Key: $KEY" \
  -F files=@open-work-requests.json | python -m json.tool
```

Check `compiles`, `unmapped` and `sql_preview` on each. Then re-run with
`?save=true&allowed_groups=...`. They arrive as drafts and still need
approval before they can be scheduled.

**Deploy the catalogue.** Export it from the report generator
(`build-windows.sh` regenerates it) and copy it to
`/opt/elp/config/namis-catalog.json`. It is deliberately not in the
repository. Confirm it loaded:

```bash
journalctl -u elp-gateway | grep "NAMIS catalogue"
# loaded NAMIS catalogue: 585 tables, 8231 columns, 723 relationships, 22 groups
```

Without it the platform falls back to live introspection, which recovers
column names but **not the join relationships** — and a compound key joined
on one column instead of two produces a cartesian product that reads as real
data.

**Checkpoint 1 — prove the account cannot write.** Do this before anyone
saves a report. The platform attempts a write and expects NAMIS to refuse:

```bash
curl -s .../v1/health/deep -H "X-API-Key: $ADMIN_KEY" \
  | python -c "import json,sys; print(json.load(sys.stdin)['components']['namis'])"
```

- `'read_only': True` — correct, carry on.
- `'read_only': False` — **stop.** The account can write to production. No
  amount of application-level validation makes that safe. Get the grant
  fixed before going further.
- `'read_only': None` — the probe could not run (a non-PostgreSQL NAMIS, or
  the connection failed). Confirm the grant manually with your DBA.

**Checkpoint 2** — confirm the platform can see the schema:

```bash
curl -s .../v1/reports/schema -H "X-API-Key: $KEY" | python -m json.tool | head -40
```

You should get real tables and real column names. This introspection is what
grounds query authoring; without it the model invents plausible-looking
columns, which is the single largest source of wrong-looking reports.

Then prove the whole path with a throwaway request:

```bash
curl -X POST .../v1/reports/ask -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"request_text":"count of work orders opened in the last 7 days"}'
```

Check three things in the response:

- **`query`** — read it. Does it use the columns you would have used?
- **`assumptions`** — this is where the model says what it had to guess.
  Usually the difference between the report you wanted and the one you asked
  for.
- **`run.row_count`** — does the number match what someone in operations
  would expect? If it does not, the query is wrong, not the database.

### 4.8 Save and schedule the first real report

```
draft  →  review  →  save  →  approve  →  schedule
```

Do not shortcut this. An ad-hoc run happens with a person watching; a
scheduled run happens at 03:00 with nobody watching, and the approval step is
the only thing standing between those two situations.

```bash
# 1. Draft — nothing is saved or run
curl -X POST .../v1/reports/draft -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"request_text":"work orders still open after 14 days, by tail and station"}'

# 2. Save the REVIEWED query (edit it first if the draft was not quite right)
curl -X POST .../v1/reports -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"name":"Ageing work orders",
       "request_text":"work orders still open after 14 days, by tail and station",
       "query":"<the SQL you reviewed>",
       "output_formats":["markdown","csv","pdf"],
       "allowed_groups":["AW139-Planning","AW139-Maint-Admins"]}'

# 3. Approve — requires reports:approve, and means "I have read this query"
curl -X POST ".../v1/reports/Ageing%20work%20orders/approve" -H "X-API-Key: $APPROVER_KEY"

# 4. Schedule
curl -X PUT ".../v1/reports/Ageing%20work%20orders/schedule" -H "X-API-Key: $APPROVER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"cron":"0 6 * * 1-5","timezone":"America/Sao_Paulo","enabled":true}'
```

Then enable the timer and verify what it thinks is due:

```bash
sudo systemctl enable --now elp-reports.timer
python scripts/run_scheduled_reports.py --dry-run
curl -s .../v1/reports/status/scheduled -H "X-API-Key: $KEY" | python -m json.tool
```

**Set `allowed_groups` on every saved report.** Report output is frequently
more sensitive than the underlying records — a table of everything overdue,
by station, is exactly the summary you would not want circulating freely.

---

## 5. Day 4 — connect people and applications

### 5.1 Prove the SSO path

Have one real person from each role sign in and call:

```bash
curl http://localhost:8080/v1/whoami -H "Authorization: Bearer $USER_TOKEN"
```

`groups` shows exactly what arrived on the token, and `roles` shows what it
mapped to. Almost every "it says I don't have access" ticket is answered here:
either the group is absent from the token (an Entra token-configuration
problem) or present but not in `ELP_AUTH__GROUP_ROLE_MAP` (a typo — the names
must match, though case does not).

### 5.2 Issue one service key per application

One key per application, never a shared key. When an app is decommissioned or
compromised you revoke exactly one thing, and the audit log tells you which app
asked what.

```bash
curl -X POST http://localhost:8080/v1/api-keys \
  -H "X-API-Key: $ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"aw139-diagnostics-ui",
       "scopes":["ask","chat","docs:read","maint:read"],
       "groups":["AW139-Engineering"]}'
```

Grant the narrowest scopes that work. An app that only answers questions has no
business holding `docs:write`.

### 5.3 Point an existing app at it

Anything that speaks the OpenAI API needs only a base URL change:

```python
from openai import OpenAI
client = OpenAI(base_url="http://elp.corp.internal:8080/v1", api_key="elp_...")

response = client.chat.completions.create(
    model="elp-grounded",                       # retrieves and cites
    messages=[{"role": "user", "content": "What is the 600-hour inspection scope?"}],
)
```

### 5.4 Register your other AI systems

Add them to `config/peers.yaml`, put the credential in the environment variable
the entry names, and restart the gateway. Then confirm:

```bash
curl http://localhost:8080/v1/health/deep -H "X-API-Key: $ADMIN_KEY" | python -m json.tool
```

Every peer should report `status: ok`. A peer that is down does not break
answers — it is skipped and noted in `warnings` — but you want to know.

---

## 6. Running it day to day

### Daily

Nothing. The `elp-scheduler` timer re-forecasts the fleet at 03:30 and exits
non-zero if any aircraft is past a hard limit, so a systemd failure alert means
something real. Wire that alert into wherever your team already looks.

```bash
systemctl status elp-scheduler          # or: journalctl -u elp-scheduler -n 50
```

### Weekly

```bash
python scripts/smoke_test.py --api-key $ADMIN_KEY
curl -s .../v1/audit?limit=200 -H "X-API-Key: $ADMIN_KEY" | python -m json.tool
```

In the audit log, look for `"outcome": "ungrounded"` — questions the platform
answered without citing anything. A cluster of those on one topic means a
document is missing from the corpus. That is your best signal for what to
index next.

### When a document is revised

Add the new revision to the manifest with the new revision letter and re-run
the ingest. The platform marks the older revision superseded automatically and
keeps it, so an answer given last month can still be reconstructed from the
revision that was current then. Never overwrite a revision in place.

### When someone joins or changes role

Nothing to do here — change their AD group. That is the point of mapping groups
to roles rather than managing users in the platform.

### Backups

```bash
pg_dump -Fc elp > elp-$(date +%F).dump
```

Postgres holds everything that matters: documents, embeddings, fleet state,
compliance history, the audit trail. Models are re-downloadable and LaTeX
artifacts are disposable.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `no kernel image is available for execution` | PyTorch built for an older architecture | Reinstall from the cu128 index; re-run `install_host.sh` |
| Chat server dies at startup, embeddings fine | Started in the wrong order — no VRAM left | `deploy/stop_models.sh` then `start_models.sh`; never start them by hand individually |
| Model load takes 10+ minutes | Weights crossing the eGPU link on a cold start | Normal on first load. If it happens every restart, check `HF_HOME` is on local disk |
| `401 authentication required` with a valid token | Audience mismatch | Check `ELP_AUTH__OIDC_AUDIENCE` matches the `aud` claim in the token |
| `403 ... lacks the required permission` | Group not in the role map, or absent from the token | `GET /v1/whoami` — compare `groups` against `ELP_AUTH__GROUP_ROLE_MAP` |
| A user sees no documents at all | Their groups are in no document's `allowed_groups` | `GET /v1/documents` as them, then `PATCH /v1/documents/{id}/access` |
| Answers cite the right document, wrong section | Chunker is not recognising that document's heading style | Note the format in the journal; the heading detection in `elp/rag/chunker.py` is regex-driven and extensible |
| A 400-page PDF produced ~10 chunks | Scanned document, no text layer | OCR it, then re-ingest with `--force` |
| Every answer has low confidence | Reranker down, or the corpus does not cover the topic | `GET /v1/health/deep`; check the reranker |
| Forecast dates look wrong | `source=default` — no utilisation history | Load history. Everything else is downstream of this |
| Forecast shifts every time it runs | Genuinely erratic flying | Check `confidence`; if low, the spread is real, not a bug |
| A cancelled visit's tasks vanished | They did not — they are re-planned | Check `replacement_event_ids` in the cancel response, and `GET /v1/maintenance/events` |
| LaTeX fails with "rejected: uses ..." | The sandbox blocked a shell-escape primitive | Working as intended. Rewrite without `\write18`/`\directlua`/absolute-path `\input` |
| GPU shows memory used with no servers running | Hard-killed vLLM | `deploy/stop_models.sh`; if it persists, `nvidia-smi --gpu-reset` |
| MEL check returns `not_in_mel` for something you know is listed | Item not imported, or a different item number | `GET /v1/mel/items?q=<text>`; check the import `issues` array |
| MEL expiry looks a day early or late | It is not — the interval excludes the day of discovery | Cat C found on the 1st expires end of the 11th. Verify against the MMEL preamble |
| `deferral cannot be recorded` on backfill | The MEL does not permit what is on the aircraft | Investigate rather than work around; the aircraft may be carrying invalid relief |
| Category A item has no expiry | Interval stated in flight hours or cycles, not days | Enter the day limit manually; the platform will not convert |
| Extension refused | Category A, already extended, or already expired | All three are correct refusals — an extension is granted before the limit, not after |
| `NAMIS is not configured` | `ELP_REPORTS__NAMIS_KIND` still `disabled` | Set it to `sql` or `rest` and supply the DSN |
| `read_only: false` in health/deep | The reporting account can write to production | Stop and fix the grant. `REVOKE` everything but `SELECT` |
| `read_only: null` | Probe could not run (non-PostgreSQL, or connection failed) | Verify the grant manually with your DBA |
| `/v1/reports/schema` returns no tables | The account cannot see them, or the allowlist excludes them | Check the account's grants first, then `ELP_REPORTS__ALLOWED_TABLES` |
| Draft query rejected as unsafe | Working as intended — read the rejection | It names the exact construct; rephrase the request or widen the allowlist |
| Report numbers look wrong | Almost always the query, not the database | Read `assumptions` on the draft; re-draft with the ambiguity spelled out |
| Scheduled report stopped firing | Someone edited the query, revoking approval | `GET /v1/reports/{name}` → `approval_current: false`. Re-approve |
| Run status `blocked` | Same cause: query no longer matches its approval | Re-approve after reviewing the change |
| Report ran but the PDF is missing | LaTeX failed; other formats still wrote | Check `warnings` on the run — the CSV and Markdown are unaffected |
| Scheduled run late by minutes | The timer checks every 10 minutes | Expected. Missed runs are picked up within a 90-minute grace window |

---

## 8. Roadmap and capability backlog

Ordered by value against effort. The first group reuses tables and logic that
already exist in this repository, so they are integration work rather than new
systems.

### Tier 1 — plugs into what is already built

**8.1 MEL / CDL dispatch decision support — DELIVERED 2026-08-25**

*"Can we dispatch with this inoperative, and for how long?"*

Built as `elp/mel/` with its own catalogue and deferral tables rather than
reusing task cards, because MEL relief has rules scheduled maintenance does
not: quantity relief (installed vs required for dispatch), interactions
between simultaneously inoperative items, (o)/(m) procedures that must be
carried out before the deferral is valid, and prohibited operations.

See §4.5 to commission it and README §5a for the API. The three rules it
enforces — unlisted means no-go, the interval excludes the day of discovery,
and relief is per quantity — are each a place real operations go wrong.

**8.2 AD and Service Bulletin applicability triage**

You already have `serialEffectivity`, `partEffectivity`, `ipdEffectivityCodes`
and `resolveConfiguration()` in `server/configuration-resolver.ts`. A new AD
arrives; the platform determines which tails it applies to by serial and
configuration, extracts the compliance deadline, generates task cards and
pushes them into the scheduler — all from components that exist. Currently a
manual, error-prone job with real regulatory exposure.

**8.3 Reliability reporting**

`ata-analytics.ts` already computes MTBF, MTTR, recurrence and trend from
`ataOccurrences` and `partReplacements`. Point the LaTeX service at it and the
monthly reliability report your maintenance programme requires generates
itself, with the numbers traceable to the records that produced them.

**8.4 Parts demand forecasting from the plan**

The planner knows every task due in the next 120 days and each task's
`required_parts`. Project that into parts demand, compare against stock via
`smart-stock.ts`, and flag long-lead items before they become AOG. Turns a
reactive stock function into a forward-looking one.

**8.4b Operational reporting from NAMIS — DELIVERED 2026-08-25**

Plain-language report requests, answered by a guarded read-only query against
NAMIS, saved and schedulable. See §4.7-4.8 to commission it and README §5b
for the API and the safety model.

### Tier 2 — modest new build

**8.5 Work order and defect write-up assistant**

A technician types a rough description, in Portuguese or English, and gets back
a compliant corrective-action entry with the correct manual reference, ATA code
and part numbers. Sloppy logbook entries are a routine audit finding, and this
addresses the cause rather than the symptom. `bge-m3` is multilingual, so the
mixed-language input costs nothing extra.

**8.6 Technical publication revision impact analysis**

A new AMM revision arrives. Which task cards, work instructions and MEL items
does it affect? The platform already stores revisions and supersession, so it
can diff the section structure between two revisions and report what changed
and what depends on it. This is a job people currently do by reading
side-by-side, badly.

**8.7 Shift handover brief**

Generate the handover automatically from open work orders, deferred items, AOG
status and tomorrow's plan. Small, and it removes a daily chore that is done
inconsistently at exactly the moment attention is lowest.

**8.8 Regulatory gap analysis**

Compare your MOE and Quality Manual against ANAC RBAC 145 (or EASA Part-145 /
FAA Part 145) requirement by requirement, and produce an audit-ready gap report
in LaTeX. Best run before an audit rather than during one.

### Tier 3 — larger, still worthwhile

**8.9 Case-based troubleshooting**

`troubleshootingHistory` and `maintenanceLogs` already hold what was wrong and
what fixed it. Given a pilot squawk, return ranked probable causes with manual
references *and* what actually resolved it last time on this fleet. Your
existing CrewAI diagnostic system is the natural home; this platform supplies
the retrieval and citation layer.

**8.10 Technician authorisation matching**

Task cards already carry `required_skills`. Hold technician authorisations
against them and the planner can flag "nobody rostered on that shift is
authorised for this task" *before* the aircraft is in the hangar.

**8.11 SMS occurrence triage**

Classify hazard and occurrence reports, link each to prior similar events, and
suggest a risk index. Note this touches safety-reporting confidentiality — the
document ACL model handles it, but agree the access rules with your safety
manager before building.

### Deliberately not recommended

- **Anything that signs off maintenance.** The platform drafts, cites and
  forecasts. A licensed engineer signs. Do not blur this line, and do not build
  a workflow that makes it easy to blur.
- **Flight-crew-facing dispatch advice.** MEL support for the maintenance
  department is one thing; an interface that reads to a pilot as authority to
  dispatch is another, and is not what this is.
- **Automatic deferral approval.** The approval permission exists precisely so
  a named person makes that call.

---

## 9. Change journal

Append an entry whenever you commission something, hit a problem worth
remembering, or change how the platform is used. Newest at the top.

<!-- Template:
### YYYY-MM-DD — <what happened>
**Who:** name
**What changed:**
**Why:**
**Watch out for:**
-->

### 2026-08-26 — Ingest pipeline validated against the real manuals

**Who:** initial build, run against AOM Rev 4, ATM Rev 4 and SOP Rev 1
**What changed:** Running three real governing manuals through the pipeline
found four defects that synthetic tests had not. All are general, not
specific to these documents.

| Defect | Effect | Fix |
|---|---|---|
| Whitespace-only spans were skipped | `Sectiontitlechangedfrom` — words glued together in LaTeX-produced PDFs, corrupting both the embedding and the quoted text | keep space spans, use them only for text |
| Contents pages indexed as content | retrieving a page number instead of the clause, plus a near-duplicate of every real heading | detect navigation pages and dot leaders |
| Running headers embedded on every page | page furniture embedded hundreds of times, competing with content | strip text repeating on ≥50% of pages |
| Lead-in labels treated as headings | `Objective:` `References:` shredded each lesson into one-line fragments | a styled line ending in a colon is a label |

Also added: revision-history rows no longer reassign the section (a row
reading *"3.7 Changed: ..."* names the section it refers to, not the section
the text is in — attributing following text to §3.7 is a wrong citation,
which is worse than a missing one), and undersized fragments sharing a
section are merged.

**Measured effect on the ATM:** 495 chunks → 283, fragments under 25 tokens
194 → 58, median chunk 36 → 104 tokens. A training lesson now stays in one
retrievable piece with its own heading instead of eight fragments.

**Current state of the three manuals:**

| | pages | chunks | median tokens | with §number | citable |
|---|---:|---:|---:|---:|---:|
| AOM | 157 | 425 | 136 | 84% | 100% |
| ATM | 184 | 283 | 104 | 31% | 100% |
| SOP | 143 | 404 | 43 | 32% | 100% |

**Watch out for:**
- **Every chunk is citable** — by section number or by heading — but only the
  AOM is mostly section-numbered. ATM and SOP citations will more often read
  *"ATM-001 Rev 4, Lesson 1: Ground — King Air B350"* than *"§4.2"*. That is
  accurate, just less precise.
- **The SOP is graphics-heavy**: 261 embedded images and about half the AOM's
  text density. Text pages index fine; pages that are purely a diagram or a
  screenshot have no text layer and will not be retrievable until OCR'd.
  Roughly 4% of its pages are image-dominant.
- The ATM and SOP are **Final Candidate** drafts. Set a real effective date
  and re-ingest once approved, so citations show the approved revision.
- `config/docs-manifest-example.yaml` is ready to run; set the access groups
  before you do.

### 2026-08-25 — The model now plans definitions, not SQL

**Who:** initial build, following the generator's proven design
**What changed:** `/v1/reports/draft` and `/ask` plan a report definition and
compile it here, rather than asking the model for SQL. Free-form SQL is now
the fallback, not the default.

`mode` selects the path — `auto` (default), `structured`, `sql`. In `auto`,
a request the definition model cannot express falls back to SQL and the
response says so, with a warning that it needs a closer read before approval.

**Why this is better than validating SQL.** The compiler resolves every
identifier through the catalogue, so a hallucinated column raises before any
SQL exists. Validating free-form SQL means proving a negative about a string
the model wrote. The rejections are also good repair signals — a model told
*"'WorkRequest' has no column 'Status'. Did you mean: StatusCd?"* fixes it on
the next attempt, which a generic "invalid query" never would.

**Watch out for:**
- **Read the `assumptions` before approving.** That is where the model says
  what it had to guess, and it is usually the difference between the report
  you wanted and the one you asked for.
- A draft that used an **INFERRED** join should say so in its assumptions —
  224 of the 723 catalogue relationships are inferred rather than declared
  foreign keys.
- `fell_back_to_sql: true` in a draft response means the structured path did
  not compile. That is a signal to read the SQL closely, not a routine
  outcome.
- Structured drafts still arrive as **drafts** and still need approval before
  they can be scheduled.

### 2026-08-25 — Note on the credential in the handoff notes

**Who:** noted while reading the NAMIS handoff notes, then corrected
**Status:** no action — confirmed by the owner as an inactive placeholder.

The generator's `HANDOFF.md` shows an application-role name and password in
plaintext. It reads like a live secret; it is not — it is a placeholder and
the account is inactive. Recorded here only so the next person to read that
document does not raise it again.

Two things remain true and separate from it:

- **Finding F-1 is real.** The app-role credential compiled into the shipped
  `NamisMaint` client binaries is recoverable with `strings`, and rotating it
  is still on their security list.
- **This platform never needs that credential.** The reporting service uses
  its own read-only login. The application role is only a fallback for a site
  where that login cannot be granted SELECT, and even then the secret stays
  in a root-owned file on the server.

### 2026-08-25 — Handoff notes reconciled: three fixes

**Who:** initial build, from the generator's handoff notes
**What changed:**
- **Row counts removed as a table-selection signal.** The handoff says the
  schema export came from a **test instance**, so volumes are not
  representative and "don't prune tables by row count". Scoring on them
  would have buried the busiest production tables behind whatever happened
  to hold rows in test.
- **Inferred joins are now labelled for the model.** 224 of the 723
  relationships were inferred from name/type matching rather than read from
  a foreign key. They are rendered separately with an instruction to declare
  reliance on one in ASSUMPTIONS.
- **`resolveCodes` added to the importer.** The handoff confirms the real
  `ReportDefinition` field names — `table`, `joins`, `fields`, `filters`,
  `sorts`, `filterLogic`, `rowLimit`, `resolveCodes` — and `resolveCodes`
  was the one spelling the importer did not know.

**Confirmed against the live catalogue:** 585 tables (NAMISNNSS 318,
AMO_NASAWeb 163, OpsDBAMO 67, WebSupport_NASAWeb 37), 8,231 columns, 723
relationships (499 real FKs + 224 inferred). Every figure matches.

**Convergent design worth noting.** Their AI planner returns a *report
definition, never SQL*, validated by the same builder as any other input —
"the AI is an untrusted suggester". This platform's structured compiler
arrived at the same conclusion independently, which is reassuring about
both.

**Still open on their side, and relevant here:**
- Views and functions are not in the catalogue yet. Reports needing them
  will fail the allowlist until the catalogue is regenerated with
  `tools/export-views-and-functions.sql`.
- GUID foreign keys resolve only through a live join, not the value-list
  snapshot.
- Ops-side inferred relationships are conservative; some tables need custom
  joins.

### 2026-08-25 — Structured reports, saved-report import, Excel export

**Who:** initial build, from the NAMIS generator's user guide
**What changed:**
- **Structured report definitions** (`elp/reports/structured.py`), compiled
  to parameterised T-SQL. Same model as the existing generator — base table,
  fields, joins, filters, sort, limit — and it has **no injection surface**:
  identifiers are resolved through the catalogue and re-emitted from
  catalogue metadata, values are always bound parameters.
- **Saved-report import** (`POST /v1/reports/import`). Existing reports carry
  over instead of being re-authored, and are compiled against the catalogue
  on import.
- **Excel export** with real types, plus a 12-column cap on PDF.

**Watch out for:**
- **The saved-report key mapping is inferred, not confirmed** — written from
  the user guide, not a sample file. Import one real export with
  `save=false` and read the `unmapped` list. Anything there is a key the
  importer does not know; send it back and it becomes a one-line fix.
- Imported reports arrive as **drafts**. They still need approval before
  they can be scheduled — the same rule as any other report.
- The compiler refuses joins between columns of different kinds. Where the
  catalogue does not state a `kind` it is now inferred from the SQL type, so
  a stale catalogue export will be stricter than a fresh one.
- Structured reports are compiled at **run** time, not stored as SQL. A
  catalogue change that breaks a report surfaces as a blocked run with a
  clear reason, rather than a query against a column that no longer exists.

### 2026-08-25 — NAMIS reporting reconciled with the real system

**Who:** initial build, from the NAMIS assessment pack
**What changed:** The reporting subsystem now targets NAMIS as it actually
is. Three defects fixed and one capability added:

- **`LIMIT` is a syntax error in T-SQL.** Row limiting now emits `TOP (n)`,
  injected into the outermost SELECT (correct for CTEs too). Every generated
  query would have failed at the server before this.
- **Bracket-quoted identifiers were invisible to the guard.**
  `[dbo].[WorkRequest]` matched no table pattern, so every bracket-quoted
  reference bypassed the allowlist unchecked.
- **System databases slipped through three-part names.** In
  `master.dbo.sysdatabases` the dangerous qualifier is the first one; only
  the last was being checked.
- **The field catalogue is now loaded and used** — 585 tables, 8,231 columns,
  723 relationships. It grounds query authoring and serves as the allowlist.

Also added: `sp_setapprole` support, SQL Server isolation-level control, a
permissions-based read-only probe, and SQL error translation copied from what
the report generator learned in this environment (18456, 229, 15151, 208).

**Watch out for:**
- **"Run as yourself" cannot work here.** A scheduled report has no signed-in
  user. Use a dedicated read-only SQL login; this is also what makes the
  read-only guarantee real.
- The catalogue is **not committed** — deploy it to
  `/opt/elp/config/namis-catalog.json`. Without it, joins are guessed.
- Reporting queries take shared locks on a live OLTP system.
  `ELP_REPORTS__MSSQL_ISOLATION_LEVEL` is the lever; ask the DBA whether
  SNAPSHOT is enabled before reaching for READ UNCOMMITTED.
- Security findings F-1/F-2 remain **live issues in the current desktop
  deployment** and are not fixed by anything here. Rotate the app-role
  password and set `Encrypt=true` independently.

### 2026-08-25 — Open item: port the existing report builder

**Who:** noted from operations
**Status:** OPEN — waiting on a port from another Claude Code session that is
not reachable from here.

**What is known:** NAMIS is a database, so the SQL adapter
(`ELP_REPORTS__NAMIS_KIND=sql`) is the correct path and the REST adapter can
be ignored. An existing report builder lives in a separate session and will
be brought over on a laptop.

**Where it plugs in.** The reporting pipeline has four seams, and the
existing builder probably replaces one or two of them rather than all four:

| Seam | Module | Replace if the existing builder... |
|---|---|---|
| Query authoring | `elp/reports/authoring.py` | already has prompts or query templates that work against NAMIS |
| Query execution | `elp/reports/datasource.py` | **do not replace** — the read-only guarantees live here |
| Rendering | `elp/reports/render.py` | has its own layouts, templates or house style |
| Saving + scheduling | `elp/reports/service.py`, `runner.py` | has its own definition format worth keeping |

**Worth bringing over with it, to reconcile quickly:**
- Any existing NAMIS table/column knowledge or query templates — this is the
  most valuable part and the hardest to reconstruct.
- Its report definition format, if it has one, so saved reports can be
  migrated rather than re-authored.
- Any house style for report layout.
- Whether it assumed a writable connection anywhere. If so, that needs
  unpicking before it runs here: this platform's NAMIS path is read-only by
  four layers and a test asserts no `commit()` exists in it.

**Do not port over:** anything that writes to NAMIS, and any embedded
credentials — connection secrets belong in `/etc/elp/namis.env`, referenced
by name only.

### 2026-08-25 — NAMIS operational reporting added

**Who:** initial build
**What changed:** New `elp/reports/` package and twelve `/v1/reports/*`
endpoints. Operations requests a report in plain language; the platform
authors a read-only query against NAMIS grounded in the introspected schema,
runs it, narrates the result and renders Markdown/CSV/HTML/JSON/PDF. Reports
can be saved, approved and scheduled; the `elp-reports` timer runs them. 98
new tests.
**Why:** Replaces a manual extract-and-format cycle, and puts the recurring
reports on a timer instead of a person's calendar reminder.
**Watch out for:**
- **The read-only NAMIS account is the control that matters.** The SQL guard,
  the read-only connection and the statement timeout are defence in depth.
  Do not treat them as a substitute for provisioning the account correctly.
  `/v1/health/deep` proves it by attempting a write; check it at
  commissioning and after any NAMIS credential change.
- Approval binds to a fingerprint of the query text. Editing a scheduled
  report's query revokes approval and stops the schedule — by design. If a
  report "stopped arriving", check `approval_current` first.
- Scheduled runs re-execute the **approved** query and never re-generate it.
  A report whose shape changed between runs would be untraceable.
- Set `allowed_groups` on every saved report. Report output is often more
  sensitive than the records behind it.
- `ELP_REPORTS__ALLOWED_TABLES` is empty by default, which permits any table
  the account can see. Tighten it once you know which tables operations
  actually uses.

### 2026-08-25 — MEL dispatch decision support added

**Who:** initial build
**What changed:** New `elp/mel/` package and nine `/v1/mel/*` endpoints:
catalogue import, dispatch check, deferral lifecycle (raise, clear, one-time
extension) and fleet dispatch status. The nightly job now expires overdue
deferrals and exits non-zero if any aircraft is undispatchable. 51 new tests.
**Why:** Highest-frequency decision in the department, and the one where an
off-by-one or a missed quantity limit has direct airworthiness consequences.
**Watch out for:**
- Category A intervals are parsed out of free-text remarks. The import
  response lists every one it read (`category_a_parsed`) and every one it
  could not (`category_a_unresolved`). **Check them against the paper MEL
  before anyone dispatches against them.**
- Intervals stated in flight hours or cycles are deliberately *not* converted
  to days. Enter those limits manually.
- The MEL must be loaded twice — as a catalogue (drives decisions) and as an
  indexed document with `doc_type: mel` (enables description lookup). Loading
  only the document gives you search but no expiry tracking.
- Raising a deferral is refused unless the (o)/(m) procedures and placarding
  are confirmed applied. This is intentional and will generate questions from
  anyone used to recording the deferral first and doing the work after.

### 2026-08-25 — Platform built and pushed

**Who:** initial build
**What changed:** Full platform delivered on branch
`claude/enterprise-llm-deployment-lvdej7`: cited document Q&A, federation with
other internal AI systems, predictive maintenance scheduling, LaTeX, code
assistance, OpenAI-compatible API, AD-group access control. 75 tests passing.
**Why:** Replace scattered manual lookup and spreadsheet-based planning with one
governed, auditable system that keeps everything on-premises.
**Watch out for:** Not yet run against live infrastructure — vLLM on the actual
5090, Postgres/pgvector queries and a real Entra token are all unverified.
`scripts/smoke_test.py` covers those on first deployment. Section 2 checkpoints
exist because that is where problems will surface.
