#!/usr/bin/env python3
"""
End-to-end smoke test against a running deployment.

    python scripts/smoke_test.py --base-url http://127.0.0.1:8080 --api-key elp_...

Checks each subsystem in turn and reports what works, what is degraded and
what is broken. Run it after every deployment and after any GPU reset.
"""

from __future__ import annotations

import argparse
import time

import httpx

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_SYMBOL = {PASS: "\033[32m PASS \033[0m", FAIL: "\033[31m FAIL \033[0m", WARN: "\033[33m WARN \033[0m"}

results: list[tuple[str, str, str]] = []


def report(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"[{_SYMBOL[status]}] {name}" + (f"\n         {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default="", help="Service key, or an SSO bearer token via --token")
    parser.add_argument("--token", default="")
    parser.add_argument("--question", default="What is the policy for deferring scheduled maintenance?")
    parser.add_argument("--tail-number", default="", help="Also exercise the maintenance forecast")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    headers = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    elif args.api_key:
        headers["X-API-Key"] = args.api_key
    else:
        parser.error("supply --api-key or --token")

    base = args.base_url.rstrip("/")
    client = httpx.Client(base_url=base, headers=headers, timeout=args.timeout)

    # --- liveness ---------------------------------------------------
    try:
        r = httpx.get(f"{base}/health", timeout=10)
        body = r.json()
        report(
            "gateway liveness",
            PASS if r.status_code == 200 else FAIL,
            f"status={body.get('status')} auth_mode={body.get('auth_mode')} db={body.get('database')}",
        )
    except Exception as exc:
        report("gateway liveness", FAIL, str(exc))
        print("\nthe gateway is not reachable; stopping here.")
        return 1

    # --- identity ---------------------------------------------------
    try:
        r = client.get("/v1/whoami")
        if r.status_code == 200:
            me = r.json()
            report(
                "authentication",
                PASS,
                f"{me['subject']} roles={me['roles']} scopes={len(me['scopes'])}",
            )
        else:
            report("authentication", FAIL, f"{r.status_code}: {r.text[:200]}")
            return 1
    except Exception as exc:
        report("authentication", FAIL, str(exc))
        return 1

    # --- dependencies ------------------------------------------------
    try:
        r = client.get("/v1/health/deep")
        if r.status_code == 403:
            report("dependency health", WARN, "needs an admin credential; skipped")
        elif r.status_code == 200:
            deep = r.json()
            status = deep.get("status")
            report(
                "dependency health",
                PASS if status == "ok" else WARN,
                f"status={status} degraded={deep.get('degraded')}",
            )
        else:
            report("dependency health", FAIL, f"{r.status_code}: {r.text[:200]}")
    except Exception as exc:
        report("dependency health", FAIL, str(exc))

    # --- corpus -------------------------------------------------------
    document_count = 0
    try:
        r = client.get("/v1/documents/stats")
        if r.status_code == 200:
            stats = r.json()
            document_count = stats.get("documents", 0)
            missing = stats.get("chunks_missing_embeddings", 0)
            report(
                "document corpus",
                PASS if document_count else WARN,
                f"{document_count} documents, {stats.get('chunks')} chunks"
                + (f", {missing} chunks missing embeddings" if missing else ""),
            )
        else:
            report("document corpus", FAIL, f"{r.status_code}: {r.text[:200]}")
    except Exception as exc:
        report("document corpus", FAIL, str(exc))

    # --- retrieval ----------------------------------------------------
    if document_count:
        try:
            started = time.monotonic()
            r = client.post("/v1/search", json={"query": args.question, "top_k": 5, "include_text": False})
            elapsed = (time.monotonic() - started) * 1000
            if r.status_code == 200:
                hits = r.json()
                report(
                    "retrieval",
                    PASS if hits else WARN,
                    f"{len(hits)} passage(s) in {elapsed:.0f} ms"
                    + (f"; top: {hits[0]['citation']}" if hits else ""),
                )
            else:
                report("retrieval", FAIL, f"{r.status_code}: {r.text[:200]}")
        except Exception as exc:
            report("retrieval", FAIL, str(exc))
    else:
        report("retrieval", WARN, "no documents indexed; skipped")

    # --- grounded answer ----------------------------------------------
    try:
        started = time.monotonic()
        r = client.post("/v1/ask", json={"question": args.question, "app": "smoke-test"})
        elapsed = (time.monotonic() - started) * 1000
        if r.status_code == 200:
            answer = r.json()
            report(
                "grounded answer",
                PASS if answer.get("grounded") else WARN,
                f"confidence={answer['confidence']} refs={len(answer['references'])} "
                f"in {elapsed:.0f} ms; {answer['answer'][:110]!r}",
            )
            for warning in answer.get("warnings", []):
                print(f"           note: {warning}")
        else:
            report("grounded answer", FAIL, f"{r.status_code}: {r.text[:300]}")
    except Exception as exc:
        report("grounded answer", FAIL, str(exc))

    # --- OpenAI compatibility ------------------------------------------
    try:
        r = client.get("/v1/models")
        models = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
        report(
            "openai-compatible surface",
            PASS if models else FAIL,
            ", ".join(models) if models else f"{r.status_code}: {r.text[:150]}",
        )
    except Exception as exc:
        report("openai-compatible surface", FAIL, str(exc))

    try:
        started = time.monotonic()
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                  "max_tokens": 16},
        )
        elapsed = (time.monotonic() - started) * 1000
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            report("local model generation", PASS, f"{text[:60]!r} in {elapsed:.0f} ms")
        else:
            report("local model generation", FAIL, f"{r.status_code}: {r.text[:200]}")
    except Exception as exc:
        report("local model generation", FAIL, str(exc))

    # --- federation ------------------------------------------------------
    try:
        r = client.get("/v1/peers")
        if r.status_code == 200:
            peers = r.json()
            report(
                "internal AI federation",
                PASS if peers else WARN,
                f"{len(peers)} peer(s) visible" if peers else "no peers registered",
            )
        elif r.status_code == 403:
            report("internal AI federation", WARN, "credential lacks federation:query")
        else:
            report("internal AI federation", FAIL, f"{r.status_code}: {r.text[:150]}")
    except Exception as exc:
        report("internal AI federation", FAIL, str(exc))

    # --- maintenance ------------------------------------------------------
    try:
        r = client.get("/v1/maintenance/fleet")
        if r.status_code == 200:
            fleet = r.json()
            report("maintenance fleet", PASS if fleet else WARN, f"{len(fleet)} aircraft")
            tail = args.tail_number or (fleet[0]["tail_number"] if fleet else "")
            if tail:
                r = client.get(f"/v1/maintenance/aircraft/{tail}/forecast")
                if r.status_code == 200:
                    forecast = r.json()
                    summary = forecast["summary"]
                    report(
                        "maintenance forecast",
                        PASS,
                        f"{tail}: {summary['task_count']} tasks, "
                        f"by_status={summary['by_status']}, "
                        f"{forecast['utilization']['description']}",
                    )
                else:
                    report("maintenance forecast", FAIL, f"{r.status_code}: {r.text[:200]}")
        else:
            report("maintenance fleet", FAIL, f"{r.status_code}: {r.text[:150]}")
    except Exception as exc:
        report("maintenance", FAIL, str(exc))

    # --- LaTeX -------------------------------------------------------------
    try:
        r = client.post(
            "/v1/latex",
            json={
                "source": (
                    "\\documentclass{article}\\begin{document}"
                    "Smoke test.\\end{document}"
                ),
                "compile": True,
            },
        )
        if r.status_code == 200:
            body = r.json()
            report(
                "latex compile",
                PASS if body["compiled"] else WARN,
                f"pages={body['page_count']} errors={body['errors'][:2]}",
            )
        elif r.status_code == 403:
            report("latex compile", WARN, "credential lacks the latex scope")
        else:
            report("latex compile", FAIL, f"{r.status_code}: {r.text[:200]}")
    except Exception as exc:
        report("latex compile", FAIL, str(exc))

    # --- summary ------------------------------------------------------------
    failures = [n for n, s, _ in results if s == FAIL]
    warnings = [n for n, s, _ in results if s == WARN]
    print(
        f"\n{len(results)} checks: {len(results) - len(failures) - len(warnings)} passed, "
        f"{len(warnings)} warnings, {len(failures)} failed"
    )
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
