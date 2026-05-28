"""
FINA terminal client for the live demo.

This console stays fixed to v8 and gives the presenter quick controls for
RAG, role, period, and preset easy/medium/hard cases.
"""

import json
import time
from pathlib import Path
from typing import Any

import requests

from benchmark_cases import TEST_CASES
from benchmark_cases_diverse import DIVERSE_IDS_BY_BAND

API_URL = "http://localhost:8105/chat"
API_STREAM_URL = "http://localhost:8105/chat/stream"
API_BASE = "http://localhost:8105"
AWAD2_API = "http://100.109.225.15:4001/api"
VALID_ROLES = {"Student", "Worker", "Freelancer"}
VALID_BANDS = {"easy", "medium", "hard"}

# Real seeded AWAD2 users — used by /sweep to drive each persona against the
# actual MySQL row, not the default user_id=1.
PERSONA_USER_ID = {
    "Student":    "11111111-1111-4111-8111-111111111111",
    "Worker":     "33333333-3333-4333-8333-333333333333",
    "Freelancer": "22222222-2222-4222-8222-222222222222",
}
SWEEP_REPORT_PATH = Path(__file__).parent / "_full_test_report.md"

CASE_BY_ID = {case["id"]: case for case in TEST_CASES}
_BAND_ALIAS = {"easy": "easy", "medium": "multi", "hard": "hard"}
DEMO_CASE_IDS = {
    band: {role: bands[_BAND_ALIAS[band]][0] for role, bands in DIVERSE_IDS_BY_BAND.items()}
    for band in VALID_BANDS
}
DEMO_CASES = {
    band: {role: CASE_BY_ID[case_id] for role, case_id in role_map.items()}
    for band, role_map in DEMO_CASE_IDS.items()
}


def _print_help() -> None:
    print(
        "\nSlash commands:\n"
        "  /rag on|off                  toggle vector RAG\n"
        "  /role Student|Worker|Freelancer\n"
        "  /period <token>              e.g. month, prev_month, 2026-05, 3m\n"
        "  /users                       list real users from AWAD2 database\n"
        "  /user <id|index>             switch user (auto-fetches role + snapshot)\n"
        "  /demo easy|medium|hard [role]\n"
        "  /sweep                       run the 9-cell matrix (3 roles x 3 bands)\n"
        "                                 and write _full_test_report.md\n"
        "  /cases                       show preset demo cases\n"
        "  /show                        print current config\n"
        "  /replay                      re-send the previous prompt with current config\n"
        "  /help                        show this list\n"
        "  /quit | /exit                leave\n"
    )


_USERS_CACHE: list[dict] = []


def _fetch_users() -> list[dict] | None:
    try:
        response = requests.get(f"{AWAD2_API}/fina/users", timeout=5)
        response.raise_for_status()
        data = response.json()
        users = data.get("users", data) if isinstance(data, dict) else data
        return users if isinstance(users, list) else None
    except requests.exceptions.RequestException as e:
        print(f"  error: cannot reach AWAD2 ({AWAD2_API}): {e}")
        return None


def _print_users() -> None:
    global _USERS_CACHE
    users = _fetch_users()
    if users is None:
        return
    _USERS_CACHE = users
    if not users:
        print("\nNo users returned.")
        return
    print(f"\nReal users from AWAD2 ({len(users)} total):")
    print(f"  {'#':>2}  {'id':36}  {'fullName':20}  {'email':36}  acc")
    for i, u in enumerate(users, start=1):
        uid = str(u.get("id", ""))[:36]
        name = str(u.get("fullName", ""))[:20]
        email = str(u.get("email", ""))[:36]
        acc = u.get("accountCount", "?")
        print(f"  {i:>2}  {uid:36}  {name:20}  {email:36}  {acc}")
    print("  tip: /user <index> or /user <full-uuid>\n")


def _resolve_user_arg(arg: str) -> str:
    """If arg is a small integer matching a cached row, return that user's id."""
    if arg.isdigit() and _USERS_CACHE:
        idx = int(arg)
        if 1 <= idx <= len(_USERS_CACHE):
            return str(_USERS_CACHE[idx - 1].get("id", arg))
    return arg


def _fetch_user_profile(user_id: str) -> dict | None:
    try:
        response = requests.get(f"{AWAD2_API}/fina/users/{user_id}", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except requests.exceptions.RequestException as e:
        print(f"  warn: profile fetch failed: {e}")
        return None


def _fetch_dashboard(user_id: str, period: str) -> dict | None:
    try:
        response = requests.get(
            f"{API_BASE}/dashboard/{user_id}",
            params={"period": period},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except requests.exceptions.RequestException as e:
        print(f"  warn: dashboard fetch failed (is api.py running?): {e}")
        return None


def _print_snapshot(user_id: str, period: str) -> None:
    profile = _fetch_user_profile(user_id)
    if profile:
        name = profile.get("name") or profile.get("fullName") or "?"
        role = profile.get("role", "?")
        currency = profile.get("currency", "?")
        print(f"  profile: name={name}  role={role}  currency={currency}")
    dash = _fetch_dashboard(user_id, period)
    if not dash:
        return
    cards = dash.get("summary_cards") or []
    if cards:
        print(f"  snapshot ({period}):")
        for c in cards:
            print(f"    - {c.get('title', '?')}: {c.get('subtitle', '?')}")
    insights = dash.get("smart_insights") or []
    for ins in insights[:1]:
        print(f"    - {ins.get('title', '?')}: {ins.get('desc', '')}")


def _print_cases() -> None:
    print("\nPreset demo cases:")
    for band in ("easy", "medium", "hard"):
        print(f"  {band}:")
        for role, case in DEMO_CASES[band].items():
            print(f"    - {role}: {case['id']} | {case['name']}")


def _print_config(cfg: dict) -> None:
    print(
        f"  user_id={cfg['user_id']}  role={cfg['role']}  "
        f"adapter=v8(fixed)  rag={'on' if cfg['use_rag'] else 'off'}  "
        f"period={cfg['period']}"
    )


def _demo_case(band: str, role: str):
    case = DEMO_CASES[band][role]
    return case


def _format_vnd(amount) -> str:
    try:
        if amount is None:
            return "0"
        return "{:,.0f}".format(float(amount)).replace(",", ".")
    except Exception:
        return str(amount)


def _fetch_ground_truth(user_id: str, period: str) -> dict | None:
    """Fetch the deterministic financial summary for a user+period from AWAD2.

    Returns the parsed /summary JSON envelope or None if unreachable. This is
    the same MySQL-derived data api.py uses to build FINANCIAL CONTEXT — so
    it is the source-of-truth panel viewers should see alongside the model's
    prose.
    """
    try:
        response = requests.get(
            f"{AWAD2_API}/fina/users/{user_id}/summary",
            params={"period": period},
            timeout=5,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def _fetch_budget_prefs(user_id: str) -> dict | None:
    """Fetch the user's needs/wants/savings split percentages from AWAD2.

    Returns {needs_pct, wants_pct, savings_pct} when the user has a saved
    preference row, None when the system falls back to 50/30/20. Mirrors
    the behaviour of mac_client.get_budget_preferences — but lives here so
    the console can render the live split alongside the ground-truth panel
    independent of api.py.
    """
    try:
        response = requests.get(
            f"{AWAD2_API}/fina/users/{user_id}/budget-preferences",
            timeout=5,
        )
        response.raise_for_status()
        if not response.content or response.text.strip() in ("", "null"):
            return None
        data = response.json()
        if isinstance(data, dict) and data.get("needs_pct") is not None:
            return data
        return None
    except requests.exceptions.RequestException:
        return None


def _print_ground_truth_panel(
    period: str, gt: dict | None, prefs: dict | None
) -> None:
    if gt is None:
        print(f"\n----- Ground truth ({period}) -----")
        print("  (could not reach AWAD2 /summary)")
        print("-----------------------------------\n")
        return
    currency = gt.get("currency", "VND")
    income = gt.get("income") or 0
    computed = gt.get("computed") or {}
    total_spent = computed.get("total_spent") or 0
    surplus = computed.get("surplus") or 0
    savings_rate = computed.get("savings_rate_pct") or 0
    top_cat = computed.get("top_category") or "-"
    top_cat_spent = computed.get("top_category_spent") or 0
    top_cat_pct = computed.get("top_category_pct") or 0  # % of income
    spent_pct_income = (total_spent / income * 100.0) if income else 0.0
    top_pct_spend = (top_cat_spent / total_spent * 100.0) if total_spent else 0.0

    print(f"\n----- Ground truth ({period}) -----")
    print(f"  Income:        {_format_vnd(income):>14} {currency}")
    print(
        f"  Total spent:   {_format_vnd(total_spent):>14} {currency}  "
        f"({spent_pct_income:.1f}% of income)"
    )
    print(f"  Surplus:       {_format_vnd(surplus):>14} {currency}")
    print(f"  Savings rate:  {savings_rate}%")
    print(
        f"  Top category:  {top_cat}  {_format_vnd(top_cat_spent)} {currency}  "
        f"({top_pct_spend:.1f}% of spend, {top_cat_pct}% of income)"
    )

    # Live budget split. Percentages: AWAD2 /budget-preferences (prefs arg).
    # VND limits: AWAD2 /summary's computed.budget_50_30_20 — server-computed
    # against the same MySQL row, so the two are guaranteed consistent. If
    # AWAD2 has no prefs row for this user, label as default and recompute
    # 50/30/20 client-side.
    if prefs is not None:
        n_pct = prefs.get("needs_pct", 50)
        w_pct = prefs.get("wants_pct", 30)
        s_pct = prefs.get("savings_pct", 20)
        split_label = f"{n_pct}/{w_pct}/{s_pct} (custom)"
    else:
        n_pct, w_pct, s_pct = 50, 30, 20
        split_label = "50/30/20 (default)"

    limits = computed.get("budget_50_30_20") or {}
    needs_limit = limits.get("needs_limit") if limits else income * n_pct / 100
    wants_limit = limits.get("wants_limit") if limits else income * w_pct / 100
    savings_target = limits.get("savings_target") if limits else income * s_pct / 100

    print(f"  Budget split:  {split_label}")
    print(f"    Needs   <= {_format_vnd(needs_limit):>14} {currency}")
    print(f"    Wants   <= {_format_vnd(wants_limit):>14} {currency}")
    print(f"    Savings >= {_format_vnd(savings_target):>14} {currency}")
    print("-----------------------------------")


def _print_retrieved_evidence_panel(sources: list[dict], rag_on: bool) -> None:
    if not rag_on:
        print("  Retrieved evidence: skipped (/rag is off)")
        return
    if not sources:
        print("  Retrieved evidence: none returned")
        return
    print(f"\n----- Retrieved evidence ({len(sources)} sources) -----")
    for src in sources:
        sid = src.get("id", "S?")
        text = (src.get("text") or "").strip().replace("\n", " ")
        print(f"  [{sid}] {text}")
    print("--------------------------------------------------")


def _send(cfg: dict, message: str) -> None:
    payload = {
        "user_id": cfg["user_id"],
        "role": cfg["role"],
        "mode": "Standard",
        "period": cfg["period"],
        "message": message,
        "use_rag": cfg["use_rag"],
    }
    print(
        f"\n[v8 | rag={'on' if cfg['use_rag'] else 'off'} | "
        f"role={cfg['role']} | period={cfg['period']}] ..."
    )

    # A. Ground-truth panel: pulled from AWAD2 /summary (deterministic, MySQL-
    # derived). Always shown — independent of the RAG toggle — so viewers see
    # the source of truth above whatever the model says.
    gt = _fetch_ground_truth(cfg["user_id"], cfg["period"])
    prefs = _fetch_budget_prefs(cfg["user_id"])
    _print_ground_truth_panel(cfg["period"], gt, prefs)

    # Stream tokens from /chat/stream so the model "types" in real time.
    # If streaming fails for any reason, fall back to a single POST to /chat
    # so the demo never goes silent.
    streamed = _send_streaming(payload, rag_on=cfg["use_rag"])
    if streamed:
        return
    print("  warn: streaming failed, falling back to blocking /chat...")

    try:
        response = requests.post(API_URL, json=payload, timeout=120)
    except requests.exceptions.ConnectionError:
        print(" Error: cannot reach api.py. Is it running on port 8105?")
        return
    if response.status_code != 200:
        print(f" Error {response.status_code}: {response.text}")
        return

    data = response.json()
    msg = data.get("response") or data.get("message") or json.dumps(data, ensure_ascii=False)
    print(f"\nFINA: {msg}")
    if isinstance(data, dict):
        rag = data.get("rag")
        rag_status = rag.get("status", "n/a") if isinstance(rag, dict) else "n/a"
        sources = data.get("retrieved_sources") or []
        _print_retrieved_evidence_panel(sources, rag_on=cfg["use_rag"])
        print(f"  RAG status: {rag_status}")
        meta = data.get("meta")
        if isinstance(meta, dict):
            if "schema_status" in meta:
                print(f"  schema_status: {meta['schema_status']}")
            action_safety = meta.get("action_safety")
            if isinstance(action_safety, dict):
                print(f"  action_safety: {action_safety}")
            numeric_guard = meta.get("numeric_guard")
            if isinstance(numeric_guard, dict) and numeric_guard.get("redacted"):
                print(f"  numeric_guard: redacted {numeric_guard['redacted']}")


def _send_streaming(payload: dict, rag_on: bool) -> bool:
    """POST to /chat/stream and print tokens as they arrive (SSE).

    Returns True on a clean stream end, False on any error so the caller can
    fall back to the blocking /chat path. Prints:
      - "FINA: " prefix
      - each token from data:{"token": ...} chunks as it arrives
      - newline + retrieved-evidence panel + RAG/schema/safety lines from
        the terminal data:{"final": ..., "rag": ..., "meta": ...} chunk
    """
    print("\nFINA: ", end="", flush=True)
    final_dict: dict | None = None
    final_rag: dict = {}
    final_meta: dict = {}
    try:
        with requests.post(
            API_STREAM_URL, json=payload, stream=True, timeout=180
        ) as response:
            if response.status_code != 200:
                print(f"\n  stream error {response.status_code}: {response.text[:200]}")
                return False
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if not raw.startswith("data:"):
                    continue
                chunk = raw[len("data:"):].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if "token" in obj:
                    print(obj["token"], end="", flush=True)
                elif "final" in obj:
                    final_dict = obj.get("final") or {}
                    final_rag = obj.get("rag") or {}
                    final_meta = obj.get("meta") or {}
    except requests.exceptions.RequestException as e:
        print(f"\n  stream exception: {e}")
        return False

    print()  # newline after stream
    sources = []
    if final_dict is None:
        # Stream ended without a {final: ...} envelope — still treat as ok if
        # we got tokens, but we have no metadata to print.
        return True
    rag_status = final_rag.get("status", "n/a") if isinstance(final_rag, dict) else "n/a"
    # rag_result.as_dict() uses key "retrieved_sources" — not "sources"
    # (rag/retriever.py:60-70).
    sources = final_rag.get("retrieved_sources") or []
    _print_retrieved_evidence_panel(sources, rag_on=rag_on)
    print(f"  RAG status: {rag_status}")
    if isinstance(final_meta, dict):
        if "schema_status" in final_meta:
            print(f"  schema_status: {final_meta['schema_status']}")
        action_safety = final_meta.get("action_safety")
        if isinstance(action_safety, dict):
            print(f"  action_safety: {action_safety}")
        numeric_guard = final_meta.get("numeric_guard")
        if isinstance(numeric_guard, dict) and numeric_guard.get("redacted"):
            print(f"  numeric_guard: redacted {numeric_guard['redacted']}")
    return True


def _post_chat(payload: dict) -> dict | None:
    """POST to /chat and return the parsed JSON envelope (no printing).

    Separate from _send() so /sweep can run cells silently and aggregate
    results without spamming the terminal with 9 full responses.
    """
    try:
        response = requests.post(API_URL, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  error: {e}")
        return None


def _run_sweep(cfg: dict) -> None:
    """Run the 9-cell matrix (3 roles x 3 bands) and write a Markdown report.

    Uses the seeded AWAD2 user IDs (PERSONA_USER_ID), the current cfg's
    `period` and `use_rag`, and the same preset cases /demo fires from
    DIVERSE_IDS_BY_BAND. cfg["user_id"] and cfg["role"] are NOT mutated.
    """
    matrix = [
        (band, role)
        for band in ("easy", "medium", "hard")
        for role in ("Student", "Worker", "Freelancer")
    ]
    print(
        f"\nRunning 9-cell sweep "
        f"(period={cfg['period']}, rag={'on' if cfg['use_rag'] else 'off'})..."
    )
    results: list[dict] = []
    for i, (band, role) in enumerate(matrix, start=1):
        cid = DEMO_CASE_IDS[band][role]
        case = CASE_BY_ID[cid]
        payload = {
            "user_id": PERSONA_USER_ID[role],
            "role": role,
            "mode": "Standard",
            "period": cfg["period"],
            "message": case["question"],
            "use_rag": cfg["use_rag"],
        }
        t0 = time.time()
        print(f"  [{i}/9] {band:6} {role:11} {cid} ...", end=" ", flush=True)
        data = _post_chat(payload)
        elapsed = round(time.time() - t0, 1)
        if data is None:
            print(f"FAILED ({elapsed}s)")
            results.append({
                "band": band, "role": role, "case_id": cid, "name": case["name"],
                "question": case["question"], "error": "request failed",
                "elapsed_sec": elapsed,
            })
            continue
        schema_status = (data.get("meta") or {}).get("schema_status") or {}
        model_output = data.get("model_output") or {}
        rag = data.get("rag") or {}
        sources = data.get("retrieved_sources") or []
        rec = {
            "band": band,
            "role": role,
            "case_id": cid,
            "name": case["name"],
            "question": case["question"],
            "response": data.get("response") or data.get("message"),
            "kind": model_output.get("kind"),
            "needs_clarification": model_output.get("needs_clarification"),
            "action": model_output.get("action"),
            "rag_status": rag.get("status"),
            "rag_sources_count": len(sources),
            "rag_source_ids": [s.get("id") for s in sources[:3]],
            "schema_status": schema_status,
            "action_safety": (data.get("meta") or {}).get("action_safety"),
            "elapsed_sec": elapsed,
        }
        print(
            f"ok ({elapsed}s, kind={rec['kind']}, "
            f"schema_ok={schema_status.get('ok')}, rag={rec['rag_status']})"
        )
        results.append(rec)

    _write_sweep_report(results, cfg)
    print(f"\nReport written: {SWEEP_REPORT_PATH}")


def _write_sweep_report(results: list[dict], cfg: dict) -> None:
    lines = ["# FINA full test report — 3 roles x 3 bands\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(
        f"Endpoint: {API_URL} | period={cfg['period']} | "
        f"use_rag={cfg['use_rag']}\n\n"
    )
    lines.append("## Summary table\n")
    lines.append(
        "| # | band | role | case | name | schema_ok | kind | rag | sources | sec |"
    )
    lines.append(
        "|---|------|------|------|------|-----------|------|-----|---------|-----|"
    )
    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(
                f"| {i} | {r['band']} | {r['role']} | {r['case_id']} | "
                f"{r['name']} | ERR | - | - | - | {r['elapsed_sec']} |"
            )
        else:
            schema_ok = (r["schema_status"] or {}).get("ok")
            lines.append(
                f"| {i} | {r['band']} | {r['role']} | {r['case_id']} | "
                f"{r['name']} | {schema_ok} | {r['kind']} | "
                f"{r['rag_status']} | {r['rag_sources_count']} | "
                f"{r['elapsed_sec']} |"
            )
    lines.append("\n## Per-case detail\n")
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. [{r['band']}/{r['role']}] {r['case_id']} — {r['name']}\n")
        lines.append(f"**Q:** {r['question']}\n")
        if "error" in r:
            lines.append(f"**ERROR:** `{r['error']}`\n")
            continue
        schema_ok = (r["schema_status"] or {}).get("ok")
        lines.append(
            f"- schema_ok: `{schema_ok}` | kind: `{r['kind']}` | "
            f"needs_clarification: `{r['needs_clarification']}`"
        )
        src_ids = ", ".join(r["rag_source_ids"]) or "-"
        lines.append(
            f"- rag_status: `{r['rag_status']}` | sources: "
            f"`{r['rag_sources_count']}` ({src_ids})"
        )
        lines.append(f"- action_safety: `{r['action_safety']}`")
        if r.get("action"):
            lines.append(f"- action: `{json.dumps(r['action'], ensure_ascii=False)}`")
        lines.append(f"- elapsed: {r['elapsed_sec']}s\n")
        lines.append("**A:**\n")
        lines.append("> " + (r.get("response") or "").replace("\n", "\n> "))
        lines.append("")

    SWEEP_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _handle_command(line: str, cfg: dict, last_prompt: str | None):
    parts = line.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("/quit", "/exit"):
        return "quit", last_prompt
    if cmd == "/help":
        _print_help()
        return "ok", last_prompt
    if cmd == "/show":
        _print_config(cfg)
        return "ok", last_prompt
    if cmd == "/cases":
        _print_cases()
        return "ok", last_prompt
    if cmd == "/rag":
        if len(args) == 1 and args[0].lower() in ("on", "off"):
            cfg["use_rag"] = args[0].lower() == "on"
            print(f"  rag -> {'on' if cfg['use_rag'] else 'off'}")
        else:
            print("  usage: /rag on|off")
        return "ok", last_prompt
    if cmd == "/role":
        if len(args) == 1 and args[0] in VALID_ROLES:
            cfg["role"] = args[0]
            print(f"  role -> {cfg['role']}")
        else:
            print(f"  invalid role. choose from: {sorted(VALID_ROLES)}")
        return "ok", last_prompt
    if cmd == "/period":
        if len(args) == 1:
            cfg["period"] = args[0]
            print(f"  period -> {cfg['period']}")
        else:
            print("  usage: /period <token>  e.g. month, prev_month, 2026-05, 3m")
        return "ok", last_prompt
    if cmd == "/users":
        _print_users()
        return "ok", last_prompt
    if cmd == "/user":
        if len(args) == 1:
            resolved = _resolve_user_arg(args[0])
            cfg["user_id"] = resolved
            print(f"  user_id -> {cfg['user_id']}")
            profile = _fetch_user_profile(resolved)
            if profile and profile.get("role") in VALID_ROLES:
                cfg["role"] = profile["role"]
                print(f"  role -> {cfg['role']} (from AWAD2)")
            _print_snapshot(resolved, cfg["period"])
        else:
            print("  usage: /user <id|index>   (run /users first to see indexes)")
        return "ok", last_prompt
    if cmd == "/demo":
        if not args:
            print("  usage: /demo easy|medium|hard [role]")
            return "ok", last_prompt
        band = args[0].lower()
        role = args[1] if len(args) > 1 else cfg["role"]
        if band not in VALID_BANDS:
            print(f"  invalid band. choose from: {sorted(VALID_BANDS)}")
            return "ok", last_prompt
        if role not in VALID_ROLES:
            print(f"  invalid role. choose from: {sorted(VALID_ROLES)}")
            return "ok", last_prompt
        case = _demo_case(band, role)
        cfg["role"] = case["role"]
        print(f"  demo case -> {band}/{role}: {case['id']} | {case['name']}")
        _send(cfg, case["question"])
        return "ok", case["question"]
    if cmd == "/sweep":
        _run_sweep(cfg)
        return "ok", last_prompt
    if cmd == "/replay":
        if last_prompt is None:
            print("  no previous prompt to replay")
        else:
            print(f"  replaying: {last_prompt!r}")
            _send(cfg, last_prompt)
        return "ok", last_prompt

    print(f"  unknown command: {cmd}. type /help for the list.")
    return "ok", last_prompt


def start_console_chat():
    print("\n" + "=" * 60)
    print(" FINA TERMINAL CLIENT (demo console)")
    print(f"    Connected to: {API_URL}")
    print("    Fixed runtime: v8")
    print("    Watch api.py logs for retrieval/context traces.")
    print("    Type /help for slash commands, /quit to exit.")
    print("=" * 60)

    cfg = {
        "user_id": "1",
        "role": "Worker",
        "use_rag": True,
        "period": "prev_month",
    }
    _print_config(cfg)
    last_prompt: str | None = None

    while True:
        prefix = (
            f"\n[u{cfg['user_id']} | {cfg['role']} | v8 | "
            f"rag:{'on' if cfg['use_rag'] else 'off'} | {cfg['period']}] You: "
        )
        try:
            user_msg = input(prefix)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_msg.strip():
            continue

        if user_msg.startswith("/"):
            status, last_prompt = _handle_command(user_msg, cfg, last_prompt)
            if status == "quit":
                print("Exiting...")
                break
            continue

        last_prompt = user_msg
        _send(cfg, user_msg)


if __name__ == "__main__":
    start_console_chat()
