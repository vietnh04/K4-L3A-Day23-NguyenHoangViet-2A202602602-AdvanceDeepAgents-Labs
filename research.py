"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    if not topic or not isinstance(topic, str):
        return "topic"
    clean = re.sub(r"[^\w]+", "-", topic.lower()).strip("-")
    clean = clean[:60].rstrip("-")
    return clean if clean else "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Please conduct an in-depth literature research survey on the topic: '{topic}'.\n\n"
        f"Follow your complete lifecycle:\n"
        f"1. Call write_todos to plan and decompose the topic into at least 3 independent sub-questions.\n"
        f"2. Delegate each sub-question in parallel to researcher subagents using the task tool. "
        f"Provide complete context (overall topic, sub-question, notes path in {WORKDIR}/research/notes/, and notes format). "
        f"Subagents run synchronously; NEVER run sleep commands.\n"
        f"3. Read notes and verify that findings span at least 3 distinct source families among arxiv, hf-daily, hf-search, and web. "
        f"Delegate an extra researcher if fewer than 3 families are found.\n"
        f"4. Merge notes into {SOURCES_PATH} (starting from n=1, no duplicate URLs).\n"
        f"5. Write the survey body to {REPORT_PATH} following REPORT_TEMPLATE.md with inline citations [n]. "
        f"DO NOT write the ## References section.\n"
        f"6. Run python3 {FINALIZER_PATH} using the execute tool to automatically generate references and renumber citations.\n"
        f"7. Run python3 {VALIDATOR_PATH} using the execute tool until it prints OK.\n"
        f"8. Delegate spot-checking of key claims to the citation-checker subagent."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_counts = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        t_calls = getattr(msg, "tool_calls", None)
        if t_calls and isinstance(t_calls, list):
            for call in t_calls:
                if isinstance(call, dict) and "name" in call:
                    tool_counts[call["name"]] += 1

        usage = getattr(msg, "usage_metadata", None)
        if usage and isinstance(usage, dict):
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

    subagent_calls = tool_counts.get("task", 0)

    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counts),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes:
        raise RuntimeError(f"Report is missing at {REPORT_PATH}")
    report_text = report_bytes.decode("utf-8", errors="replace").strip()
    if not report_text:
        raise RuntimeError(f"Report is empty at {REPORT_PATH}")

    if not sources_bytes:
        raise RuntimeError(f"sources.json is missing at {SOURCES_PATH}")
    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources_data, list) or not sources_data:
            raise ValueError("sources.json must be a non-empty JSON list")
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON or empty: {exc}")

    slug = slugify(topic)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    distinct_families = sorted(list({
        str(s.get("source")) for s in sources_data if isinstance(s, dict) and s.get("source")
    }))
    summary = summarize(messages, elapsed, model_name)
    meta = {
        "topic": topic,
        **summary,
        "n_sources": len(sources_data),
        "source_families": distinct_families,
    }

    meta_path = reports_path / f"{slug}.meta.json"
    sources_file = reports_path / f"{slug}.sources.json"
    report_file = reports_path / f"{slug}.md"

    sources_file.write_bytes(sources_bytes)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_file.write_bytes(report_bytes)

    return report_file


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic_str = str(topic).strip()
    if not topic_str:
        print("Usage: python research.py \"<topic>\"", file=sys.stderr)
        return 2

    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or str(model)
    start = time.monotonic()

    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
        })

        print(f"[{time.strftime('%X')}] Sandbox initialized. Building lead agent...", flush=True)
        agent = build_lead_agent(backend, model)
        print(f"[{time.strftime('%X')}] Running agent for '{topic_str}'...", flush=True)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": build_prompt(topic_str)}]},
            config={"recursion_limit": 1000},
        )
        print(f"[{time.strftime('%X')}] Agent invocation finished. Saving outputs...", flush=True)

        messages = result.get("messages", []) if isinstance(result, dict) else []
        elapsed = time.monotonic() - start

        try:
            out_file = save_outputs(backend, topic_str, messages, elapsed, model_name)
            print(f"SUCCESS: Report saved to {out_file}")
            return 0
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
