"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_URL_REGEX = re.compile(r"https?://\S+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not sources or not isinstance(sources, list):
        return ["no sources in sources.json"]

    source_by_n = {}
    seen_urls = {}
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"invalid source entry (not a dict): {entry!r}")
            continue
        n = entry.get("n")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source n={n!r} is not an integer")
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] has invalid url: {url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate url in sources: {url} (sources [{seen_urls[url]}] and [{n}])")
        else:
            seen_urls[url] = n

        if n in source_by_n:
            problems.append(f"duplicate source number in sources.json: [{n}]")
        else:
            source_by_n[n] = entry

    # 3. Check ## References heading
    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' heading")
        body = report_text
        ref_text = ""
    else:
        last_match = matches[-1]
        body = report_text[:last_match.start()]
        ref_text = report_text[last_match.end():]

    # 4. Check citations in body
    segments = _CODE.split(body)
    cited_numbers = set()
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            continue  # inside code block or inline code
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited_numbers.add(num)

    source_numbers = set(source_by_n.keys())
    for n in sorted(cited_numbers - source_numbers):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(source_numbers - cited_numbers):
        problems.append(f"source [{n}] never cited")

    # 5 & 6. Check References section lines
    ref_lines = [line.strip() for line in ref_text.splitlines() if line.strip()]
    seen_ref_n = {}
    for line in ref_lines:
        m = re.match(r"^\[(\d+)\]", line)
        if not m:
            problems.append(f"reference line does not start with '[n]': {line[:60]}")
            continue
        ref_n = int(m.group(1))
        seen_ref_n[ref_n] = seen_ref_n.get(ref_n, 0) + 1

        if ref_n not in source_by_n:
            problems.append(f"reference line [{ref_n}] is not in sources.json")
            continue

        urls = _URL_REGEX.findall(line)
        # Strip trailing punctuation from matched URLs if any
        cleaned_urls = [u.rstrip("),.;]") for u in urls]
        expected_url = source_by_n[ref_n].get("url")

        if len(cleaned_urls) == 0:
            problems.append(f"reference line [{ref_n}] has no URL")
        elif len(cleaned_urls) > 1:
            problems.append(f"reference line [{ref_n}] has multiple URLs (bundled sources)")
        elif cleaned_urls[0] != expected_url:
            problems.append(f"reference line [{ref_n}] URL mismatch: got '{cleaned_urls[0]}' expected '{expected_url}'")

    for n in sorted(source_numbers):
        count = seen_ref_n.get(n, 0)
        if count == 0:
            problems.append(f"missing reference line for source [{n}]")
        elif count > 1:
            problems.append(f"duplicate reference line for source [{n}] ({count} times)")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
