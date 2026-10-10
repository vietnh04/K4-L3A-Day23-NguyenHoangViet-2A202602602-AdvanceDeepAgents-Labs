"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_LAST_ARXIV_CALL = 0.0
_ARXIV_LOCK = threading.Lock()


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Treat these as retryable: RetryableError, HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in (429, 500, 502, 503, 504):
                raise
            if attempt == attempts - 1:
                raise

            retry_after = None
            if isinstance(exc, RetryableError) and exc.retry_after is not None:
                try:
                    retry_after = float(exc.retry_after)
                except (ValueError, TypeError):
                    pass
            elif isinstance(exc, httpx.HTTPStatusError):
                header_val = exc.response.headers.get("retry-after") or exc.response.headers.get("Retry-After")
                if header_val:
                    try:
                        retry_after = float(header_val)
                    except ValueError:
                        pass

            if retry_after is not None:
                delay = retry_after
            else:
                delay = base * (2 ** attempt) + random.uniform(0.1, 0.5)

            delay = min(delay, cap)
            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        terms = re.findall(r"[A-Za-z0-9\-]+", query)
        if not terms:
            return "NO RESULTS"

        search_query = " AND ".join(f"all:{t}" for t in terms)
        clamped_results = max(1, min(30, int(max_results)))

        # Respect arXiv etiquette: at least 3 seconds between two calls
        with _ARXIV_LOCK:
            now = time.monotonic()
            elapsed = now - _LAST_ARXIV_CALL
            if elapsed < 3.0:
                time.sleep(3.0 - elapsed)
            _LAST_ARXIV_CALL = time.monotonic()

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    ARXIV_URL,
                    params={
                        "search_query": search_query,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                        "max_results": clamped_results,
                    },
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("retry-after")
                    raise RetryableError(f"arXiv HTTP {resp.status_code}", retry_after=retry_after)
                resp.raise_for_status()
                return resp.text

        xml_text = with_retry(_fetch, attempts=5, base=2.0, cap=60.0)

        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", ns):
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or not id_elem.text:
                continue
            raw_id = id_elem.text.strip().split("/abs/")[-1]
            paper_id = re.sub(r"v\d+$", "", raw_id)
            url = f"https://arxiv.org/abs/{paper_id}"

            pub_elem = entry.find("atom:published", ns)
            published = (pub_elem.text.strip()[:10]) if pub_elem is not None and pub_elem.text else ""

            title_elem = entry.find("atom:title", ns)
            title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else "Untitled"

            sum_elem = entry.find("atom:summary", ns)
            summary = " ".join(sum_elem.text.split())[:600] if sum_elem is not None and sum_elem.text else ""

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(100, int(limit)))
        params = {"limit": clamped_limit}
        if date and str(date).strip():
            params["date"] = str(date).strip()

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_DAILY_URL, params=params)
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("retry-after")
                    raise RetryableError(f"HF Daily HTTP {resp.status_code}", retry_after=retry_after)
                resp.raise_for_status()
                return resp.json()

        data = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list):
            return "NO RESULTS"

        records = []
        kw = keyword.strip().lower() if keyword else ""
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(paper, dict) or not paper.get("id"):
                continue

            paper_id = str(paper["id"])
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = " ".join(str(paper.get("title") or item.get("title") or "").split())
            summary = " ".join(str(paper.get("summary") or item.get("summary") or "").split())[:600]
            upvotes = int(paper.get("upvotes") or 0)
            github = str(paper.get("githubRepo") or "")
            stars = int(paper.get("githubStars") or 0)

            if kw:
                if kw not in title.lower() and kw not in summary.lower():
                    continue

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        q = str(query).strip()
        if not q:
            return "NO RESULTS"
        clamped_limit = max(1, min(50, int(limit)))

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_SEARCH_URL, params={"q": q, "limit": clamped_limit})
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("retry-after")
                    raise RetryableError(f"HF Search HTTP {resp.status_code}", retry_after=retry_after)
                resp.raise_for_status()
                return resp.json()

        data = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list):
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(paper, dict) or not paper.get("id"):
                continue

            paper_id = str(paper["id"])
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = " ".join(str(paper.get("title") or item.get("title") or "").split())
            raw_summary = (
                paper.get("ai_summary")
                or item.get("ai_summary")
                or paper.get("summary")
                or item.get("summary")
                or ""
            )
            summary = " ".join(str(raw_summary).split())[:600]
            upvotes = int(paper.get("upvotes") or 0)
            github = str(paper.get("githubRepo") or "")
            stars = int(paper.get("githubStars") or 0)

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    """Helper to invoke an MCP tool on Exa over plain HTTP with SSE parsing, rate-limit retry, and key redaction."""
    exa_key = (os.getenv("EXA_API_KEY") or "").strip()
    endpoint = f"{EXA_URL}?exaApiKey={exa_key}" if exa_key else EXA_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "DeepResearchAgent/1.0",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _post():
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("retry-after")
                raise RetryableError(f"Exa HTTP {resp.status_code}", retry_after=retry_after)
            resp.raise_for_status()

            # Parse SSE response lines
            data_payload = None
            for line in resp.text.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    raw_data = line[5:].strip()
                    try:
                        data_payload = json.loads(raw_data)
                        break
                    except json.JSONDecodeError:
                        continue

            if data_payload is None:
                # If not standard SSE, check direct json
                try:
                    data_payload = resp.json()
                except Exception:
                    raise RuntimeError("Invalid response from Exa MCP: neither SSE data nor JSON")

            if "error" in data_payload:
                err_info = str(data_payload["error"])
                if "rate" in err_info.lower() or "limit" in err_info.lower():
                    raise RetryableError(f"Exa rate limit error: {err_info}")
                raise RuntimeError(f"Exa JSON-RPC error: {err_info}")

            result = data_payload.get("result", {})

            # Check rate limiting in result._meta (free tier sends HTTP 200 with rate limit flags)
            meta = result.get("_meta")
            if meta:
                meta_str = str(meta).lower()
                if any(k in meta_str for k in ("rate_limit", "rate limited", "ratelimit", "limit exceeded", "quota")):
                    raise RetryableError(f"Exa rate limited in _meta: {meta}")

            content_list = result.get("content", [])
            texts = []
            for item in content_list:
                if item.get("type") == "text":
                    t = item.get("text", "")
                    # Also detect rate limit message inside returned text
                    lower_t = t.lower()
                    if "rate limit" in lower_t or "rate-limit" in lower_t or "too many requests" in lower_t:
                        raise RetryableError(f"Exa rate limited in text: {t[:100]}")
                    texts.append(t)

            combined_text = "\n\n".join(texts).strip()
            return combined_text

    try:
        raw_result = with_retry(_post, attempts=2, base=1.0, cap=5.0)
        return raw_result
    except Exception as exc:
        msg = f"ERROR: {type(exc).__name__}: {exc}"
        if exa_key:
            msg = msg.replace(exa_key, "[REDACTED]")
        return msg


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        q = str(query).strip()
        if not q:
            return "NO RESULTS"
        obj = str(objective).strip() if objective else f"Find authoritative research and survey resources for {q}"
        num = max(1, min(20, int(num_results)))

        text = _call_exa_mcp("web_search_exa", {"query": q, "objective": obj, "numResults": num})
        if text.startswith("ERROR:"):
            return text
        if not text:
            return "NO RESULTS"
        return text
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        u = str(url).strip()
        if not u or not (u.startswith("http://") or u.startswith("https://")):
            return "NO RESULTS"

        text = _call_exa_mcp("web_fetch_exa", {"urls": [u]})
        if text.startswith("ERROR:"):
            return text
        if not text:
            return "NO RESULTS"
        return text[:12000]
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
