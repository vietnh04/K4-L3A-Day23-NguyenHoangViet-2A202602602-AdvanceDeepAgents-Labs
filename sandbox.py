"""Sandboxes for the agent: Daytona (default, https://docs.langchain.com/oss/python/deepagents/sandboxes) or a local
Docker container (SANDBOX=docker) for development without a Daytona account.

Usage:
    with open_sandbox() as backend:        # a sandbox backend: file tools + the `execute` shell tool
        upload(backend, {"/tmp/work/x.py": b"print(1)"})
        agent = create_deep_agent(..., backend=backend)
        files = download(backend, ["/tmp/work/report/report.md"])

The sandbox is ALWAYS stopped/removed on exit (sandboxes cost resources until stopped).
NEVER put secrets inside a sandbox: an agent that reads attacker-controlled web text can be tricked into
running commands there. The Docker sandbox therefore runs with `--network none`.
"""
import os
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager

from daytona import CreateSandboxFromSnapshotParams, Daytona
from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox
from dotenv import load_dotenv
from langchain_daytona import DaytonaSandbox

load_dotenv()

DEFAULT_IMAGE = os.getenv("SANDBOX_IMAGE", "python:3.12-slim")
MAX_OUTPUT_CHARS = 100_000


class FastDaytonaSandbox(DaytonaSandbox):
    """Daytona sandbox backend that skips redundant sleep commands."""

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if re.search(r"\bsleep\s+\d+", command):
            return ExecuteResponse(
                output="Notice: sleep command is skipped. Subagents execute synchronously and all results are already available. Proceed directly to inspecting notes or running python scripts.",
                exit_code=0,
                truncated=False,
            )
        return super().execute(command, timeout=timeout)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if re.search(r"\bsleep\s+\d+", command):
            return ExecuteResponse(
                output="Notice: sleep command is skipped. Subagents execute synchronously and all results are already available. Proceed directly to inspecting notes or running python scripts.",
                exit_code=0,
                truncated=False,
            )
        return await super().aexecute(command, timeout=timeout)



class DockerSandbox(BaseSandbox):
    """A throw-away Docker container used as a Deep Agents sandbox backend (needs python3 inside the image)."""

    def __init__(self, image=DEFAULT_IMAGE, runner=subprocess.run, name=None):
        self._run = runner
        self._name = name or f"deepresearch-{uuid.uuid4().hex[:8]}"
        try:
            started = self._run(
                ["docker", "run", "-d", "--rm", "--name", self._name, "--network", "none", "--memory", "1g",
                 "--cpus", "1", "--pids-limit", "256", "-w", "/tmp", image, "sleep", "infinity"],
                capture_output=True, timeout=120)
            error = None if started.returncode == 0 else started.stderr.decode(errors="replace")
        except subprocess.TimeoutExpired:
            error = "docker run timed out"
        if error is not None:
            self._run(["docker", "rm", "-f", self._name], capture_output=True)  # never leave a half-created container
            raise RuntimeError(f"cannot start the sandbox container: {error}")

    @property
    def id(self):
        return self._name

    def execute(self, command, *, timeout=None):
        if re.search(r"\bsleep\s+\d+", command):
            return ExecuteResponse(
                output="Notice: sleep command is skipped. Subagents execute synchronously and all results are already available. Proceed directly to inspecting notes or running python scripts.",
                exit_code=0,
                truncated=False,
            )
        timeout = timeout or 120
        try:
            # the output is capped INSIDE the container: the host never buffers more than MAX_OUTPUT_CHARS
            capped = f"set -o pipefail; ( {command}\n) 2>&1 | head -c {MAX_OUTPUT_CHARS}"
            proc = self._run(["docker", "exec", self._name, "bash", "-c", capped], capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecuteResponse(output=f"Command timed out after {timeout}s", exit_code=124, truncated=False)
        text = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
        return ExecuteResponse(output=text[:MAX_OUTPUT_CHARS], exit_code=proc.returncode,
                               truncated=len(text) >= MAX_OUTPUT_CHARS)

    def upload_files(self, files):
        results = []
        for path, content in files:
            proc = self._run(["docker", "exec", "-i", self._name, "sh", "-c",
                              'mkdir -p "$(dirname "$1")" && cat > "$1"', "sh", path],
                             input=content, capture_output=True)
            results.append(FileUploadResponse(path=path, error=None if proc.returncode == 0 else "permission_denied"))
        return results

    def download_files(self, paths):
        results = []
        for path in paths:
            proc = self._run(["docker", "exec", self._name, "sh", "-c", 'cat "$1"', "sh", path], capture_output=True)
            if proc.returncode == 0:
                results.append(FileDownloadResponse(path=path, content=proc.stdout, error=None))
                continue
            stderr = proc.stderr.decode(errors="replace")
            error = "is_directory" if "directory" in stderr.lower() else (
                "permission_denied" if "denied" in stderr.lower() else "file_not_found")
            results.append(FileDownloadResponse(path=path, content=None, error=error))
        return results

    def stop(self):
        self._run(["docker", "rm", "-f", self._name], capture_output=True)


def sandbox_kind():
    return (os.getenv("SANDBOX") or "daytona").strip().lower()


def _cleanup(*steps):
    for step in steps:
        try:
            step()
        except Exception as exc:  # never hide the original error
            print(f"[sandbox] cleanup warning: {exc}", file=sys.stderr)


def _cleanup_with_fallback(primary, fallback):
    try:
        primary()
    except Exception as exc:  # never hide the original error; try the fallback so nothing is left running
        print(f"[sandbox] cleanup warning: {exc}", file=sys.stderr)
        _cleanup(fallback)


@contextmanager
def open_sandbox(kind=None):
    kind = (kind or sandbox_kind()).lower()
    if kind == "daytona":
        client = Daytona()  # reads DAYTONA_API_KEY
        # no network (a context-injected agent cannot exfiltrate; all network tools run on the host) and ephemeral
        # (removed when stopped, so a killed process does not leave a billable sandbox behind)
        box = client.create(CreateSandboxFromSnapshotParams(network_block_all=True, ephemeral=True,
                                                            auto_stop_interval=30))
        try:
            yield FastDaytonaSandbox(sandbox=box)
        finally:
            _cleanup_with_fallback(box.stop, lambda: client.delete(box))  # ephemeral: stop() already removes it
    elif kind == "docker":
        box = DockerSandbox()
        try:
            yield box
        finally:
            _cleanup(box.stop)
    else:
        raise ValueError(f"SANDBOX must be 'daytona' or 'docker', got {kind!r}")


def upload(backend, files):
    """files: {absolute_path: bytes}. Seeds the sandbox before the agent runs."""
    backend.upload_files(list(files.items()))


def download(backend, paths):
    """Returns {absolute_path: bytes or None (missing/failed)}."""
    return {result.path: result.content for result in backend.download_files(paths)}
