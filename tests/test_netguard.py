"""The orchestrator has no network capability, and CI proves it two ways.

1. Importing every orchestrator module must not load a networking module,
   not even indirectly through a dependency.
2. At runtime, the netguard audit hook turns any socket use into a crash.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

# Modules that are *meant* to reach the network (or loopback). Each one runs
# in its own child process, never inside the orchestrator. Adding to this
# list needs a design reason (see §4 of docs/DESIGN.md).
NETWORK_CAPABLE = {
    "asif.infer.worker",  # talks to llama-server on loopback, inside the inference sandbox
}

# Not "_socket": typing_extensions (via pydantic) imports that C module on
# Python < 3.14 only to borrow its capsule type, never to open anything. The
# audit hook below still blocks every socket, _socket's included.
FORBIDDEN = (
    "socket",
    "ssl",
    "_ssl",
    "http",
    "urllib.request",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "xmlrpc",
    "asyncio",
    "requests",
    "httpx",
    "urllib3",
    "aiohttp",
    "webbrowser",
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_orchestrator_never_imports_networking_modules() -> None:
    proc = _run(
        f"""
        import importlib, json, pkgutil, sys
        import asif
        skipped = {sorted(NETWORK_CAPABLE)!r}
        imported = []
        for info in pkgutil.walk_packages(asif.__path__, "asif."):
            if info.name in skipped or info.name == "asif.__main__":
                continue
            importlib.import_module(info.name)
            imported.append(info.name)
        forbidden = {FORBIDDEN!r}
        bad = sorted(m for m in sys.modules
                     if any(m == f or m.startswith(f + ".") for f in forbidden))
        print(json.dumps({{"imported": imported, "bad": bad}}))
        """
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert "asif.cli" in report["imported"]
    assert "asif.pipeline.local" in report["imported"]
    assert "asif.infer.client" in report["imported"]
    assert report["bad"] == [], (
        f"orchestrator code pulled in networking modules: {report['bad']}. "
        "Move network use into a separate, sandboxed process."
    )


def test_audit_hook_blocks_sockets_dns_and_urls() -> None:
    proc = _run(
        """
        from asif import netguard
        netguard.install()
        netguard.install()  # idempotent
        import _socket, socket, urllib.request
        results = []
        for name, attempt in [
            ("socket", lambda: socket.socket()),
            ("_socket", lambda: _socket.socket()),
            ("dns", lambda: socket.getaddrinfo("example.com", 443)),
            ("url", lambda: urllib.request.urlopen("http://127.0.0.1:9/")),
        ]:
            try:
                attempt()
                results.append(name + ":allowed")
            except netguard.NetworkDenied:
                results.append(name + ":denied")
        print(",".join(results))
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "socket:denied,_socket:denied,dns:denied,url:denied"


def test_cli_installs_the_tripwire_first() -> None:
    proc = _run(
        """
        import sys
        from asif import cli, netguard
        sys.argv = ["asif", "version"]
        try:
            cli.main()
        except SystemExit:
            pass
        print(netguard.installed())
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("True")


def test_subprocesses_still_work_under_the_tripwire() -> None:
    proc = _run(
        """
        import subprocess, sys
        from asif import netguard
        netguard.install()
        out = subprocess.run([sys.executable, "-c", "print(42)"], capture_output=True, text=True)
        print(out.stdout.strip())
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "42"
