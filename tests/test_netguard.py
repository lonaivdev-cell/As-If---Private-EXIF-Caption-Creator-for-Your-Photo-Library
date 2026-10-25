"""The orchestrator has no network capability, and CI proves it three ways.

1. Importing every orchestrator module must not load a networking module,
   not even indirectly through a dependency (apart from a short, reasoned
   list of harmless importers).
2. No orchestrator source file imports one, even lazily inside a function.
3. At runtime, the netguard audit hook turns any socket use into a crash.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path

import pytest

import asif

# Modules that are *meant* to reach the network (or loopback). Each one runs
# in its own child process, never inside the orchestrator. Adding to this
# list needs a design reason (see §4 of docs/DESIGN.md).
NETWORK_CAPABLE = {
    "asif.infer.worker",  # talks to llama-server on loopback, inside the inference sandbox
}

FORBIDDEN = (
    "socket",
    "_socket",
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

# Imports of a forbidden module known to be harmless: the module, and the
# files whose import statements may load it. The audit hook still blocks any
# actual socket use in the orchestrator, these included.
BENIGN_IMPORTERS: dict[str, tuple[str, ...]] = {
    # The capsule type, which typing_extensions (via pydantic) looks up:
    # Python 3.13's types.CapsuleType imports _socket to get it, and on 3.12
    # typing_extensions does the same itself. Nothing is opened.
    "_socket": (
        os.sep + "types.py",
        os.sep + "typing_extensions.py",
    ),
    # On Python 3.12 the stdlib's email.utils imports socket at module level,
    # only for make_msgid()'s getfqdn(). It's reached through
    # importlib.metadata (pydantic's plugin loader). Python 3.13 made it lazy.
    "socket": (os.sep + os.path.join("email", "utils.py"),),
}

# Runs in a fresh interpreter. Records every import statement that names a
# forbidden module (with the importing file), not just the first one, so an
# offender can't hide behind a harmless module that loaded it earlier.
_PROBE = """
import builtins, importlib, importlib.abc, json, pkgutil, sys, traceback

FORBIDDEN = {forbidden!r}
seen = []


def forbidden(name):
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN)


class Spy(importlib.abc.MetaPathFinder):
    # First loads, including importlib.import_module(), which skips __import__.
    def find_spec(self, name, path, target=None):
        if forbidden(name):
            # Skip importlib's frames and this probe's own (the __import__ wrapper).
            frames = [f.filename for f in traceback.extract_stack()[:-1]
                      if not f.filename.startswith(("<frozen", "<string>"))]
            seen.append((name, frames[-1] if frames else "?"))
        return None


_real_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    # Every import statement, even of a module that is already loaded.
    if level == 0 and forbidden(name):
        seen.append((name, (globals or {{}}).get("__file__") or "?"))
    return _real_import(name, globals, locals, fromlist, level)


startup = sorted(m for m in sys.modules if forbidden(m))
sys.meta_path.insert(0, Spy())
builtins.__import__ = _import
imported = []
for target in {targets!r}:
    if target == "*":
        import asif
        for info in pkgutil.walk_packages(asif.__path__, "asif."):
            if info.name in {skipped!r} or info.name == "asif.__main__":
                continue
            importlib.import_module(info.name)
            imported.append(info.name)
    else:
        try:
            importlib.import_module(target)
        except ModuleNotFoundError as exc:
            if exc.name != target:
                raise
            # "from pkg import name": name is an attribute, perhaps one that
            # loads a module on first use (concurrent.futures.ProcessPoolExecutor).
            parent, _, attr = target.rpartition(".")
            getattr(importlib.import_module(parent), attr, None)
        imported.append(target)
builtins.__import__ = _real_import
# Imports made *by* a forbidden module are already blamed on whoever loaded it.
inside = sorted(getattr(sys.modules[m], "__file__", None) or "" for m in sys.modules
                if forbidden(m))
print(json.dumps({{"imported": imported, "seen": sorted(set(seen)), "startup": startup,
                  "inside": inside}}))
"""


def _run(code: str, *, extra_path: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if extra_path:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [extra_path, env.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )


def forbidden_imports(
    *targets: str, extra_path: str | None = None
) -> tuple[list[str], list[tuple[str, str]]]:
    """Import ``targets`` ("*" = every orchestrator module) in a fresh interpreter.

    Returns what was imported, and every (forbidden module, importing file)
    pair that isn't on the benign list.
    """
    proc = _run(
        _PROBE.format(forbidden=FORBIDDEN, targets=list(targets), skipped=sorted(NETWORK_CAPABLE)),
        extra_path=extra_path,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["startup"] == [], f"loaded before any import: {report['startup']}"
    inside = set(report["inside"]) - {""}
    bad = [
        (module, importer)
        for module, importer in report["seen"]
        if importer not in inside and not importer.endswith(BENIGN_IMPORTERS.get(module, ()))
    ]
    return report["imported"], bad


def test_orchestrator_never_imports_networking_modules() -> None:
    imported, bad = forbidden_imports("*")
    assert "asif.cli" in imported
    assert "asif.pipeline.local" in imported
    assert "asif.infer.client" in imported
    assert bad == [], (
        f"orchestrator code pulled in networking modules (module, importer): {bad}. "
        "Move network use into a separate, sandboxed process."
    )


def test_the_probe_catches_an_offender_hiding_behind_a_benign_import(tmp_path: Path) -> None:
    # Guards the guard. asif.config loads pydantic first, which on Python 3.12
    # already pulls in socket harmlessly; the offender's own import must still
    # be reported.
    (tmp_path / "offender.py").write_text("import asif.config\nimport socket\n")
    _, bad = forbidden_imports("offender", extra_path=str(tmp_path))
    assert ("socket", str(tmp_path / "offender.py")) in bad


_DYNAMIC_IMPORTERS = {"import_module", "__import__"}


def _imports_in(tree: ast.AST, package: str = "asif") -> tuple[set[str], list[str]]:
    """Every module a source file in ``package`` can import, and its dynamic imports."""
    names: set[str] = set()
    dynamic: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # "from . import worker" inside asif/infer/ means asif.infer.worker.
            base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called in _DYNAMIC_IMPORTERS:
                name = _named_import(node, called)
                if name is None:
                    dynamic.append(f"line {node.lineno}")
                else:
                    names.add(name)
    return names, dynamic


def _named_import(call: ast.Call, called: str) -> str | None:
    """The module an import_module() or __import__() call names, or None when
    the source doesn't spell it out."""
    arg = call.args[0] if call.args else None
    if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
        return None
    # __import__("pkg", globals(), None, ["mod"]) imports pkg.mod as well, and
    # a level makes the name relative: only the plain form names its module.
    if called == "__import__" and (call.keywords or not all(map(_empty, call.args[3:]))):
        return None
    return arg.value


def _empty(node: ast.expr) -> bool:
    """A literal empty fromlist, or level 0."""
    if isinstance(node, ast.List | ast.Tuple):
        return not node.elts
    return isinstance(node, ast.Constant) and not node.value


def _is_under(name: str, roots: Iterable[str]) -> bool:
    return any(name == r or name.startswith(r + ".") for r in roots)


def source_check(root: Path, top: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Read every source file of package ``top`` (in ``root``).

    Returns the files' own offending imports, and what importing everything
    they name from outside ``top`` brings in, as ``forbidden_imports`` does.
    """
    offenders = []
    outside: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        module = ".".join((top, *path.relative_to(root).with_suffix("").parts))
        if module.removesuffix(".__init__") in NETWORK_CAPABLE:
            continue
        package = (
            module.removesuffix(".__init__")
            if path.name == "__init__.py"
            else (module.rpartition(".")[0])
        )
        names, dynamic = _imports_in(ast.parse(path.read_text(encoding="utf-8")), package)
        where = path.relative_to(root)
        offenders += [f"{where}: an import the source doesn't spell out ({d})" for d in dynamic]
        for name in sorted(names):
            if _is_under(name, FORBIDDEN) or _is_under(name, NETWORK_CAPABLE):
                offenders.append(f"{where} imports {name}")
            elif not _is_under(name, (top,)):
                # Exactly what the source names: "import logging.handlers"
                # loads socket, and logging alone doesn't.
                outside.add(name)
    _, bad = forbidden_imports(*sorted(outside))
    return offenders, bad


def test_no_orchestrator_source_imports_networking_modules() -> None:
    # Catches what importing can't: imports inside functions that haven't
    # run, dynamic imports, and lazy imports of a network-capable module.
    # And everything those files could ever import from outside As If, even
    # lazily, mustn't pull in networking either.
    offenders, bad = source_check(Path(asif.__file__).parent, "asif")
    assert offenders == []
    assert bad == []


@pytest.mark.parametrize(
    ("source", "importer"),
    [
        (
            "def open_log(path):\n"
            "    import logging.handlers\n"
            "    return logging.handlers.RotatingFileHandler(path)\n",
            os.path.join("logging", "handlers.py"),
        ),
        # An attribute that loads its module on first use.
        (
            "def pool():\n"
            "    from concurrent.futures import ProcessPoolExecutor\n"
            "    return ProcessPoolExecutor()\n",
            os.path.join("multiprocessing", "connection.py"),
        ),
    ],
    ids=["submodule", "lazy-attribute"],
)
def test_the_source_check_imports_what_the_source_names(
    tmp_path: Path, source: str, importer: str
) -> None:
    # Guards the guard. Importing just logging or concurrent loads no
    # networking module; what the source names does, once it runs.
    package = tmp_path / "lazy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "later.py").write_text(source)
    offenders, bad = source_check(package, "lazy")
    assert offenders == []
    assert any(module == "socket" and path.endswith(importer) for module, path in bad), bad


def test_the_source_check_sees_lazy_and_dynamic_imports() -> None:
    sample = ast.parse(
        "def later():\n"
        "    import socket\n"
        "    from asif.infer import worker\n"
        "    importlib.import_module('http.client')\n"
        "    __import__(name)\n"
        "    __import__('asif.infer', globals(), None, ['worker'])\n"
        "    __import__('urllib', fromlist=['request'])\n"
        "    __import__('json', globals(), locals(), [], 0)\n"
    )
    names, dynamic = _imports_in(sample)
    assert {"socket", "asif.infer.worker", "http.client", "json"} <= names
    # A fromlist imports modules the call doesn't name.
    assert dynamic == ["line 5", "line 6", "line 7"]
    relative = ast.parse("def later():\n    from . import worker\n    from .worker import main\n")
    names, _ = _imports_in(relative, "asif.infer")
    assert {"asif.infer.worker", "asif.infer.worker.main"} <= names


def test_pydantic_plugins_are_disabled_in_the_orchestrator() -> None:
    # Plugins are third-party entry-point code that pydantic would otherwise
    # run inside this process.
    proc = _run("import asif, os; print(os.environ.get('PYDANTIC_DISABLE_PLUGINS'))")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "__all__"


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
