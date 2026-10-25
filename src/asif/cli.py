"""The ``asif`` command line.

Every command runs in the orchestrator process, which has no network
capability: ``main()`` installs the netguard tripwire before anything else.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from asif import __version__, netguard, paths, proc
from asif.config import Config, ConfigError, load_config
from asif.infer.client import InferenceError
from asif.pipeline.states import State

if TYPE_CHECKING:
    from asif.pipeline.local import ItemOutcome, RunSummary

app = typer.Typer(
    name="asif",
    help="As If: private, offline AI captions for your photo library.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
models_app = typer.Typer(help="The model catalogue.", no_args_is_help=True)
config_app = typer.Typer(help="Where your settings live and what they are.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")

console = Console()
err = Console(stderr=True)

EXIT_ITEMS_FAILED = 1
EXIT_USAGE = 2
EXIT_ABORTED = 3
EXIT_INTERRUPTED = 130  # the shell convention for "stopped by Ctrl+C"


class _State:
    config_path: Path | None = None


class Compat(StrEnum):
    xmp = "xmp"
    wide = "wide"


_state = _State()


def _load() -> Config:
    if _state.config_path is not None and not _state.config_path.is_file():
        err.print(f"[red]Config problem:[/] {escape(str(_state.config_path))} doesn't exist.")
        raise typer.Exit(EXIT_USAGE)
    try:
        return load_config(_state.config_path)
    except ConfigError as exc:
        err.print(f"[red]Config problem:[/] {escape(str(exc))}")
        raise typer.Exit(EXIT_USAGE) from exc


@app.callback()
def _root(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Use this config file instead of the default one."),
    ] = None,
) -> None:
    _state.config_path = config


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"As If {__version__}")


@app.command()
def doctor(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Check this machine: specs, tools, swap, disk encryption, and a model suggestion."""
    from asif.doctor import render_report, report_to_dict, run_doctor

    report = run_doctor(_load())
    if as_json:
        console.print_json(json.dumps(report_to_dict(report)))
    else:
        render_report(report, console)
    if any(check.status == "fail" for check in report.checks):
        raise typer.Exit(EXIT_USAGE)


@app.command()
def caption(
    source: Annotated[Path, typer.Argument(help="Folder with your photos. Only ever read.")],
    dest: Annotated[Path, typer.Argument(help="Folder for the captioned copies.")],
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="'stub' (fake captions, to try the pipeline), 'local' (the --gguf/--mmproj "
            "files), or a model id from `asif models list`.",
        ),
    ] = None,
    gguf: Annotated[
        Path | None, typer.Option(help="The model weights (.gguf).", show_default=False)
    ] = None,
    mmproj: Annotated[
        Path | None,
        typer.Option(help="The vision projector (.gguf) for those weights.", show_default=False),
    ] = None,
    llama_server: Annotated[
        str | None, typer.Option(help="Path to llama-server, if it isn't on your PATH.")
    ] = None,
    lang: Annotated[
        str | None, typer.Option(help="Caption language as a BCP-47 tag, like en or pt-BR.")
    ] = None,
    compat: Annotated[
        Compat | None,
        typer.Option(
            help="xmp (default), or wide to also write IPTC keywords for older apps "
            "(JPEG and TIFF only, and only where existing IPTC is UTF-8)."
        ),
    ] = None,
    max_edge: Annotated[
        int | None, typer.Option(help="Longest edge, in pixels, the model gets to see.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the captions; write nothing.")
    ] = False,
    limit: Annotated[int | None, typer.Option(min=1, help="Only do the first N photos.")] = None,
    allow_disk_workdir: Annotated[
        bool,
        typer.Option(
            "--allow-disk-workdir",
            help="If no RAM-backed folder exists, work on photos in an on-disk temp folder "
            "instead of refusing. Weaker: copies can survive in disk blocks.",
        ),
    ] = False,
) -> None:
    """Caption the photos in SOURCE into copies in DEST. Originals are never modified.

    Each photo is copied into RAM, described by a local model, tagged with
    exiftool, and checked by every integrity gate before its copy is written.
    """
    from pydantic import ValidationError

    from asif.pipeline.local import CaptionError, caption_folder, choose_model, iter_problems
    from asif.pipeline.timings import TimingsLog, new_run_id, prune

    cfg = _load()
    try:
        if model is not None:
            cfg.model.id = model
        if gguf is not None:
            cfg.model.gguf = str(gguf)
            if model is None and cfg.model.id == "auto":
                cfg.model.id = "local"
        if mmproj is not None:
            cfg.model.mmproj = str(mmproj)
        if llama_server is not None:
            cfg.model.llama_server = llama_server
        if lang is not None:
            cfg.general.caption_lang = lang
        if compat is not None:
            cfg.general.compat = compat.value
        if max_edge is not None:
            cfg.resources.max_edge_px = max_edge
        if allow_disk_workdir:
            cfg.resources.allow_disk_workdir = True
    except ValidationError as exc:
        problem = exc.errors()[0]
        err.print(f"[red]Invalid option:[/] {escape(str(problem['msg']))}")
        raise typer.Exit(EXIT_USAGE) from exc

    try:
        choice = choose_model(cfg, notice=lambda msg: err.print(f"[dim]{escape(msg)}[/]"))
    except CaptionError as exc:
        err.print(f"[red]Can't start:[/] {escape(str(exc))}")
        raise typer.Exit(EXIT_USAGE) from exc

    if choice.backend == "stub":
        err.print(
            Panel(
                "The [bold]stub[/] model makes fake captions from each photo's colours. "
                "Use it to try the pipeline, not on photos you care about.\n"
                "Every stub caption carries the keyword [bold]stub[/], so it's easy to find.",
                title="Stub model",
                border_style="yellow",
            )
        )

    run_id = new_run_id()
    log_path: Path | None = None
    if cfg.telemetry.enabled and not dry_run:
        log_path = paths.state_dir() / "timings.jsonl"
        prune(log_path, cfg.telemetry.retention_days)
    log = TimingsLog(log_path, run_id=run_id, model=choice.model_id)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
        transient=True,
    )
    task_id = progress.add_task("Loading model", total=None)

    def on_start(total: int) -> None:
        progress.update(task_id, total=total)

    def on_progress(_outcome: ItemOutcome, index: int, total: int) -> None:
        progress.update(task_id, completed=index, description="Captioning")

    notices: list[str] = []
    try:
        with progress:
            summary = caption_folder(
                source,
                dest,
                cfg,
                model=choice,
                run_id=run_id,
                log=log,
                dry_run=dry_run,
                limit=limit,
                on_progress=on_progress,
                on_start=on_start,
                notice=notices.append,
            )
    except KeyboardInterrupt as exc:
        err.print("Stopped before any photo was started.")
        raise typer.Exit(EXIT_INTERRUPTED) from exc
    except CaptionError as exc:
        err.print(f"[red]Can't start:[/] {escape(str(exc))}")
        if "RAM-backed" in str(exc):
            err.print(
                "As If keeps photos in RAM while it works on them. On Linux that needs a "
                "tmpfs (/dev/shm or $XDG_RUNTIME_DIR); you can point [bold]work_dir[/] under "
                "[resources] at one. To accept the weaker option, pass "
                "[bold]--allow-disk-workdir[/]."
            )
        raise typer.Exit(EXIT_USAGE) from exc
    except InferenceError as exc:
        err.print(f"[red]The model couldn't start[/] ({exc.kind}).")
        if exc.detail:
            err.print(
                Panel(escape(exc.detail[-2000:]), title="model log (tail)", border_style="red")
            )
        err.print("Run [bold]asif doctor[/] to check llama-server and your model files.")
        raise typer.Exit(EXIT_USAGE) from exc

    for message in notices:
        err.print(f"[yellow]Note:[/] {escape(message)}")
    _print_summary(summary, list(iter_problems(summary)))
    if summary.interrupted:
        raise typer.Exit(EXIT_INTERRUPTED)
    if summary.aborted:
        raise typer.Exit(EXIT_ABORTED)
    if summary.failed or summary.unreadable_dirs:
        raise typer.Exit(EXIT_ITEMS_FAILED)


def _print_summary(summary: RunSummary, problems: list[ItemOutcome]) -> None:
    if summary.dry_run:
        table = Table(title="Dry run: what would be written", show_lines=True)
        table.add_column("Photo", overflow="fold")
        table.add_column("Caption")
        table.add_column("New keywords")
        for o in summary.outcomes:
            if o.plan is None:
                continue
            caption_cell = (
                escape(o.plan.caption)
                if o.plan.caption is not None
                else "[dim](your caption is kept)[/]"
            )
            keywords = ", ".join(o.plan.add_subjects) or "[dim](none new)[/]"
            table.add_row(escape(o.rel), caption_cell, escape(keywords))
        if table.row_count:
            console.print(table)

    kept = sum(
        1
        for o in summary.outcomes
        if o.state in (State.CLEAN, State.DESCRIBED) and o.reason.startswith("kept")
    )
    labels: list[tuple[State, str, str]] = [
        (State.CLEAN, "captioned", "green"),
        (State.DESCRIBED, "described (dry run)", "green"),
        (State.SKIPPED_DONE, "already done", "dim"),
        (State.SKIPPED_FORMAT, "format not supported yet", "yellow"),
        (State.SKIPPED_CONFLICT, "skipped: conflict", "yellow"),
        (State.FAILED_GATE, "refused by a gate", "red"),
        (State.FAILED_UPLOAD, "write failed", "red"),
        (State.QUARANTINE, "quarantined", "red"),
    ]
    counts = Table.grid(padding=(0, 2))
    for state, label, style in labels:
        n = summary.count(state)
        if n:
            counts.add_row(f"[{style}]{n}[/]", label)
    if kept:
        counts.add_row(f"[dim]{kept}[/]", "of those kept your own caption (keywords only)")
    if summary.ignored:
        counts.add_row(
            f"[dim]{summary.ignored}[/]",
            "other files ignored (not photos, hidden, links or special files)",
        )
    if not summary.outcomes:
        if summary.aborted:  # always set when a run is stopped early
            console.print("No photo was finished before the run stopped.")
        else:
            console.print("No photos found.")
    else:
        console.print(Panel(counts, title=f"Run {summary.run_id}", expand=False))

    if problems or summary.unreadable_dirs:
        table = Table(title="Needs a look")
        table.add_column("Photo", overflow="fold")
        table.add_column("What happened")
        for folder in summary.unreadable_dirs:
            table.add_row(escape(folder + "/"), "folder couldn't be read (permission denied?)")
        for item in problems:
            table.add_row(escape(item.rel), escape(item.reason))
        console.print(table)
    notes = [o for o in summary.outcomes if o.note]
    if notes:
        table = Table(title="Notes")
        table.add_column("Photo", overflow="fold")
        table.add_column("Note")
        for item in notes:
            table.add_row(escape(item.rel), escape(item.note))
        console.print(table)
    if summary.aborted:
        err.print(f"[red bold]Run stopped:[/] {escape(summary.aborted)}")


@models_app.command("list")
def models_list() -> None:
    """Show the models As If knows about."""
    from asif.models.manifest import load_manifest

    table = Table(title="Model catalogue (§9 of the design)")
    for col in ("id", "tier", "size", "languages", "licence", "pinned"):
        table.add_column(col)
    for m in load_manifest():
        langs = "many" if m.languages == ("*",) else ", ".join(m.languages)
        table.add_row(
            m.id,
            m.tier,
            f"{m.total_gb:.1f} GB",
            langs,
            m.licence,
            "yes" if m.pinned else "[yellow]not yet[/]",
        )
    console.print(table)
    console.print("[dim]Downloading and verifying models arrives in M2 (`asif models pull`).[/]")


@models_app.command("recommend")
def models_recommend() -> None:
    """Suggest a model for this machine, and say why."""
    from asif.doctor import detect_specs, recommend_for

    cfg = _load()
    specs = detect_specs(cfg)
    try:
        rec = recommend_for(specs, cfg.general.caption_lang)
    except ValueError as exc:
        err.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(EXIT_ITEMS_FAILED) from exc
    console.print(f"[bold]{escape(rec.model_name)}[/] ({rec.model_id}) on {rec.device}")
    console.print(escape(rec.reason))
    for alt in rec.alternatives:
        console.print(f"  • {escape(alt)}")


@config_app.command("path")
def config_path() -> None:
    """Print where the config file is (or would be)."""
    console.print(str(_state.config_path or paths.config_file()))


@config_app.command("show")
def config_show() -> None:
    """Print the settings in effect (defaults plus your config file)."""
    console.print_json(json.dumps(_load().model_dump(mode="json")))


@config_app.command("init")
def config_init(
    force: Annotated[bool, typer.Option(help="Overwrite an existing config file.")] = False,
) -> None:
    """Write a commented starter config file."""
    target = _state.config_path or paths.config_file()
    if target.exists() and not force:
        err.print(f"{target} already exists. Use --force to replace it.")
        raise typer.Exit(EXIT_USAGE)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(STARTER_CONFIG, encoding="utf-8")
    console.print(f"Wrote {target}")


STARTER_CONFIG = """\
# As If configuration. Every setting is optional; these are the defaults.
# Full reference: docs/DESIGN.md, section 17.

[general]
caption_lang = "en"     # BCP-47 tag: "en", "pt-BR", "de", "ja"...
compat       = "xmp"    # "xmp", or "wide" to also write IPTC keywords for older apps

[resources]
max_edge_px = 1024      # longest edge the model sees; lower is faster on CPU
# work_dir  = ""        # RAM work dir; empty = auto-detect a tmpfs
# allow_disk_workdir = false  # true = use an on-disk folder if there's no tmpfs (weaker)

[model]
id     = "auto"         # "auto", "local", "stub", or an id from `asif models list`
device = "auto"         # "auto", "cpu", "gpu" or "hybrid"
# Until `asif models pull` exists (M2), point at files you downloaded:
# gguf         = "~/models/your-model.Q4_K_M.gguf"
# mmproj       = "~/models/your-model-mmproj-F16.gguf"
# llama_server = "llama-server"  # or a script that ends with: exec llama-server "$@"
# threads      = 0      # 0 = let llama.cpp decide
# server_args  = []     # extra llama-server options: speed, memory, GPU, logging only

[guards]
max_failure_rate = 0.02 # stop if more than 2% of the last 200 photos fail

[telemetry]
enabled        = true   # local timings log only; never leaves this machine
retention_days = 90
"""


def main() -> None:
    """Entry point for the ``asif`` script."""
    # Before anything reads a photo: no core files with pixels in them (for
    # this process and every child), and stop signals that unwind cleanly.
    proc.disable_core_dumps()
    proc.install_stop_signals()
    netguard.install()
    app()
