"""``asif doctor``: what this machine is, what's installed, what's risky.

Read-only. Never needs root, never touches the network, never crashes:
anything it can't detect is reported as unknown.

Every observation goes through a ``Probe`` (read a file, run a program, look
at the environment), so tests can describe a whole machine (a LUKS laptop on
battery, a Mac, a WSL box with an NVIDIA card) without having one.
``SystemProbe`` is the real thing.

The checks follow the threat model (§3 of docs/DESIGN.md): plaintext photos
live only in RAM, so swap has to be zram or encrypted, the work dir has to be
a tmpfs, and the disk underneath should be encrypted anyway. Every check that
isn't plainly fine comes with a hint a newcomer can act on.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from asif.config import Config
from asif.models.recommend import Recommendation, recommend
from asif.workdir import find_ram_dir

MIB = 1024 * 1024
LOW_DISK_MB = 10 * 1024
"""Below this much free space at the staging path, batches get too small to be useful (§11)."""

# CPU features that decide how fast llama.cpp runs on the CPU.
INTERESTING_FLAGS = ("avx", "avx2", "avx512f", "f16c", "fma", "neon", "asimd")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# -- probing -------------------------------------------------------------------


class Probe(Protocol):
    """Everything the doctor learns about the machine goes through here."""

    def read_text(self, path: str) -> str | None:
        """File contents, or None if it can't be read."""
        ...

    def run(self, argv: list[str], timeout: float = 5.0) -> tuple[int, str] | None:
        """(exit code, stdout+stderr), or None if the program is missing or timed out."""
        ...

    def which(self, name: str) -> str | None: ...

    def env(self, name: str) -> str | None: ...

    def platform(self) -> str:
        """``sys.platform``: "linux", "darwin", "win32"..."""
        ...

    def machine(self) -> str:
        """CPU architecture: "x86_64", "arm64", "aarch64"..."""
        ...

    def disk_free_mb(self, path: Path) -> int | None:
        """Free space on the filesystem holding ``path`` (or its nearest existing parent)."""
        ...

    def listdir(self, path: str) -> list[str]:
        """Entry names, sorted; [] if the directory can't be listed."""
        ...

    def realpath(self, path: str) -> str: ...

    def stat_dev(self, path: str) -> tuple[int, int] | None:
        """(major, minor) of the device the filesystem holding ``path`` lives on."""
        ...


class SystemProbe:
    """The real machine. Every method turns errors into "unknown"."""

    def read_text(self, path: str) -> str | None:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return None

    def run(self, argv: list[str], timeout: float = 5.0) -> tuple[int, str] | None:
        try:
            proc = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                # The C locale keeps tool output parseable whatever the user's language.
                env={**os.environ, "LC_ALL": "C"},
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.returncode, proc.stdout.decode("utf-8", "replace")

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def env(self, name: str) -> str | None:
        return os.environ.get(name)

    def platform(self) -> str:
        return sys.platform

    def machine(self) -> str:
        return platform.machine()

    def disk_free_mb(self, path: Path) -> int | None:
        for candidate in (path, *path.parents):
            try:
                return shutil.disk_usage(candidate).free // MIB
            except OSError:
                continue
        return None

    def listdir(self, path: str) -> list[str]:
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []

    def realpath(self, path: str) -> str:
        return os.path.realpath(path)

    def stat_dev(self, path: str) -> tuple[int, int] | None:
        try:
            dev = os.stat(path).st_dev
            return os.major(dev), os.minor(dev)
        except (OSError, AttributeError):
            return None


# -- report types --------------------------------------------------------------

SwapKind = Literal["none", "zram", "encrypted", "plain", "mixed", "unknown"]
CheckStatus = Literal["ok", "warn", "fail", "info"]


@dataclass(frozen=True)
class GpuInfo:
    vendor: str
    """"nvidia", "amd" or "apple"."""
    name: str
    vram_total_mb: int | None
    vram_free_mb: int | None


@dataclass(frozen=True)
class SwapInfo:
    kind: SwapKind
    """zram and encrypted are safe. "mixed" means some swap is plain and some isn't."""
    detail: str


@dataclass(frozen=True)
class PowerInfo:
    on_ac: bool | None
    has_battery: bool | None


@dataclass(frozen=True)
class Specs:
    os_name: str
    os_release: str | None
    is_wsl: bool
    arch: str
    cpu_model: str | None
    cores_logical: int | None
    cores_physical: int | None
    cpu_flags: list[str]
    ram_total_mb: int | None
    ram_available_mb: int | None
    gpus: list[GpuInfo]
    apple_silicon: bool
    unified_memory_mb: int | None
    swap: SwapInfo
    power: PowerInfo
    staging_path: str
    disk_free_mb: int | None
    root_encrypted: bool | None
    ram_dir: str
    ram_dir_in_ram: bool


@dataclass(frozen=True)
class Dependency:
    name: str
    found: bool
    path: str | None
    version: str | None
    required: bool
    purpose: str


@dataclass(frozen=True)
class Check:
    name: str
    status: CheckStatus
    summary: str
    hint: str
    """What to do about it, in plain words. Empty only for "ok"."""


@dataclass(frozen=True)
class DoctorReport:
    specs: Specs
    deps: list[Dependency]
    checks: list[Check]
    recommendation: Recommendation | None


# -- specs ---------------------------------------------------------------------


def detect_specs(cfg: Config, probe: Probe | None = None) -> Specs:
    """Describe this machine. Linux and WSL in full, macOS via sysctl/pmset, else unknown."""
    probe = probe or SystemProbe()
    plat = probe.platform()
    arch = probe.machine()
    staging = cfg.resources.staging_path()
    ram_dir = find_ram_dir(cfg.resources.work_dir)

    is_wsl = False
    gpus: list[GpuInfo] = []
    apple = False
    root_encrypted: bool | None = None
    if plat.startswith("linux"):
        os_name, os_release = "Linux", _linux_release(probe)
        is_wsl = _is_wsl(probe)
        cpu_model, logical, physical, flags = _linux_cpu(probe)
        ram_total, ram_available = _linux_memory(probe)
        gpus = _nvidia_gpus(probe) + _amd_gpus(probe)
        swap = _linux_swap(probe)
        power = _linux_power(probe, is_wsl)
        # WSL's disk is a file on the Windows side; BitLocker is invisible from here.
        root_encrypted = None if is_wsl else _linux_root_encrypted(probe)
    elif plat == "darwin":
        os_name, os_release = "macOS", _output(probe, ["sw_vers", "-productVersion"])
        cpu_model, logical, physical, flags = _mac_cpu(probe)
        ram_total, ram_available = _mac_memory(probe)
        # Rosetta reports x86_64 to a translated Python; hw.optional.arm64 tells the truth.
        apple = arch in ("arm64", "aarch64") or _sysctl(probe, "hw.optional.arm64") == "1"
        if apple:
            gpus = [GpuInfo("apple", f"{cpu_model or 'Apple Silicon'} GPU (Metal)", None, None)]
        swap = SwapInfo("encrypted", "macOS always encrypts swap")
        power = _mac_power(probe)
        root_encrypted = _mac_filevault(probe)
    else:
        os_name = "Windows" if plat == "win32" else plat
        os_release, cpu_model, logical, physical, flags = None, None, None, None, []
        ram_total = ram_available = None
        swap = SwapInfo("unknown", "not checked on this platform")
        power = PowerInfo(None, None)

    return Specs(
        os_name=os_name,
        os_release=os_release,
        is_wsl=is_wsl,
        arch=arch,
        cpu_model=cpu_model,
        cores_logical=logical,
        cores_physical=physical,
        cpu_flags=flags,
        ram_total_mb=ram_total,
        ram_available_mb=ram_available,
        gpus=gpus,
        apple_silicon=apple,
        unified_memory_mb=ram_total if apple else None,
        swap=swap,
        power=power,
        staging_path=str(staging),
        disk_free_mb=probe.disk_free_mb(staging),
        root_encrypted=root_encrypted,
        ram_dir=str(ram_dir.path),
        ram_dir_in_ram=ram_dir.in_ram,
    )


def _read(probe: Probe, path: str) -> str:
    """Stripped file contents, "" if unreadable. Handy for one-value sysfs files."""
    return (probe.read_text(path) or "").strip()


def _to_int(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _interesting(flags: set[str]) -> list[str]:
    return [f for f in INTERESTING_FLAGS if f in flags]


def _is_wsl(probe: Probe) -> bool:
    version = (probe.read_text("/proc/version") or "").lower()
    return "microsoft" in version or "wsl" in version or bool(probe.env("WSL_DISTRO_NAME"))


def _linux_release(probe: Probe) -> str | None:
    pretty = None
    for line in (probe.read_text("/etc/os-release") or "").splitlines():
        if line.startswith("PRETTY_NAME="):
            pretty = line.split("=", 1)[1].strip().strip("\"'") or None
    kernel = _read(probe, "/proc/sys/kernel/osrelease")
    parts = [p for p in (pretty, f"kernel {kernel}" if kernel else None) if p]
    return ", ".join(parts) or None


def _linux_cpu(probe: Probe) -> tuple[str | None, int | None, int | None, list[str]]:
    text = probe.read_text("/proc/cpuinfo") or ""
    first: dict[str, str] = {}
    logical = 0
    cores: set[tuple[str, str]] = set()
    flags: set[str] = set()
    for block in re.split(r"\n\s*\n", text):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                fields[key.strip().lower()] = value.strip()
        for key, value in fields.items():
            first.setdefault(key, value)
        if "processor" in fields and fields["processor"].isdigit():
            logical += 1
        if "physical id" in fields and "core id" in fields:
            cores.add((fields["physical id"], fields["core id"]))
        # x86 calls them "flags", ARM "features".
        flags.update(fields.get("flags", fields.get("features", "")).split())

    # x86 has "model name"; ARM boards have at most "Model" or "Hardware", and
    # x86's own "model" is a number, so it doesn't count.
    model = first.get("model name")
    if not model and not first.get("model", "").isdigit():
        model = first.get("model")
    model = model or first.get("hardware")
    if not model:
        model = _read(probe, "/proc/device-tree/model").rstrip("\0") or None
    return model, logical or None, len(cores) or None, _interesting(flags)


def _linux_memory(probe: Probe) -> tuple[int | None, int | None]:
    kb: dict[str, int] = {}
    for line in (probe.read_text("/proc/meminfo") or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            kb[key.strip()] = int(parts[0])
    total = kb.get("MemTotal")
    if "MemAvailable" in kb:
        available: int | None = kb["MemAvailable"]
    elif "MemFree" in kb:  # kernels before 3.14
        available = kb["MemFree"] + kb.get("Buffers", 0) + kb.get("Cached", 0)
    else:
        available = None
    return (
        total // 1024 if total is not None else None,
        available // 1024 if available is not None else None,
    )


def _nvidia_gpus(probe: Probe) -> list[GpuInfo]:
    out = probe.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if out is None or out[0] != 0:
        return []
    gpus = []
    for line in out[1].splitlines():
        parts = [p.strip() for p in line.rsplit(",", 2)]
        if len(parts) == 3 and parts[0]:
            gpus.append(GpuInfo("nvidia", parts[0], _to_int(parts[1]), _to_int(parts[2])))
    return gpus


def _amd_gpus(probe: Probe) -> list[GpuInfo]:
    out = probe.run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if out is None or out[0] != 0 or "{" not in out[1]:
        return []
    try:
        # rocm-smi sometimes prints warnings before the JSON.
        data, _ = json.JSONDecoder().raw_decode(out[1][out[1].index("{") :])
    except ValueError:
        return []
    if not isinstance(data, dict):
        return []
    gpus = []
    for key, card in data.items():
        if not str(key).lower().startswith("card") or not isinstance(card, dict):
            continue
        # Key spelling varies between ROCm releases ("Card series" vs "Card Series").
        fields = {str(k).strip().lower(): str(v).strip() for k, v in card.items()}
        name = fields.get("card series") or fields.get("card model") or str(key)
        total = _to_int(fields.get("vram total memory (b)"))
        used = _to_int(fields.get("vram total used memory (b)"))
        gpus.append(
            GpuInfo(
                "amd",
                name,
                total // MIB if total is not None else None,
                (total - used) // MIB if total is not None and used is not None else None,
            )
        )
    return gpus


# Swap and disk encryption both come down to one question: is this block
# device dm-crypt, or stacked on top of dm-crypt (LVM inside LUKS)?


def _on_dm_crypt(probe: Probe, devnum: str, depth: int = 0) -> bool:
    base = f"/sys/dev/block/{devnum}"
    if _read(probe, f"{base}/dm/uuid").startswith("CRYPT-"):
        return True
    if depth >= 8:  # device stacks are shallow; this only guards against loops
        return False
    for lower in probe.listdir(f"{base}/slaves"):
        sub = _read(probe, f"/sys/class/block/{lower}/dev")
        if sub and _on_dm_crypt(probe, sub, depth + 1):
            return True
    return False


def _devnum_of_block(probe: Probe, device: str) -> str | None:
    """ "major:minor" of a block device node like /dev/mapper/cryptswap."""
    name = os.path.basename(probe.realpath(device))
    return _read(probe, f"/sys/class/block/{name}/dev") or None


def _mount_source(probe: Probe, path: str) -> str | None:
    """The device /proc/mounts lists for the mount holding ``path``."""
    target = PurePosixPath(path)
    best: tuple[int, str] | None = None
    for line in (probe.read_text("/proc/mounts") or "").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        mount_point = PurePosixPath(parts[1].replace("\\040", " "))
        if (target == mount_point or mount_point in target.parents) and (
            best is None or len(mount_point.parts) >= best[0]
        ):
            best = (len(mount_point.parts), parts[0])
    return best[1] if best else None


def _devnum_of_path(probe: Probe, path: str) -> str | None:
    """ "major:minor" of the block device under the filesystem holding ``path``."""
    path = probe.realpath(path)
    dev = probe.stat_dev(path)
    if dev is not None and dev[0] != 0:
        return f"{dev[0]}:{dev[1]}"
    # Major 0 is an anonymous device (btrfs, overlayfs, ZFS...): ask /proc/mounts
    # which block device backs it instead.
    source = _mount_source(probe, path)
    if source is None or not source.startswith("/dev/"):
        return None
    return _devnum_of_block(probe, source)


def _swap_entry_kind(probe: Probe, name: str, swap_type: str) -> SwapKind:
    if swap_type == "partition":
        if os.path.basename(probe.realpath(name)).startswith("zram"):
            return "zram"
        devnum = _devnum_of_block(probe, name)
    else:
        # A swap file is as safe as the filesystem it lives on.
        devnum = _devnum_of_path(probe, name)
    if devnum is None:
        return "unknown"
    return "encrypted" if _on_dm_crypt(probe, devnum) else "plain"


def _linux_swap(probe: Probe) -> SwapInfo:
    text = probe.read_text("/proc/swaps")
    if text is None:
        return SwapInfo("unknown", "couldn't read /proc/swaps")
    entries: list[tuple[str, SwapKind]] = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].replace("\\040", " ").removesuffix(" (deleted)")
        entries.append((name, _swap_entry_kind(probe, name, parts[1])))
    if not entries:
        return SwapInfo("none", "no swap configured")
    kinds = {kind for _, kind in entries}
    kind: SwapKind
    if "plain" in kinds:
        kind = "plain" if kinds == {"plain"} else "mixed"
    elif "unknown" in kinds:
        kind = "unknown"
    elif kinds == {"zram"}:
        kind = "zram"
    else:
        kind = "encrypted"  # zram and dm-crypt together are both fine
    return SwapInfo(kind, ", ".join(f"{name}: {k}" for name, k in entries))


def _linux_root_encrypted(probe: Probe) -> bool | None:
    devnum = _devnum_of_path(probe, "/")
    return None if devnum is None else _on_dm_crypt(probe, devnum)


def _linux_power(probe: Probe, is_wsl: bool) -> PowerInfo:
    root = "/sys/class/power_supply"
    names = probe.listdir(root)
    if not names:
        # A desktop or server has no power supplies listed and runs on mains;
        # WSL may simply not pass the laptop's battery through.
        return PowerInfo(None, None) if is_wsl else PowerInfo(True, False)
    mains: list[bool] = []
    statuses: list[str] = []
    for name in names:
        base = f"{root}/{name}"
        kind = _read(probe, f"{base}/type")
        if kind in ("Mains", "USB"):  # USB-C chargers report as USB
            mains.append(_read(probe, f"{base}/online") == "1")
        elif kind == "Battery":
            if _read(probe, f"{base}/scope") == "Device":  # a mouse or headset battery
                continue
            if _read(probe, f"{base}/present") == "0":
                continue
            statuses.append(_read(probe, f"{base}/status"))
    if not statuses:
        return PowerInfo(True, False)
    if any(mains):
        on_ac: bool | None = True
    elif "Discharging" in statuses or mains:
        on_ac = False
    elif all(s in ("Charging", "Full", "Not charging") for s in statuses):
        on_ac = True
    else:
        on_ac = None
    return PowerInfo(on_ac, True)


def _output(probe: Probe, argv: list[str]) -> str | None:
    """A command's trimmed output if it succeeded, else None."""
    out = probe.run(argv)
    if out is None or out[0] != 0:
        return None
    return out[1].strip() or None


def _sysctl(probe: Probe, key: str) -> str | None:
    return _output(probe, ["sysctl", "-n", key])


def _mac_cpu(probe: Probe) -> tuple[str | None, int | None, int | None, list[str]]:
    features = " ".join(
        _sysctl(probe, key) or "" for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features")
    )
    flags = {f.lower().replace("avx1.0", "avx") for f in features.split()}
    if _sysctl(probe, "hw.optional.neon") == "1":
        flags.add("neon")
    return (
        _sysctl(probe, "machdep.cpu.brand_string"),
        _to_int(_sysctl(probe, "hw.logicalcpu")),
        _to_int(_sysctl(probe, "hw.physicalcpu")),
        _interesting(flags),
    )


def _mac_memory(probe: Probe) -> tuple[int | None, int | None]:
    total = _to_int(_sysctl(probe, "hw.memsize"))
    out = probe.run(["vm_stat"])
    available = None
    if out is not None and out[0] == 0:
        page = re.search(r"page size of (\d+) bytes", out[1])
        pages = {
            m.group(1).lower(): int(m.group(2))
            for m in re.finditer(r"^Pages ([\w ]+?):\s+(\d+)\.?$", out[1], re.MULTILINE)
        }
        if page and "free" in pages:
            # Roughly what Activity Monitor calls available: free plus reclaimable.
            reclaimable = pages["free"] + pages.get("inactive", 0) + pages.get("speculative", 0)
            available = reclaimable * int(page.group(1)) // MIB
    return (total // MIB if total is not None else None), available


def _mac_power(probe: Probe) -> PowerInfo:
    out = probe.run(["pmset", "-g", "batt"])
    if out is None or out[0] != 0:
        return PowerInfo(None, None)
    text = out[1]
    on_ac: bool | None = None
    if "'AC Power'" in text:
        on_ac = True
    elif "'Battery Power'" in text or "'UPS Power'" in text:
        on_ac = False
    return PowerInfo(on_ac, "InternalBattery" in text)


def _mac_filevault(probe: Probe) -> bool | None:
    status = _output(probe, ["fdesetup", "status"]) or ""
    if "FileVault is On" in status:
        return True
    if "FileVault is Off" in status:
        return False
    return None


# -- dependencies --------------------------------------------------------------


def find_dependencies(cfg: Config, probe: Probe | None = None) -> list[Dependency]:
    """The external tools As If uses, whether they're installed, and their versions."""
    probe = probe or SystemProbe()
    # (display name, what to look up, version flags or None, required, purpose)
    tools: list[tuple[str, str, list[str] | None, bool, str]] = [
        (
            "exiftool",
            "exiftool",
            ["-ver"],
            True,
            "writes captions into your photos and proves the image data is untouched",
        ),
        (
            "llama-server",
            cfg.model.llama_server,
            ["--version"],
            True,
            "runs the AI model locally (part of llama.cpp); needed for real captions",
        ),
        ("bwrap", "bwrap", ["--version"], False, "walls the model off from the network (M3)"),
        ("age", "age", ["--version"], False, "encrypts downloaded photos while they wait (M4)"),
        ("rclone", "rclone", ["--version"], False, "optional: reaches clouds other than Proton"),
        # Not run: we don't know whether it phones home for updates, and the
        # doctor promises not to touch the network.
        ("proton-drive", "proton-drive", None, False, "optional: Proton Drive's official CLI"),
    ]
    deps = []
    for name, lookup, version_args, required, purpose in tools:
        path = probe.which(lookup)
        version = None
        if path is not None and version_args is not None:
            out = probe.run([path, *version_args], timeout=10.0)
            if out is not None:
                version = _first_line(out[1])
        deps.append(Dependency(name, path is not None, path, version, required, purpose))
    return deps


def _first_line(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:80]
    return None


# -- checks ----------------------------------------------------------------------

_EXIFTOOL_INSTALL = (
    "Install it: `sudo apt install libimage-exiftool-perl` (Debian, Ubuntu), "
    "`sudo dnf install perl-Image-ExifTool` (Fedora), `brew install exiftool` (macOS), "
    "or get it from exiftool.org. Then run `asif doctor` again."
)


def _exiftool_check(probe: Probe, dep: Dependency) -> Check:
    if not dep.found or dep.path is None:
        return Check(
            "exiftool",
            "fail",
            "exiftool isn't installed. As If needs it to write captions into photos.",
            _EXIFTOOL_INSTALL,
        )
    missing = _missing_exiftool_features(probe, dep.path)
    version = f"exiftool {dep.version}" if dep.version else "This exiftool"
    if missing:
        return Check(
            "exiftool",
            "fail",
            f"{version} can't {' or '.join(missing)}. As If needs that to prove "
            "it never touched your pictures.",
            "Install a current exiftool from exiftool.org or your package manager, "
            "then run `asif doctor` again.",
        )
    return Check(
        "exiftool",
        "ok",
        f"{version} can hash image data (SHA-256) and record that hash in the file.",
        "",
    )


def _missing_exiftool_features(probe: Probe, exe: str) -> list[str]:
    """Ask exiftool to do what the gates need, rather than trusting a version number."""
    from PIL import Image  # only the doctor's self-test needs it here

    missing = []
    hashed = None
    # A synthetic 16x16 image in a private (0700) temp dir, removed right after.
    with tempfile.TemporaryDirectory(prefix="asif-doctor-") as tmp:
        sample = Path(tmp) / "sample.jpg"
        try:
            Image.new("RGB", (16, 16), (90, 140, 200)).save(sample, "JPEG")
        except OSError:
            pass
        else:
            hashed = probe.run(
                [
                    exe,
                    "-api",
                    "requesttags=imagedatahash",
                    "-api",
                    "imagehashtype=SHA256",
                    "-s3",
                    "-ImageDataHash",
                    str(sample),
                ],
                timeout=20.0,
            )
    if (
        hashed is None
        or hashed[0] != 0
        or not any(_HEX64.match(line.strip()) for line in hashed[1].splitlines())
    ):
        missing.append("compute a SHA-256 ImageDataHash")
    listed = probe.run([exe, "-listw", "-XMP-et:all"], timeout=20.0)
    tags = set(listed[1].split()) if listed is not None and listed[0] == 0 else set()
    if not {"OriginalImageHash", "OriginalImageHashType"} <= tags:
        missing.append("write XMP-et:OriginalImageHash")
    return missing


def _llama_check(cfg: Config, dep: Dependency) -> Check:
    if dep.found:
        return Check("llama-server", "ok", f"llama-server found ({dep.version or dep.path}).", "")
    configured = cfg.model.llama_server
    where = "" if configured == "llama-server" else f" (looked for {configured!r})"
    return Check(
        "llama-server",
        "warn",
        f"llama-server (from llama.cpp) isn't installed{where}, so real captions can't run yet.",
        "Install llama.cpp: `brew install llama.cpp` on macOS; on Linux, your distro's "
        "package or a release build for your GPU from github.com/ggml-org/llama.cpp. "
        "If it's somewhere else, set llama_server under [model] to its full path. "
        "Meanwhile, `asif caption --model stub` tries the whole pipeline with fake captions.",
    )


def _encryption_check(specs: Specs) -> Check:
    mac = specs.os_name == "macOS"
    if specs.root_encrypted:
        what = "FileVault is on." if mac else "The system disk is encrypted."
        return Check("Disk encryption", "ok", what, "")
    if specs.root_encrypted is False:
        return Check(
            "Disk encryption",
            "warn",
            "FileVault is off." if mac else "The system disk doesn't look encrypted.",
            "Photos stay in RAM, but As If's threat model assumes full-disk encryption for "
            "everything else (vault, models, logs). "
            + (
                "Turn on FileVault in System Settings > Privacy & Security."
                if mac
                else "Turn on LUKS (most installers offer it). If you encrypt another way "
                "(ZFS native encryption, a self-encrypting drive), you can ignore this."
            ),
        )
    if specs.is_wsl:
        hint = (
            "WSL's disk lives on Windows: make sure BitLocker or Device Encryption is on "
            "for that drive (Windows Settings > Privacy & security)."
        )
    else:
        hint = (
            "Make sure full-disk encryption is on (LUKS on Linux, FileVault on macOS, "
            "BitLocker on Windows); As If's threat model assumes it."
        )
    return Check("Disk encryption", "info", "Couldn't tell whether the disk is encrypted.", hint)


_SWAP_HINT = (
    "Photos held in RAM could be paged out to disk unencrypted. Switch to zram "
    "(e.g. the zram-generator or zram-tools package) or put swap inside LUKS. "
    "As a stopgap, `sudo swapoff -a` turns swap off until the next reboot."
)


def _swap_check(swap: SwapInfo) -> Check:
    if swap.kind == "none":
        return Check("Swap", "ok", "No swap, so memory is never written to disk.", "")
    if swap.kind == "zram":
        return Check("Swap", "ok", "Swap is zram (compressed RAM); it never touches a disk.", "")
    if swap.kind == "encrypted":
        return Check("Swap", "ok", f"Swap is encrypted ({swap.detail}).", "")
    if swap.kind == "plain":
        return Check("Swap", "warn", f"Swap isn't encrypted ({swap.detail}).", _SWAP_HINT)
    if swap.kind == "mixed":
        return Check("Swap", "warn", f"Some swap isn't encrypted ({swap.detail}).", _SWAP_HINT)
    return Check(
        "Swap",
        "info",
        f"Couldn't tell whether swap is encrypted ({swap.detail}).",
        "Run `swapon --show` and `lsblk`: zram devices and swap on a crypt device are fine; "
        "anything else can write photos held in RAM to disk unencrypted.",
    )


def _ram_dir_check(specs: Specs) -> Check:
    if specs.ram_dir_in_ram:
        return Check(
            "RAM work dir", "ok", f"Photos being captioned live only in RAM ({specs.ram_dir}).", ""
        )
    if specs.os_name == "macOS":
        hint = (
            "macOS has no RAM folder by default; RAM-disk support arrives later (M10). "
            "With FileVault on, those files are at least encrypted on disk."
        )
    else:
        hint = (
            "Set work_dir under [resources] to a RAM-backed folder such as /dev/shm, "
            "or log in through a normal session so $XDG_RUNTIME_DIR exists."
        )
    return Check(
        "RAM work dir",
        "warn",
        f"The work dir ({specs.ram_dir}) is on disk, so photos being captioned touch the disk.",
        hint,
    )


def _disk_check(specs: Specs) -> Check:
    where = specs.staging_path
    if specs.disk_free_mb is None:
        return Check(
            "Staging disk",
            "info",
            f"Couldn't check free space for the staging folder ({where}).",
            "Make sure that drive has at least 10 GB free, or set staging_dir under "
            "[resources] to a folder on a drive that does.",
        )
    free = _gb(specs.disk_free_mb)
    if specs.disk_free_mb < LOW_DISK_MB:
        return Check(
            "Staging disk",
            "warn",
            f"Only {free} free for the staging folder ({where}).",
            "Downloaded photos wait there (encrypted) before captioning. Free up space, or "
            "set staging_dir under [resources] to a folder on a drive with 10 GB or more free.",
        )
    return Check("Staging disk", "ok", f"{free} free for the staging folder ({where}).", "")


def _power_check(cfg: Config, power: PowerInfo) -> Check:
    if power.on_ac is False:
        if cfg.guards.require_ac_power:
            return Check(
                "Power",
                "warn",
                "Running on battery, and As If is set to need mains power for long runs.",
                "Plug in the charger. To allow battery runs anyway, set "
                "require_ac_power = false under [guards].",
            )
        return Check("Power", "info", "Running on battery; a long run will drain it.", "Plug in.")
    if power.on_ac:
        return Check("Power", "ok", "Plugged in." if power.has_battery else "On mains power.", "")
    return Check(
        "Power",
        "info",
        "Couldn't tell whether this machine is plugged in.",
        "Keep it plugged in during long runs.",
    )


def _wsl_check() -> Check:
    return Check(
        "WSL",
        "info",
        "Running inside WSL.",
        "Keep your photos and As If's data inside the WSL filesystem (for example ~/photos), "
        "not under /mnt/c: it's faster, and Windows indexers and antivirus never see them.",
    )


def _checks(cfg: Config, specs: Specs, deps: list[Dependency], probe: Probe) -> list[Check]:
    by_name = {d.name: d for d in deps}
    checks = [
        _exiftool_check(probe, by_name["exiftool"]),
        _llama_check(cfg, by_name["llama-server"]),
        _encryption_check(specs),
        _swap_check(specs.swap),
        _ram_dir_check(specs),
        _disk_check(specs),
        _power_check(cfg, specs.power),
    ]
    if specs.is_wsl:
        checks.append(_wsl_check())
    return checks


# -- the whole report ----------------------------------------------------------------


def run_doctor(cfg: Config, probe: Probe | None = None) -> DoctorReport:
    """Specs, dependencies, checks and a model recommendation for this machine."""
    probe = probe or SystemProbe()
    specs = detect_specs(cfg, probe)
    deps = find_dependencies(cfg, probe)
    checks = _checks(cfg, specs, deps, probe)

    vram_mb = max(
        (g.vram_total_mb for g in specs.gpus if g.vendor != "apple" and g.vram_total_mb),
        default=None,
    )
    recommendation = None
    try:
        recommendation = recommend(
            ram_total_gb=(specs.ram_total_mb or 0) / 1024,
            vram_gb=vram_mb / 1024 if vram_mb else None,
            apple_silicon=specs.apple_silicon,
            unified_memory_gb=specs.unified_memory_mb / 1024 if specs.unified_memory_mb else None,
            lang=cfg.general.caption_lang,
        )
    except ValueError as exc:
        checks.append(
            Check(
                "Model",
                "warn",
                f"No model suggestion: {exc}",
                "Set caption_lang under [general] to a language a model lists "
                "(see `asif models list`); most of them are multilingual.",
            )
        )
    return DoctorReport(specs, deps, checks, recommendation)


def report_to_dict(report: DoctorReport) -> dict[str, Any]:
    """Plain data for ``asif doctor --json``."""
    return asdict(report)


# -- rendering -------------------------------------------------------------------

_MARKS: dict[CheckStatus, tuple[str, str]] = {
    "ok": ("✓", "green"),
    "warn": ("!", "yellow"),
    "fail": ("✗", "red"),
    "info": ("i", "blue"),
}

_SWAP_WORDS: dict[SwapKind, str] = {
    "none": "none",
    "zram": "zram (RAM only)",
    "encrypted": "encrypted",
    "plain": "NOT encrypted",
    "mixed": "partly NOT encrypted",
    "unknown": "unknown",
}


_DEVICE_WORDS: dict[str, str] = {
    "cpu": "the CPU",
    "gpu": "the GPU",
    "hybrid": "GPU + CPU (hybrid)",
    "metal": "the Apple GPU (Metal)",
}


def _gb(mb: int) -> str:
    return f"{mb / 1024:.1f} GB"


def _yes_no(value: bool | None, yes: str = "yes", no: str = "no") -> str:
    return "unknown" if value is None else yes if value else no


def _spec_rows(s: Specs) -> list[tuple[str, str]]:
    system = s.os_name
    if s.os_release:
        # "macOS 15.1", but "Linux (Ubuntu 24.04 LTS, kernel 6.8.0)".
        system += f" {s.os_release}" if s.os_release[0].isdigit() else f" ({s.os_release})"
    rows = [("System", f"{system}{' under WSL' if s.is_wsl else ''}, {s.arch}")]

    cpu = s.cpu_model or "unknown"
    if s.cores_physical and s.cores_logical:
        cpu += f"; {s.cores_physical} cores, {s.cores_logical} threads"
    elif s.cores_logical:
        cpu += f"; {s.cores_logical} threads"
    if s.cpu_flags:
        cpu += f"; {' '.join(f.upper() for f in s.cpu_flags)}"
    rows.append(("CPU", cpu))

    if s.ram_total_mb is None:
        rows.append(("Memory", "unknown"))
    else:
        memory = f"{_gb(s.ram_total_mb)} total"
        if s.ram_available_mb is not None:
            memory += f", {_gb(s.ram_available_mb)} available"
        if s.apple_silicon:
            memory += " (unified: shared with the GPU)"
        rows.append(("Memory", memory))

    gpus = []
    for g in s.gpus:
        text = g.name
        if g.vram_total_mb:
            text += f", {_gb(g.vram_total_mb)} VRAM"
            if g.vram_free_mb is not None:
                text += f" ({_gb(g.vram_free_mb)} free)"
        gpus.append(text)
    rows.append(("GPU", "\n".join(gpus) or "none found"))

    rows += [
        ("Swap", f"{_SWAP_WORDS[s.swap.kind]} ({s.swap.detail})"),
        ("Disk encryption", _yes_no(s.root_encrypted)),
        (
            "Power",
            _yes_no(s.power.on_ac, "plugged in" if s.power.has_battery else "mains", "on battery"),
        ),
        (
            "Staging folder",
            s.staging_path
            + (f" ({_gb(s.disk_free_mb)} free)" if s.disk_free_mb is not None else ""),
        ),
        ("RAM work dir", f"{s.ram_dir} ({'in RAM' if s.ram_dir_in_ram else 'on disk'})"),
    ]
    return rows


def render_report(report: DoctorReport, console: Console) -> None:
    """Friendly terminal output: specs, tools, checks with hints, then the model pick."""
    specs = Table(title="This machine", show_header=False, title_justify="left")
    specs.add_column(style="bold", no_wrap=True)
    specs.add_column(overflow="fold")
    for label, value in _spec_rows(report.specs):
        specs.add_row(label, Text(value))
    console.print(specs)

    tools = Table(title="Tools", title_justify="left")
    tools.add_column("", no_wrap=True)
    tools.add_column("Tool", no_wrap=True)
    tools.add_column("Version", overflow="fold")
    tools.add_column("What it's for")
    for dep in report.deps:
        if dep.found:
            mark = Text("✓", style="green")
        else:
            mark = Text("✗", style="red") if dep.required else Text("-", style="dim")
        version = dep.version or ("found" if dep.found else "not installed")
        tools.add_row(mark, Text(dep.name), Text(version), Text(dep.purpose))
    console.print(tools)

    console.print(Text("Checks", style="italic"))
    checks = Table.grid(padding=(0, 1))
    checks.add_column(no_wrap=True)
    checks.add_column(overflow="fold")
    for check in report.checks:
        symbol, color = _MARKS[check.status]
        body = Text(check.summary)
        if check.hint:
            body.append("\n" + check.hint, style="dim")
        checks.add_row(Text(f" {symbol}", style=f"bold {color}"), body)
    console.print(checks)
    console.print()

    rec = report.recommendation
    if rec is not None:
        heading = Text.assemble((rec.model_name, "bold"), f" ({rec.model_id}) on ")
        heading.append(_DEVICE_WORDS[rec.device])
        alternatives = Table.grid(padding=(0, 1))
        alternatives.add_column(no_wrap=True)
        alternatives.add_column(overflow="fold")
        for alt in rec.alternatives:
            alternatives.add_row(Text("  or", style="dim"), Text(alt, style="dim"))
        console.print(
            Panel(
                Group(heading, Text(rec.reason), alternatives),
                title="Suggested model",
                title_align="left",
            )
        )

    fails = sum(c.status == "fail" for c in report.checks)
    warns = sum(c.status == "warn" for c in report.checks)
    if fails or warns:
        parts = [
            f"{n} {word}{'' if n == 1 else 's'}"
            for n, word in ((fails, "problem"), (warns, "warning"))
            if n
        ]
        console.print(f"{' and '.join(parts)} above; each one says what to do.")
    else:
        console.print("[green]All checks passed.[/]")
