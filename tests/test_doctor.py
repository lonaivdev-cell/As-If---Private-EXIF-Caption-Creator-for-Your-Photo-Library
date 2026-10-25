"""``asif doctor``, driven by a fake machine (and once by the real one).

The samples below are what the real files and tools print, trimmed: a fake
``Probe`` serves them, so every branch of the detection runs here whatever
machine the tests happen to run on.
"""

from __future__ import annotations

import io
import json
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from rich.console import Console

from asif import doctor, workdir
from asif.config import Config
from asif.doctor import (
    Check,
    DoctorReport,
    SystemProbe,
    detect_specs,
    find_dependencies,
    render_report,
    report_to_dict,
    run_doctor,
)
from asif.workdir import RamDirChoice
from tests.conftest import requires_exiftool

GOOD_HASH = "9017786507e193ea2363a137ff678dbdc1406a498f3c88e2334b2ebb1acbd987"


# -- the fake machine ----------------------------------------------------------


@dataclass
class FakeProbe:
    """A machine made of canned files and command outputs.

    ``commands`` maps "program arg1 arg2" (program by base name) to its result;
    the longest key that prefixes the actual command line wins.
    """

    plat: str = "linux"
    arch: str = "x86_64"
    files: dict[str, str] = field(default_factory=dict)
    dirs: dict[str, list[str]] = field(default_factory=dict)
    commands: dict[str, tuple[int, str]] = field(default_factory=dict)
    programs: dict[str, str] = field(default_factory=dict)
    environ: dict[str, str] = field(default_factory=dict)
    links: dict[str, str] = field(default_factory=dict)
    devs: dict[str, tuple[int, int]] = field(default_factory=dict)
    free_mb: int | None = 200_000
    ran: list[list[str]] = field(default_factory=list)
    seen_files: dict[str, tuple[bytes, int]] = field(default_factory=dict)

    def read_text(self, path: str) -> str | None:
        return self.files.get(path)

    def run(self, argv: list[str], timeout: float = 5.0) -> tuple[int, str] | None:
        self.ran.append(list(argv))
        for arg in argv[1:]:
            # Record any real file handed to a tool (the doctor's sample JPEG).
            if arg.startswith("/") and os.path.isfile(arg):
                mode = stat.S_IMODE(os.stat(os.path.dirname(arg)).st_mode)
                self.seen_files[arg] = (Path(arg).read_bytes(), mode)
        line = " ".join([os.path.basename(argv[0]), *argv[1:]])
        keys = [k for k in self.commands if line == k or line.startswith(k + " ")]
        return self.commands[max(keys, key=len)] if keys else None

    def which(self, name: str) -> str | None:
        return self.programs.get(name)

    def env(self, name: str) -> str | None:
        return self.environ.get(name)

    def platform(self) -> str:
        return self.plat

    def machine(self) -> str:
        return self.arch

    def disk_free_mb(self, path: Path) -> int | None:
        return self.free_mb

    def listdir(self, path: str) -> list[str]:
        return sorted(self.dirs.get(path, []))

    def realpath(self, path: str) -> str:
        return self.links.get(path, path)

    def stat_dev(self, path: str) -> tuple[int, int] | None:
        return self.devs.get(path)


# -- samples -------------------------------------------------------------------

CPUINFO_INTEL = "\n".join(
    f"""processor\t: {i}
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 158
model name\t: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
physical id\t: 0
siblings\t: 8
core id\t\t: {i % 4}
cpu cores\t: 4
flags\t\t: fpu vme de pse tsc msr sse sse2 ssse3 fma cx16 sse4_1 sse4_2 avx f16c avx2 bmi2
"""
    for i in range(8)
)

CPUINFO_PI5 = """processor\t: 0
BogoMIPS\t: 108.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm
CPU implementer\t: 0x41
CPU architecture: 8
CPU part\t: 0xd0b

processor\t: 1
BogoMIPS\t: 108.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm
CPU implementer\t: 0x41
CPU architecture: 8
CPU part\t: 0xd0b

Revision\t: d04170
Serial\t\t: 4d2f9e8a1b3c5d7e
Model\t\t: Raspberry Pi 5 Model B Rev 1.0
"""

MEMINFO = """MemTotal:       16127808 kB
MemFree:         1203456 kB
MemAvailable:    9876544 kB
Buffers:          345678 kB
Cached:          7654321 kB
SwapCached:            0 kB
SwapTotal:       8388604 kB
"""

MEMINFO_OLD_KERNEL = """MemTotal:        4046132 kB
MemFree:          102400 kB
Buffers:           51200 kB
Cached:           870400 kB
"""

SWAPS_HEADER = "Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n"
SWAPS_ZRAM = (
    SWAPS_HEADER + "/dev/zram0                              partition\t8388604\t\t0\t\t100\n"
)
SWAPS_DM = SWAPS_HEADER + "/dev/dm-1                               partition\t8388604\t\t0\t\t-2\n"
SWAPS_PLAIN = (
    SWAPS_HEADER + "/dev/nvme0n1p3                          partition\t8388604\t\t0\t\t-2\n"
)
SWAPS_FILE = SWAPS_HEADER + "/swapfile                               file\t\t4194300\t\t0\t\t-2\n"

PROC_VERSION_WSL = (
    "Linux version 5.15.153.1-microsoft-standard-WSL2 (root@941d701f84f1) "
    "(gcc (GCC) 11.2.0, GNU ld (GNU Binutils) 2.37) #1 SMP Fri Mar 29 23:14:13 UTC 2024\n"
)
PROC_VERSION_NATIVE = (
    "Linux version 6.8.0-45-generic (buildd@lcy02-amd64-075) (x86_64-linux-gnu-gcc-13 "
    "(Ubuntu 13.2.0-23ubuntu4) 13.2.0) #45-Ubuntu SMP PREEMPT_DYNAMIC\n"
)

NVIDIA_SMI = "NVIDIA GeForce RTX 3060, 12288, 11801\n"
NVIDIA_SMI_TWO = "NVIDIA GeForce RTX 4090, 24564, 23010\nNVIDIA GeForce GTX 1050 Ti, 4096, [N/A]\n"
NVIDIA_SMI_NO_DRIVER = (
    "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
    "Make sure that the latest NVIDIA driver is installed and running.\n"
)
ROCM_SMI = (
    "WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status\n\n"
    + json.dumps(
        {
            "card0": {
                "Card Series": "Navi 31 [Radeon RX 7900 XTX]",
                "Card Model": "0x744c",
                "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
                "GFX Version": "gfx1100",
                "VRAM Total Memory (B)": "25753026560",
                "VRAM Total Used Memory (B)": "1073741824",
            }
        }
    )
    + "\n"
)

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               61040.
Pages active:                            575262.
Pages inactive:                          561097.
Pages speculative:                        13939.
Pages throttled:                              0.
Pages wired down:                        237917.
Pages purgeable:                          15412.
"Translation faults":                 1176926596.
Pages copy-on-write:                   48434386.
"""
PMSET_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=24838243)\t87%; discharging; 6:12 remaining present: true\n"
)
PMSET_AC = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=24838243)\t100%; charged; 0:00 remaining present: true\n"
)
PMSET_DESKTOP = "Now drawing from 'AC Power'\n"

EXIFTOOL_OK = {
    "exiftool -ver": (0, "12.76\n"),
    "exiftool -api requesttags=imagedatahash -api imagehashtype=SHA256": (0, GOOD_HASH + "\n"),
    "exiftool -listw -XMP-et:all": (
        0,
        "Writable XMP-et tags:\n  OriginalImageHash OriginalImageHashType OriginalImageMD5\n",
    ),
}


def linux(**extra: object) -> FakeProbe:
    """A plain Ubuntu desktop: LUKS root, zram swap, no GPU, exiftool installed."""
    probe = FakeProbe(
        files={
            "/proc/cpuinfo": CPUINFO_INTEL,
            "/proc/meminfo": MEMINFO,
            "/proc/swaps": SWAPS_ZRAM,
            "/proc/version": PROC_VERSION_NATIVE,
            "/etc/os-release": 'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.1 LTS"\nID=ubuntu\n',
            "/proc/sys/kernel/osrelease": "6.8.0-45-generic\n",
            "/sys/dev/block/253:0/dm/uuid": "CRYPT-LUKS2-5e1b2f0a3c6d4e8f9a0b1c2d3e4f5a6b-luks\n",
            "/sys/class/block/dm-0/dev": "253:0\n",
        },
        devs={"/": (253, 0)},
        programs={"exiftool": "/usr/bin/exiftool"},
        commands=dict(EXIFTOOL_OK),
    )
    for key, value in extra.items():
        setattr(probe, key, value)
    return probe


def mac(**extra: object) -> FakeProbe:
    """An M2 Pro MacBook with 32 GB, on battery, FileVault on."""
    probe = FakeProbe(
        plat="darwin",
        arch="arm64",
        commands={
            "sysctl -n machdep.cpu.brand_string": (0, "Apple M2 Pro\n"),
            "sysctl -n hw.logicalcpu": (0, "12\n"),
            "sysctl -n hw.physicalcpu": (0, "12\n"),
            "sysctl -n hw.memsize": (0, "34359738368\n"),
            "sysctl -n hw.optional.neon": (0, "1\n"),
            "sysctl -n hw.optional.arm64": (0, "1\n"),
            "sysctl -n machdep.cpu.features": (1, "sysctl: unknown oid 'machdep.cpu.features'\n"),
            "sw_vers -productVersion": (0, "15.1\n"),
            "vm_stat": (0, VM_STAT),
            "pmset -g batt": (0, PMSET_BATTERY),
            "fdesetup status": (0, "FileVault is On.\n"),
            **EXIFTOOL_OK,
        },
        programs={"exiftool": "/opt/homebrew/bin/exiftool"},
    )
    for key, value in extra.items():
        setattr(probe, key, value)
    return probe


@pytest.fixture(autouse=True)
def ram_dir_in_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real find_ram_dir looks at this machine; pin it for repeatable reports."""
    monkeypatch.setattr(
        doctor, "find_ram_dir", lambda override="": RamDirChoice(Path("/dev/shm"), True, "tmpfs")
    )


def check(report: DoctorReport, name: str) -> Check:
    [found] = [c for c in report.checks if c.name == name]
    return found


# -- CPU, memory, OS -----------------------------------------------------------


def test_linux_intel_cpu_memory_and_os() -> None:
    specs = detect_specs(Config(), linux())
    assert specs.os_name == "Linux"
    assert specs.os_release == "Ubuntu 24.04.1 LTS, kernel 6.8.0-45-generic"
    assert specs.arch == "x86_64"
    assert specs.cpu_model == "Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz"
    assert (specs.cores_physical, specs.cores_logical) == (4, 8)
    assert specs.cpu_flags == ["avx", "avx2", "f16c", "fma"]  # no avx512f on this one
    assert specs.ram_total_mb == 16127808 // 1024
    assert specs.ram_available_mb == 9876544 // 1024
    assert not specs.is_wsl and not specs.apple_silicon
    assert specs.unified_memory_mb is None


def test_raspberry_pi_cpu_takes_the_board_model_and_not_the_serial() -> None:
    probe = linux(arch="aarch64")
    probe.files["/proc/cpuinfo"] = CPUINFO_PI5
    specs = detect_specs(Config(), probe)
    assert specs.cpu_model == "Raspberry Pi 5 Model B Rev 1.0"
    assert specs.cores_logical == 2
    assert specs.cores_physical is None  # ARM cpuinfo doesn't say
    assert specs.cpu_flags == ["asimd"]
    assert "4d2f9e8a1b3c5d7e" not in json.dumps(report_to_dict(run_doctor(Config(), probe)))


def test_arm_board_without_a_model_line_uses_the_device_tree() -> None:
    probe = linux(arch="aarch64")
    probe.files["/proc/cpuinfo"] = CPUINFO_PI5.replace(
        "Model\t\t: Raspberry Pi 5 Model B Rev 1.0\n", ""
    )
    probe.files["/proc/device-tree/model"] = "Pine64 RockPro64 v2.1\0"
    assert detect_specs(Config(), probe).cpu_model == "Pine64 RockPro64 v2.1"


def test_old_kernel_without_memavailable() -> None:
    probe = linux()
    probe.files["/proc/meminfo"] = MEMINFO_OLD_KERNEL
    specs = detect_specs(Config(), probe)
    assert specs.ram_total_mb == 4046132 // 1024
    assert specs.ram_available_mb == (102400 + 51200 + 870400) // 1024


def test_unreadable_proc_means_unknown_not_a_crash() -> None:
    specs = detect_specs(Config(), FakeProbe())
    assert specs.cpu_model is None and specs.cores_logical is None and specs.cpu_flags == []
    assert specs.ram_total_mb is None and specs.ram_available_mb is None
    assert specs.os_release is None
    assert specs.swap.kind == "unknown"
    assert specs.root_encrypted is None


@pytest.mark.parametrize(
    ("version", "environ"),
    [(PROC_VERSION_WSL, {}), (PROC_VERSION_NATIVE, {"WSL_DISTRO_NAME": "Ubuntu-24.04"})],
)
def test_wsl_detection(version: str, environ: dict[str, str]) -> None:
    probe = linux(environ=environ)
    probe.files["/proc/version"] = version
    specs = detect_specs(Config(), probe)
    assert specs.is_wsl
    # The virtual disk lives on Windows: dm-crypt says nothing about BitLocker.
    assert specs.root_encrypted is None


def test_native_linux_is_not_wsl() -> None:
    assert not detect_specs(Config(), linux()).is_wsl


# -- GPUs -----------------------------------------------------------------------


def test_nvidia_gpu() -> None:
    probe = linux()
    probe.commands[
        "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits"
    ] = (0, NVIDIA_SMI)
    [gpu] = detect_specs(Config(), probe).gpus
    assert (gpu.vendor, gpu.name, gpu.vram_total_mb, gpu.vram_free_mb) == (
        "nvidia",
        "NVIDIA GeForce RTX 3060",
        12288,
        11801,
    )


def test_two_nvidia_gpus_and_a_missing_value() -> None:
    probe = linux()
    probe.commands["nvidia-smi"] = (0, NVIDIA_SMI_TWO)
    gpus = detect_specs(Config(), probe).gpus
    assert [(g.name, g.vram_total_mb, g.vram_free_mb) for g in gpus] == [
        ("NVIDIA GeForce RTX 4090", 24564, 23010),
        ("NVIDIA GeForce GTX 1050 Ti", 4096, None),
    ]


def test_nvidia_smi_without_a_driver_means_no_gpu() -> None:
    probe = linux()
    probe.commands["nvidia-smi"] = (9, NVIDIA_SMI_NO_DRIVER)
    assert detect_specs(Config(), probe).gpus == []


def test_amd_gpu_via_rocm_smi_with_a_warning_first() -> None:
    probe = linux()
    probe.commands["rocm-smi --showproductname --showmeminfo vram --json"] = (0, ROCM_SMI)
    [gpu] = detect_specs(Config(), probe).gpus
    assert gpu.vendor == "amd"
    assert gpu.name == "Navi 31 [Radeon RX 7900 XTX]"
    assert gpu.vram_total_mb == 25753026560 // (1024 * 1024)
    assert gpu.vram_free_mb == (25753026560 - 1073741824) // (1024 * 1024)


@pytest.mark.parametrize("output", ["not json at all", "{broken", "[1, 2]", '{"system": {}}'])
def test_garbled_rocm_smi_is_ignored(output: str) -> None:
    probe = linux()
    probe.commands["rocm-smi"] = (0, output)
    assert detect_specs(Config(), probe).gpus == []


def test_nvidia_gpu_drives_the_recommendation() -> None:
    probe = linux()
    probe.commands["nvidia-smi"] = (0, NVIDIA_SMI)
    rec = run_doctor(Config(), probe).recommendation
    assert rec is not None
    assert (rec.model_id, rec.device) == ("qwen3.5-9b-q4km", "gpu")
    assert "12 GB of VRAM" in rec.reason


# -- swap -------------------------------------------------------------------------


def swap_probe(swaps: str, **files: str) -> FakeProbe:
    probe = linux()
    probe.files["/proc/swaps"] = swaps
    probe.files.update(files)
    return probe


def test_no_swap_is_fine() -> None:
    report = run_doctor(Config(), swap_probe(SWAPS_HEADER))
    assert report.specs.swap.kind == "none"
    assert check(report, "Swap").status == "ok"


def test_zram_swap_is_fine() -> None:
    report = run_doctor(Config(), swap_probe(SWAPS_ZRAM))
    assert report.specs.swap.kind == "zram"
    assert "/dev/zram0" in report.specs.swap.detail
    assert check(report, "Swap").status == "ok"


def test_dm_crypt_swap_partition_is_encrypted() -> None:
    probe = swap_probe(
        SWAPS_DM,
        **{
            "/sys/class/block/dm-1/dev": "253:1\n",
            "/sys/dev/block/253:1/dm/uuid": "CRYPT-PLAIN-cryptswap\n",
        },
    )
    report = run_doctor(Config(), probe)
    assert report.specs.swap.kind == "encrypted"
    assert check(report, "Swap").status == "ok"


def test_swap_on_lvm_inside_luks_is_encrypted() -> None:
    probe = swap_probe(
        SWAPS_HEADER + "/dev/mapper/vg-swap partition 8388604 0 -2\n",
        **{
            "/sys/class/block/dm-2/dev": "253:2\n",
            "/sys/dev/block/253:2/dm/uuid": "LVM-Hd7v2kq0aBcDeFgHiJkLmNoPqRsTuVwX\n",
        },
    )
    probe.links["/dev/mapper/vg-swap"] = "/dev/dm-2"
    probe.dirs["/sys/dev/block/253:2/slaves"] = ["dm-0"]
    assert detect_specs(Config(), probe).swap.kind == "encrypted"


def test_plain_swap_partition_warns_with_a_hint() -> None:
    probe = swap_probe(SWAPS_PLAIN, **{"/sys/class/block/nvme0n1p3/dev": "259:3\n"})
    report = run_doctor(Config(), probe)
    assert report.specs.swap.kind == "plain"
    swap = check(report, "Swap")
    assert swap.status == "warn"
    assert "zram" in swap.hint and "LUKS" in swap.hint


def test_swap_file_on_an_encrypted_filesystem_is_encrypted() -> None:
    probe = swap_probe(SWAPS_FILE)
    probe.devs["/swapfile"] = (253, 0)  # same LUKS device as /
    assert detect_specs(Config(), probe).swap.kind == "encrypted"


def test_swap_file_on_a_plain_filesystem_is_plain() -> None:
    probe = swap_probe(SWAPS_FILE)
    probe.devs["/swapfile"] = (259, 2)
    assert detect_specs(Config(), probe).swap.kind == "plain"


def test_swap_file_on_btrfs_on_luks_is_found_through_proc_mounts() -> None:
    probe = swap_probe(
        SWAPS_HEADER + "/swap/swapfile file 4194300 0 -2\n",
        **{
            "/proc/mounts": (
                "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0\n"
                "/dev/mapper/luks-1234 / btrfs rw,relatime,ssd,subvol=/@ 0 0\n"
                "/dev/mapper/luks-1234 /swap btrfs rw,relatime,ssd,subvol=/@swap 0 0\n"
                "/dev/nvme0n1p1 /boot/efi vfat rw,relatime 0 0\n"
            ),
        },
    )
    probe.devs["/swap/swapfile"] = (0, 35)  # btrfs: an anonymous device
    probe.links["/dev/mapper/luks-1234"] = "/dev/dm-0"
    assert detect_specs(Config(), probe).swap.kind == "encrypted"


def test_deleted_swap_file_and_escaped_spaces() -> None:
    probe = swap_probe(SWAPS_HEADER + "/my\\040swap\\040(deleted) file 4194300 0 -2\n")
    probe.devs["/my swap"] = (259, 2)
    swap = detect_specs(Config(), probe).swap
    assert swap.kind == "plain"
    assert "/my swap" in swap.detail and "deleted" not in swap.detail


def test_some_plain_swap_is_mixed_and_warns() -> None:
    probe = swap_probe(
        SWAPS_ZRAM + "/dev/nvme0n1p3 partition 8388604 0 -2\n",
        **{"/sys/class/block/nvme0n1p3/dev": "259:3\n"},
    )
    report = run_doctor(Config(), probe)
    assert report.specs.swap.kind == "mixed"
    assert check(report, "Swap").status == "warn"


def test_zram_plus_encrypted_is_encrypted() -> None:
    probe = swap_probe(
        SWAPS_ZRAM + "/dev/dm-1 partition 8388604 0 -2\n",
        **{
            "/sys/class/block/dm-1/dev": "253:1\n",
            "/sys/dev/block/253:1/dm/uuid": "CRYPT-PLAIN-cryptswap\n",
        },
    )
    assert detect_specs(Config(), probe).swap.kind == "encrypted"


def test_undeterminable_swap_is_unknown_and_informs() -> None:
    probe = swap_probe(SWAPS_FILE)  # no stat, no mounts
    report = run_doctor(Config(), probe)
    assert report.specs.swap.kind == "unknown"
    swap = check(report, "Swap")
    assert swap.status == "info" and "swapon" in swap.hint


def test_unreadable_proc_swaps_is_unknown() -> None:
    probe = linux()
    del probe.files["/proc/swaps"]
    assert detect_specs(Config(), probe).swap.kind == "unknown"


# -- disk encryption --------------------------------------------------------------


def test_luks_root_is_encrypted() -> None:
    report = run_doctor(Config(), linux())
    assert report.specs.root_encrypted is True
    assert check(report, "Disk encryption").status == "ok"


def test_lvm_on_luks_root_is_encrypted() -> None:
    probe = linux(devs={"/": (253, 1)})
    probe.files["/sys/dev/block/253:1/dm/uuid"] = "LVM-abcdefghijklmnop\n"
    probe.dirs["/sys/dev/block/253:1/slaves"] = ["dm-0"]
    assert detect_specs(Config(), probe).root_encrypted is True


def test_plain_root_warns() -> None:
    report = run_doctor(Config(), linux(devs={"/": (259, 2)}))
    assert report.specs.root_encrypted is False
    enc = check(report, "Disk encryption")
    assert enc.status == "warn" and "LUKS" in enc.hint


def test_zfs_root_is_unknown() -> None:
    probe = linux(devs={"/": (0, 40)})
    probe.files["/proc/mounts"] = "rpool/ROOT/ubuntu / zfs rw,relatime,xattr,posixacl 0 0\n"
    report = run_doctor(Config(), probe)
    assert report.specs.root_encrypted is None
    assert check(report, "Disk encryption").status == "info"


def test_wsl_disk_encryption_points_at_bitlocker() -> None:
    probe = linux()
    probe.files["/proc/version"] = PROC_VERSION_WSL
    report = run_doctor(Config(), probe)
    enc = check(report, "Disk encryption")
    assert enc.status == "info" and "BitLocker" in enc.hint
    wsl = check(report, "WSL")
    assert wsl.status == "info" and "/mnt/c" in wsl.hint


def test_no_wsl_check_outside_wsl() -> None:
    assert not [c for c in run_doctor(Config(), linux()).checks if c.name == "WSL"]


# -- power -------------------------------------------------------------------------

PS = "/sys/class/power_supply"


def laptop(ac_online: str, battery_status: str) -> FakeProbe:
    probe = linux()
    probe.dirs[PS] = ["AC", "BAT0", "hidpp_battery_0"]
    probe.files.update(
        {
            f"{PS}/AC/type": "Mains\n",
            f"{PS}/AC/online": f"{ac_online}\n",
            f"{PS}/BAT0/type": "Battery\n",
            f"{PS}/BAT0/present": "1\n",
            f"{PS}/BAT0/status": f"{battery_status}\n",
            # A wireless mouse: a battery, but not ours.
            f"{PS}/hidpp_battery_0/type": "Battery\n",
            f"{PS}/hidpp_battery_0/scope": "Device\n",
            f"{PS}/hidpp_battery_0/status": "Discharging\n",
        }
    )
    return probe


def test_laptop_on_ac() -> None:
    report = run_doctor(Config(), laptop("1", "Charging"))
    assert (report.specs.power.on_ac, report.specs.power.has_battery) == (True, True)
    assert check(report, "Power").status == "ok"


def test_laptop_on_battery_warns_when_ac_is_required() -> None:
    report = run_doctor(Config(), laptop("0", "Discharging"))
    assert (report.specs.power.on_ac, report.specs.power.has_battery) == (False, True)
    power = check(report, "Power")
    assert power.status == "warn" and "require_ac_power" in power.hint


def test_laptop_on_battery_only_informs_when_ac_is_not_required() -> None:
    cfg = Config()
    cfg.guards.require_ac_power = False
    assert check(run_doctor(cfg, laptop("0", "Discharging")), "Power").status == "info"


def test_usb_c_charger_counts_as_mains() -> None:
    probe = linux()
    probe.dirs[PS] = ["BAT0", "ucsi-source-psy-USBC000:001"]
    probe.files.update(
        {
            f"{PS}/BAT0/type": "Battery\n",
            f"{PS}/BAT0/status": "Charging\n",
            f"{PS}/ucsi-source-psy-USBC000:001/type": "USB\n",
            f"{PS}/ucsi-source-psy-USBC000:001/online": "1\n",
        }
    )
    assert detect_specs(Config(), probe).power.on_ac is True


@pytest.mark.parametrize(
    ("status", "on_ac"),
    [("Full", True), ("Not charging", True), ("Discharging", False), ("Unknown", None)],
)
def test_battery_status_alone(status: str, on_ac: bool | None) -> None:
    probe = linux()
    probe.dirs[PS] = ["BAT1"]
    probe.files.update({f"{PS}/BAT1/type": "Battery\n", f"{PS}/BAT1/status": f"{status}\n"})
    assert detect_specs(Config(), probe).power.on_ac is on_ac


def test_desktop_without_a_battery_is_on_mains() -> None:
    probe = linux()
    probe.dirs[PS] = ["hidpp_battery_0"]
    probe.files.update(
        {f"{PS}/hidpp_battery_0/type": "Battery\n", f"{PS}/hidpp_battery_0/scope": "Device\n"}
    )
    power = detect_specs(Config(), probe).power
    assert (power.on_ac, power.has_battery) == (True, False)
    nothing_listed = detect_specs(Config(), linux()).power
    assert (nothing_listed.on_ac, nothing_listed.has_battery) == (True, False)


def test_wsl_without_power_supplies_is_unknown() -> None:
    probe = linux()
    probe.files["/proc/version"] = PROC_VERSION_WSL
    report = run_doctor(Config(), probe)
    assert (report.specs.power.on_ac, report.specs.power.has_battery) == (None, None)
    assert check(report, "Power").status == "info"


# -- macOS and others ----------------------------------------------------------------


def test_apple_silicon_mac() -> None:
    report = run_doctor(Config(), mac())
    s = report.specs
    assert (s.os_name, s.os_release) == ("macOS", "15.1")
    assert s.apple_silicon and s.unified_memory_mb == 32768 and s.ram_total_mb == 32768
    assert s.ram_available_mb == (61040 + 561097 + 13939) * 16384 // (1024 * 1024)
    assert s.cpu_model == "Apple M2 Pro" and (s.cores_logical, s.cores_physical) == (12, 12)
    assert s.cpu_flags == ["neon"]
    assert [g.vendor for g in s.gpus] == ["apple"]
    assert s.swap.kind == "encrypted"
    assert (s.power.on_ac, s.power.has_battery) == (False, True)
    assert s.root_encrypted is True
    assert check(report, "Swap").status == "ok"
    assert check(report, "Disk encryption").summary == "FileVault is on."
    assert check(report, "Power").status == "warn"
    rec = report.recommendation
    assert rec is not None
    assert (rec.model_id, rec.device) == ("gemma-4-26b-a4b-q4", "metal")


def test_mac_desktop_on_ac_with_filevault_off() -> None:
    probe = mac()
    probe.commands["pmset -g batt"] = (0, PMSET_DESKTOP)
    probe.commands["fdesetup status"] = (0, "FileVault is Off.\n")
    report = run_doctor(Config(), probe)
    assert (report.specs.power.on_ac, report.specs.power.has_battery) == (True, False)
    enc = check(report, "Disk encryption")
    assert enc.status == "warn" and "FileVault" in enc.hint


def test_mac_on_ac_power() -> None:
    probe = mac()
    probe.commands["pmset -g batt"] = (0, PMSET_AC)
    assert detect_specs(Config(), probe).power.on_ac is True


def test_python_under_rosetta_still_counts_as_apple_silicon() -> None:
    assert detect_specs(Config(), mac(arch="x86_64")).apple_silicon


def test_intel_mac() -> None:
    probe = mac(arch="x86_64")
    del probe.commands["sysctl -n hw.optional.arm64"]
    del probe.commands["sysctl -n hw.optional.neon"]
    probe.commands["sysctl -n machdep.cpu.brand_string"] = (
        0,
        "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz\n",
    )
    probe.commands["sysctl -n machdep.cpu.features"] = (0, "FPU VME SSE3 FMA AVX1.0 F16C RDRAND\n")
    probe.commands["sysctl -n machdep.cpu.leaf7_features"] = (0, "SMEP ERMS AVX2 BMI2\n")
    specs = detect_specs(Config(), probe)
    assert not specs.apple_silicon and specs.unified_memory_mb is None and specs.gpus == []
    assert specs.cpu_flags == ["avx", "avx2", "f16c", "fma"]


def test_mac_without_tools_is_unknown_not_a_crash() -> None:
    report = run_doctor(Config(), FakeProbe(plat="darwin", arch="arm64"))
    assert report.specs.ram_total_mb is None
    assert (report.specs.power.on_ac, report.specs.root_encrypted) == (None, None)


def test_mac_ram_dir_hint_is_about_macos() -> None:
    def on_disk(override: str = "") -> RamDirChoice:
        return RamDirChoice(Path("/var/folders/x/T"), False, "no tmpfs found")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(doctor, "find_ram_dir", on_disk)
        ram = check(run_doctor(Config(), mac()), "RAM work dir")
    assert ram.status == "warn" and "macOS" in ram.hint


def test_other_platforms_are_unknown() -> None:
    report = run_doctor(Config(), FakeProbe(plat="win32", arch="AMD64"))
    assert report.specs.os_name == "Windows"
    assert report.specs.swap.kind == "unknown"
    assert report.specs.power.on_ac is None
    assert check(report, "exiftool").status == "fail"


# -- dependencies -------------------------------------------------------------------


def test_dependencies_with_versions_trimmed_to_one_line() -> None:
    probe = linux()
    probe.programs.update(
        {
            "llama-server": "/usr/local/bin/llama-server",
            "bwrap": "/usr/bin/bwrap",
            "proton-drive": "/usr/local/bin/proton-drive",
        }
    )
    probe.commands["llama-server --version"] = (
        0,
        "\nversion: 4567 (1a2b3c4d)\nbuilt with cc (GCC) 13.2.0 for x86_64-linux-gnu\n",
    )
    probe.commands["bwrap --version"] = (0, "bubblewrap 0.9.0\n")
    deps = {d.name: d for d in find_dependencies(Config(), probe)}
    assert list(deps) == ["exiftool", "llama-server", "bwrap", "age", "rclone", "proton-drive"]
    assert deps["exiftool"].version == "12.76" and deps["exiftool"].required
    assert deps["llama-server"].version == "version: 4567 (1a2b3c4d)"
    assert deps["bwrap"].version == "bubblewrap 0.9.0" and not deps["bwrap"].required
    assert not deps["age"].found and deps["age"].path is None
    assert all(d.purpose for d in deps.values())
    # Found, but never executed: it might phone home, and the doctor doesn't.
    assert deps["proton-drive"].found and deps["proton-drive"].version is None
    assert not any("proton-drive" in argv[0] for argv in probe.ran)


def test_llama_server_from_the_config_path() -> None:
    cfg = Config()
    cfg.model.llama_server = "/opt/llama/bin/llama-server"
    probe = linux()
    probe.programs["/opt/llama/bin/llama-server"] = "/opt/llama/bin/llama-server"
    probe.commands["llama-server --version"] = (0, "version: 1 (abc)\n")
    report = run_doctor(cfg, probe)
    assert check(report, "llama-server").status == "ok"


def test_missing_llama_server_warns_and_mentions_the_stub_model() -> None:
    report = run_doctor(Config(), linux())
    llama = check(report, "llama-server")
    assert llama.status == "warn"
    assert "--model stub" in llama.hint and "llama.cpp" in llama.hint


def test_missing_custom_llama_server_names_the_configured_path() -> None:
    cfg = Config()
    cfg.model.llama_server = "/opt/llama/bin/llama-server"
    assert "/opt/llama/bin/llama-server" in check(run_doctor(cfg, linux()), "llama-server").summary


def test_hanging_version_command_is_survivable() -> None:
    probe = linux()
    probe.programs["age"] = "/usr/bin/age"  # no canned output: behaves like a timeout
    deps = {d.name: d for d in find_dependencies(Config(), probe)}
    assert deps["age"].found and deps["age"].version is None


# -- exiftool, tested by what it can do -----------------------------------------------


def test_exiftool_missing_fails_with_install_hint() -> None:
    probe = linux(programs={})
    exif = check(run_doctor(Config(), probe), "exiftool")
    assert exif.status == "fail"
    assert "apt install" in exif.hint and "brew install exiftool" in exif.hint


def test_exiftool_that_can_hash_and_list_the_tag_passes() -> None:
    exif = check(run_doctor(Config(), linux()), "exiftool")
    assert exif.status == "ok" and "12.76" in exif.summary and exif.hint == ""


def test_exiftool_without_imagedatahash_fails() -> None:
    probe = linux()
    probe.commands["exiftool -api requesttags=imagedatahash -api imagehashtype=SHA256"] = (0, "\n")
    exif = check(run_doctor(Config(), probe), "exiftool")
    assert exif.status == "fail"
    assert "ImageDataHash" in exif.summary and "OriginalImageHash" not in exif.summary
    assert "exiftool.org" in exif.hint


def test_exiftool_hash_command_failing_fails() -> None:
    probe = linux()
    probe.commands["exiftool -api requesttags=imagedatahash -api imagehashtype=SHA256"] = (
        1,
        f"Error: something\n{GOOD_HASH}\n",
    )
    assert check(run_doctor(Config(), probe), "exiftool").status == "fail"


def test_exiftool_without_original_image_hash_tag_fails() -> None:
    probe = linux()
    probe.commands["exiftool -listw -XMP-et:all"] = (0, "Writable XMP-et tags:\n  [empty list]\n")
    exif = check(run_doctor(Config(), probe), "exiftool")
    assert exif.status == "fail"
    assert "OriginalImageHash" in exif.summary and "ImageDataHash" not in exif.summary


def test_exiftool_sample_is_a_real_jpeg_in_a_private_dir_and_is_removed() -> None:
    probe = linux()
    run_doctor(Config(), probe)
    [(path, (data, mode))] = probe.seen_files.items()
    assert data[:2] == b"\xff\xd8"  # a JPEG made just for this
    assert mode == 0o700
    assert "asif-doctor-" in path
    assert not os.path.exists(path) and not os.path.exists(os.path.dirname(path))


# -- other checks ----------------------------------------------------------------------


def test_ram_dir_on_disk_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor,
        "find_ram_dir",
        lambda override="": RamDirChoice(Path("/tmp"), False, "no tmpfs found"),
    )
    report = run_doctor(Config(), linux())
    assert (report.specs.ram_dir, report.specs.ram_dir_in_ram) == ("/tmp", False)
    ram = check(report, "RAM work dir")
    assert ram.status == "warn" and "/dev/shm" in ram.hint


def test_configured_work_dir_is_passed_to_find_ram_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(override: str = "") -> RamDirChoice:
        seen.append(override)
        return RamDirChoice(Path(override), True, "configured")

    monkeypatch.setattr(doctor, "find_ram_dir", fake)
    cfg = Config()
    cfg.resources.work_dir = "/run/user/1000/asif"
    assert detect_specs(cfg, linux()).ram_dir == "/run/user/1000/asif"
    assert seen == ["/run/user/1000/asif"]


@pytest.mark.parametrize(
    ("free", "status"), [(5_000, "warn"), (10_239, "warn"), (10_240, "ok"), (None, "info")]
)
def test_staging_disk(free: int | None, status: str, tmp_path: Path) -> None:
    cfg = Config()
    cfg.resources.staging_dir = str(tmp_path / "staging")
    report = run_doctor(cfg, linux(free_mb=free))
    assert report.specs.staging_path == str(tmp_path / "staging")
    disk = check(report, "Staging disk")
    assert disk.status == status
    if status != "ok":
        assert "staging_dir" in disk.hint


def test_model_check_appears_when_no_model_fits_the_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**_: object) -> None:
        raise ValueError("no model in the catalogue lists the caption language 'tlh'")

    monkeypatch.setattr(doctor, "recommend", refuse)
    report = run_doctor(Config(), linux())
    assert report.recommendation is None
    model = check(report, "Model")
    assert model.status == "warn" and "caption_lang" in model.hint


def test_caption_language_reaches_the_recommendation() -> None:
    probe = linux()
    probe.files["/proc/meminfo"] = "MemTotal:  8000000 kB\nMemAvailable: 4000000 kB\n"
    cfg = Config()
    cfg.general.caption_lang = "pt-BR"
    rec = run_doctor(cfg, probe).recommendation
    assert rec is not None and rec.model_id == "gemma-4-e2b-q4"


MACHINES = {
    "desktop": linux,
    "wsl": lambda: linux(files={**linux().files, "/proc/version": PROC_VERSION_WSL}),
    "laptop-battery": lambda: laptop("0", "Discharging"),
    "plain-swap": lambda: swap_probe(SWAPS_PLAIN, **{"/sys/class/block/nvme0n1p3/dev": "259:3\n"}),
    "unencrypted": lambda: linux(devs={"/": (259, 2)}, free_mb=1_000),
    "mac": mac,
    "bare": FakeProbe,
    "windows": lambda: FakeProbe(plat="win32"),
}


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_every_check_that_is_not_ok_says_what_to_do(machine: str) -> None:
    report = run_doctor(Config(), MACHINES[machine]())
    assert report.checks
    for c in report.checks:
        assert c.status in ("ok", "warn", "fail", "info")
        assert c.summary.strip()
        if c.status != "ok":
            assert len(c.hint) > 20, c


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_report_is_plain_json_and_renders(machine: str) -> None:
    report = run_doctor(Config(), MACHINES[machine]())
    data = json.loads(json.dumps(report_to_dict(report)))
    assert set(data) == {"specs", "deps", "checks", "recommendation"}
    assert {"swap", "power", "gpus", "staging_path", "ram_dir_in_ram"} <= set(data["specs"])
    out = io.StringIO()
    render_report(report, Console(file=out, width=100, color_system=None))
    assert "This machine" in out.getvalue()


# -- rendering --------------------------------------------------------------------------


def render(report: DoctorReport, width: int = 100) -> str:
    out = io.StringIO()
    render_report(report, Console(file=out, width=width, color_system=None))
    return out.getvalue()


def test_render_shows_specs_tools_checks_hints_and_the_pick() -> None:
    probe = laptop("0", "Discharging")
    probe.commands["nvidia-smi"] = (0, NVIDIA_SMI)
    text = render(run_doctor(Config(), probe))
    for expected in (
        "This machine",
        "Intel(R) Core(TM) i7-8700",
        "4 cores, 8 threads",
        "AVX2",
        "15.4 GB total",
        "NVIDIA GeForce RTX 3060, 12.0 GB VRAM",
        "zram",
        "Tools",
        "exiftool",
        "12.76",
        "not installed",
        "✓",
        "!",
        "Plug in the charger",
        "Suggested model",
        "Qwen3.5-9B (qwen3.5-9b-q4km) on the GPU",
        "because the GPU has 12 GB of VRAM",
        "or ",
        "2 warnings above",
    ):
        assert expected in text, expected


def test_render_marks_failures_and_counts_them() -> None:
    text = render(run_doctor(Config(), linux(programs={})))
    assert "✗" in text
    assert "1 problem and 1 warning above" in text


def test_render_all_good() -> None:
    probe = linux()
    probe.programs["llama-server"] = "/usr/bin/llama-server"
    text = render(run_doctor(Config(), probe))
    assert "All checks passed." in text


def test_render_never_interprets_tool_output_as_markup() -> None:
    probe = linux()
    probe.commands["exiftool -ver"] = (0, "12.76 [/] [bold red]x[/bold red]\n")
    text = render(run_doctor(Config(), probe), width=200)
    assert "12.76 [/] [bold red]x[/bold red]" in text


def test_render_narrow_terminal_does_not_crash() -> None:
    assert "Suggested model" in render(run_doctor(Config(), mac()), width=40)


# -- the real machine -----------------------------------------------------------------------


@requires_exiftool
def test_smoke_real_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "find_ram_dir", workdir.find_ram_dir)  # the real one, here
    report = run_doctor(Config())
    deps = {d.name: d for d in report.deps}
    assert deps["exiftool"].found and deps["exiftool"].version
    assert check(report, "exiftool").status == "ok"
    assert report.specs.os_name == {"linux": "Linux", "darwin": "macOS"}.get(sys.platform)
    assert report.specs.cores_logical and report.specs.ram_total_mb
    assert report.recommendation is not None
    json.dumps(report_to_dict(report))
    assert "Suggested model" in render(report)


def test_system_probe_files_dirs_and_paths(tmp_path: Path) -> None:
    p = SystemProbe()
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b").mkdir()
    assert p.read_text(str(tmp_path / "f.txt")) == "hello"
    assert p.read_text(str(tmp_path / "missing")) is None
    assert p.read_text(str(tmp_path)) is None  # a directory
    assert p.listdir(str(tmp_path)) == ["b", "f.txt"]
    assert p.listdir(str(tmp_path / "missing")) == []
    (tmp_path / "link").symlink_to(tmp_path / "f.txt")
    assert p.realpath(str(tmp_path / "link")) == str((tmp_path / "f.txt").resolve())
    assert p.stat_dev(str(tmp_path)) is not None
    assert p.stat_dev(str(tmp_path / "missing")) is None
    free = p.disk_free_mb(tmp_path / "not" / "there" / "yet")
    assert isinstance(free, int) and free >= 0
    assert p.platform() == sys.platform
    assert p.machine()
    assert p.which(os.path.basename(sys.executable)) or p.which("sh")


def test_system_probe_run(monkeypatch: pytest.MonkeyPatch) -> None:
    p = SystemProbe()
    monkeypatch.setenv("ASIF_DOCTOR_TEST", "x")
    assert p.env("ASIF_DOCTOR_TEST") == "x"
    out = p.run(
        [
            sys.executable,
            "-c",
            "import os, sys; print(os.environ['LC_ALL']); print('err', file=sys.stderr); "
            "sys.exit(3)",
        ]
    )
    assert out is not None
    code, text = out
    assert code == 3
    assert "C" in text.split() and "err" in text  # C locale, stderr merged
    assert p.run(["/nonexistent/asif-no-such-tool"]) is None
    assert p.run([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.3) is None
