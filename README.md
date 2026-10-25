# As If

**Private, offline AI captions for your photo library.**
A model on *your* machine describes each photo, and the description and keywords go into the photo's own metadata (XMP), so any photo app can search them. Your photos, and everything said about them, never reach a third party. *As if.*

- 🔒 **Offline by design.** The model runs locally. The main `asif` process can't even open a network socket: it crashes if anything tries.
- 🧾 **Your data stays yours.** It only *adds* to empty fields and never overwrites your own captions or keywords. Everything it adds is labelled, so it can be removed cleanly.
- 🛡️ **Your pixels stay as they are.** Before any copy is written, As If proves the image data, dates, GPS, orientation, colour profile, previews, and every other byte outside the caption are unchanged.

> **Status: early (M0 + M1 of 11).** It captions a local folder into a separate folder of copies. Cloud sync, sandboxing, encryption and search are designed but not built yet. See [Status](#status).

---

## Quick start

You need **Linux** (or WSL2 on Windows; macOS should mostly work but isn't tested yet), **Python 3.12+**, [**uv**](https://docs.astral.sh/uv/) and **exiftool**.

```bash
# 1. exiftool (Debian/Ubuntu; other systems: your package manager)
sudo apt install libimage-exiftool-perl

# 2. As If
git clone https://github.com/lonaivdev-cell/As-If---Private-EXIF-Caption-Creator-for-Your-Photo-Library.git as-if
cd as-if
uv sync

# 3. Check your machine (read-only; nothing is changed)
uv run asif doctor
```

### Try the pipeline without a model

The **stub** model writes fake captions from each photo's colours, plus the keyword `stub`. It's for seeing how things work, not for photos you care about.

```bash
uv run asif caption ~/Pictures/test ~/Pictures/test-captioned --model stub --limit 10
```

Your originals in `~/Pictures/test` are only read. The captioned copies go to `~/Pictures/test-captioned`, with the same folder layout.

### Use a real model

1. Install **llama.cpp** so you have `llama-server` (on macOS: `brew install llama.cpp`; on Linux: a release build from the llama.cpp project, matching your GPU).
2. Ask As If what fits your machine: `uv run asif models recommend`.
3. Download that model's **`.gguf` weights and its `mmproj` `.gguf`** (the vision part) from Hugging Face. `asif models list` shows the catalogue. Automatic, checksum-verified downloads arrive in M2.
4. Preview first; `--dry-run` writes nothing:

```bash
uv run asif caption ~/Pictures/test ~/Pictures/test-captioned \
  --gguf ~/models/model.Q4_K_M.gguf --mmproj ~/models/mmproj-F16.gguf --dry-run
```

5. Drop `--dry-run` when you like what you see. Add `--lang pt-BR` (or any language tag) for captions in another language.

**Speed tip:** on a CPU, image size matters most. `--max-edge 768` is noticeably faster than the default 1024.

---

## What gets written

| What | Where in the file | Rule |
|---|---|---|
| Caption | `XMP-dc:Description` (default language + yours) | Only if empty, or if As If wrote it and nobody has edited it since |
| Keywords | `XMP-dc:Subject` (plus `IPTC:Keywords` with `--compat wide`) | Added, de-duplicated, never removed |
| "As If did this" | `XMP-asif:*`: tool and model version, model hash, prompt version, run id, time, and exactly what was added | Always |
| Image fingerprint | `XMP-et:OriginalImageHash` (SHA-256 of the pixel data) | Once, never overwritten |
| **Never touched** | Dates, GPS, orientation, colour profile, maker notes, previews, the image itself | — |

Formats written today: **JPEG, PNG, WebP, TIFF**. HEIC/AVIF, RAW (sidecars) and video are skipped and listed in the summary.

---

## How your photos are protected

Each one is a rule enforced by code and checked by the test suite, not just a promise.

| Guarantee | How |
|---|---|
| Originals are never modified | M1 only reads them and writes copies to a different folder. A file already at the destination is never overwritten. |
| The image data is unchanged | exiftool's `ImageDataHash` (SHA-256) before and after must match. |
| Nothing else changed either | Every tag outside As If's own fields is compared, binary ones byte-for-byte. For JPEG, PNG and WebP, every byte outside the XMP packet must match too. |
| Motion photos and multi-picture files survive | The data after a JPEG's end marker (motion-photo video, Ultra HDR gain maps) must be byte-identical, or the photo is skipped. |
| Nothing is clobbered | If the original changes while As If works on it, the photo is skipped. |
| The copy on disk is the copy that was checked | It's written atomically, then read back and compared. |
| Plaintext stays in RAM | Photos are worked on in a private RAM folder (tmpfs), overwritten and deleted after each one. |
| No network in the main process | CI fails if any orchestrator module imports networking code. At runtime an audit hook turns any socket use into a crash. |
| The model can't be used by other programs | `llama-server` listens on 127.0.0.1 only, behind a random API key passed through the environment (not the command line). |
| A bad model answer can't do damage | Output must match a strict JSON schema, and control characters are stripped, so a caption can't smuggle options into exiftool. |
| Systematic bugs stop the run | More than 2 % failures in the last 200 photos stops everything. |
| Telemetry stays home | The only log is a local timings file with stage durations. No file names, paths or captions; an allow-list enforces it. |

**Not yet** (honest list): the model isn't sandboxed yet (M3: no network at all). Nothing is encrypted yet (M4), so the timings log is plain text, though it holds no personal data. There's no cloud sync (M5) and no search (M8). As If assumes your disk is encrypted and your machine is yours: `asif doctor` warns about unencrypted swap and disks.

---

## Commands

```text
asif doctor [--json]              check this machine (read-only)
asif caption SRC DEST [options]   caption a folder into copies
    -m, --model stub|local|<id>   --gguf FILE --mmproj FILE   --llama-server PATH
    --lang TAG   --compat xmp|wide   --max-edge PX   --dry-run   --limit N
asif models list | recommend      the model catalogue; a pick for this machine
asif config path | show | init    where settings live; what's in effect; a starter file
asif version
```

Exit codes: `0` all good · `1` some photos failed or were quarantined · `2` bad options or setup · `3` run stopped by the failure budget.

Settings live in `~/.config/asif/config.toml`; `asif config init` writes a commented starter file. Set `ASIF_HOME` to keep everything (config, logs, later the vault) in one folder, such as an encrypted volume.

---

## Status

The full plan is in **[docs/DESIGN.md](docs/DESIGN.md)**, one milestone at a time, each with a checkpoint.

| | Milestone | State |
|---|---|---|
| M0 | Skeleton + `doctor` | ✅ |
| M1 | Local-only core: folder in → copies out, every gate, timings log | ✅ |
| M2 | Model manifest, pull, verify, bench | catalogue + recommendation done; downloads next |
| M3 | Inference + writer sandboxes (bubblewrap), isolation canary | planned |
| M4 | Encrypted staging + vault | planned |
| M5 | Cloud adapters (rclone, Proton Drive CLI, SSH) | planned |
| M6 | In-place write-back + `asif undo` | planned |
| M7 | Batches, resume after a crash, thermal/power guards | planned |
| M8 | Duplicates, thumbnails, review, search | planned |
| M9 | Encrypted telemetry | planned |
| M10 | macOS + Windows/WSL2 | planned |
| M11 | Reproducible, signed releases | planned |

---

## Development

```bash
uv sync                      # Python deps, pinned by uv.lock
uv run pytest                # all tests (needs exiftool); -m "not slow" skips the 50 MP panorama
uv run ruff check . && uv run ruff format --check . && uv run mypy
python tests/fixtures/make_fixtures.py /tmp/fixtures   # look at the synthetic test photos
```

The test photos are generated, never real: `tests/fixtures/make_fixtures.py` draws them with Pillow and tags them with exiftool (EXIF, GPS, rotation, motion-photo trailer, multi-picture, truncated, corrupt...). `tests/fake_llama_server.py` stands in for llama-server, so the whole pipeline is tested without downloading a model.

## Licence

MIT, see [LICENSE](LICENSE). Models and tools As If talks to (llama.cpp, exiftool, the models themselves) have their own licences and are never bundled.
