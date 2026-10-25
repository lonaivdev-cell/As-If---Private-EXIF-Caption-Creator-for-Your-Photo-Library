# As If

**Private, offline AI captions for your photo library.**
A model on *your* machine describes each photo, and the description and keywords go into the photo's own metadata (XMP), so any photo app can search them. Your photos, and everything said about them, never reach a third party. *As if.*

- 🔒 **Offline by design.** The model runs locally, in its own process. The main `asif` process has no network code at all, and any Python code in it that tries to open a socket crashes it.
- 🧾 **Your data stays yours.** It only *adds* to empty fields and never overwrites your own captions or keywords. The file itself records exactly what As If added, so `asif undo` (M6) can remove just that.
- 🛡️ **Your pixels stay as they are.** Before any copy is written, As If proves the image data, dates, GPS, orientation, colour profile, previews and the rest of your metadata are unchanged. For JPEG, PNG and WebP that means every byte outside the XMP packet (and the IPTC block with `--compat wide`), and every value you had inside it (exiftool does rewrite the packet's layout); a file it can't prove that for is skipped, not risked.

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

# 3. Check your machine (read-only; nothing is changed, nothing goes online)
uv run asif doctor
```

### Try the pipeline without a model

The **stub** model writes fake captions from each photo's colours, plus the keyword `stub` (where As If can add keywords; see [What gets written](#what-gets-written)). It's for seeing how things work, not for photos you care about.

```bash
uv run asif caption ~/Pictures/test ~/Pictures/test-captioned --model stub --limit 10
```

Your originals in `~/Pictures/test` are only read. The captioned copies go to `~/Pictures/test-captioned`, with the same folder layout.

### Use a real model

1. Install **llama.cpp** so you have `llama-server` (on macOS: `brew install llama.cpp`; on Linux: a release build from the llama.cpp project, matching your GPU). If you point As If at a wrapper script instead, it must end with `exec llama-server "$@"`.
2. Ask As If what fits your machine: `uv run asif models recommend`.
3. Download that model's **`.gguf` weights and its `mmproj` `.gguf`** (the vision part) from Hugging Face. `asif models list` shows the catalogue. Automatic, checksum-verified downloads arrive in M2.
4. Preview first; `--dry-run` writes no copies (it still makes the test write on a copy in RAM, so it skips the same photos a real run would, without running the model on them):

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
| Caption | `XMP-dc:Description` (default language + yours) | Only if the photo has no caption yet (in XMP, IPTC or EXIF; camera filler like "OLYMPUS DIGITAL CAMERA" doesn't count), or if As If wrote it and nobody has edited it since |
| Keywords | `XMP-dc:Subject` (plus `IPTC:Keywords` with `--compat wide`: JPEG and TIFF, unless the file has a Photoshop IPTC checksum exiftool can't keep in step: in any TIFF, or a JPEG with more than one or one outside its usual place) | Added and de-duplicated against your XMP and IPTC keywords, never removed. Keywords you only have in IPTC are copied into XMP first, so apps that read XMP still show them; if that can't be done safely, no keywords are added and the summary says why. With `--compat wide`, a Photoshop IPTC checksum (IPTCDigest) that matched your IPTC is updated along with it, so apps that follow the MWG guidelines (Lightroom, Bridge) keep showing your XMP |
| "As If did this" | `XMP-asif:*`: tool and model version, model hash, prompt version, run id, time, and exactly what was added | Always |
| Image fingerprint | `XMP-et:OriginalImageHash` (SHA-256 of the pixel data) | Once, never overwritten |
| **Never touched** | Dates, GPS, orientation, colour profile, maker notes, previews, the image itself | A TIFF with maker notes is skipped, because exiftool would have to rewrite them, and so is a JPEG whose maker note points to a preview after the image |

Formats written today: **JPEG, PNG, WebP, TIFF**. HEIC/AVIF, RAW (sidecars) and video are skipped and listed in the summary, and so is any file As If can't prove safe to write: a photo with Content Credentials (C2PA), embedded or linked from its XMP (`dcterms:provenance`), which any change would invalidate; a TIFF with maker notes, with data nothing points to, with JPEG-compressed strips or with a preview exiftool would move; XMP with markup inside a value or unlinked extended XMP; metadata exiftool can only read in part (like Photoshop's runaway "DocumentAncestors" list); an ImageMagick "raw profile" PNG or an Apple PNG whose iDOT index new XMP would break; a JPEG whose maker note points to a preview stored after the image, which new XMP would move (many Olympus cameras, some Minolta ones, and Sony's from about 2009 to 2014: NEX, SLT, the A6000 and the first A7 models), or whose thumbnail runs past its EXIF segment (Sigma's DP2 Quattro); a BigTIFF, which exiftool can't write; a TIFF whose strips, tiles or JPEG blocks share bytes, or that holds old-style JPEG tables (some document scanners); XMP followed by leftovers of an older packet that may hold values; or a file whose extension doesn't match its content. A PNG with a bad chunk checksum is quarantined as damaged. Photos rated or tagged in Windows (File Explorer, Photo Viewer) are skipped for now: exiftool would rewrite the address of Windows' own rating fields. Before the model runs, As If writes a placeholder caption into a second copy in RAM and runs every gate on it, so a photo that can't be written cleanly is skipped without spending model time. A skipped photo is never decoded, so if its pixels are damaged too, it's still listed as skipped.

---

## How your photos are protected

Each one is a rule enforced by code and checked by the test suite, not just a promise.

| Guarantee | How |
|---|---|
| Originals are never modified | M1 only reads them and writes copies to a different folder. A file already at the destination is never overwritten. |
| The image data is unchanged | exiftool's `ImageDataHash` (SHA-256) before and after must match. |
| Nothing else changed either | Every tag outside As If's own fields is compared, unknown ones too, and binary ones byte for byte. For JPEG, PNG and WebP, every byte outside the XMP packet must match too; for TIFF, every byte must be reachable from what exiftool rewrites. The XMP's own text is also read with an XML parser, each value with the property it belongs to, so nothing exiftool's parser misses can silently vanish or move to another property. |
| Your captions and keywords are never removed | The gate checks this itself instead of trusting the writer: every keyword there before is still there after, and a caption may only change if it's As If's own, unedited (its hash matches the one As If recorded). |
| Motion photos and multi-picture files survive | The data after a JPEG's end marker (motion-photo video, Ultra HDR gain maps) must be byte-identical, or the photo is skipped. |
| Nothing is clobbered | If the original changes while As If works on it, the photo is skipped. If it changed after its copy was made, a re-run says so instead of calling the old copy done. |
| The copy on disk is the copy that was checked | It's written atomically, then read back and compared. |
| Plaintext stays in RAM | Photos are worked on in a private RAM folder (tmpfs). Every copy, including the test write's copy and the ones exiftool replaces during each write, is overwritten and deleted after each photo. No RAM folder? As If refuses to start unless you pass `--allow-disk-workdir`. |
| Stopping is safe | Ctrl+C, `kill`, closing the terminal: the RAM copy is wiped and every helper process stops before As If exits, and a second signal can't cut that cleanup short. On Linux, if As If is killed outright (`kill -9`), exiftool, the model worker and llama-server stop with it. Core dumps are off, so a crash can't write a photo to disk. If As If is killed outright, the next run wipes what it left in RAM, even if you changed `work_dir` in between (unless the old one is on a network drive or another FUSE mount, which could hang the run: that one is wiped by the next run that uses it). |
| Private photos stay private | A copy never gets more permissions than its original effectively had: photos from a `0700` folder are copied into `0700` folders. A copy only keeps its group if that group could reach the original, and never picks up a shared folder's group or inherited ACLs (POSIX ACLs, and NFSv4 ACLs on NFS; SMB and macOS ACLs aren't checked yet). As If won't write into a folder that people can list who can't list your source's folder; it checks which group that is, ACLs, who owns the folders on the way and where a symlinked folder really points. A drive that gives every file the same permissions (FAT, exFAT, many network shares) is refused unless nobody else can get to it, as under `/media/<you>`. On a network drive (NFS, SMB, sshfs, cloud mounts) the share, not this computer, decides who sees files, so As If doesn't put private photos there yet, checking each photo and its folder (encrypted NAS and cloud sync is planned, M5); photos everyone on this computer can already see may go, and the share's groups get no more than everyone else. As If holds the destination open while it runs: if it's gone or unmounted anyway (a drive unplugged), As If stops instead of writing to whatever disk is left. |
| No network in the main process | CI fails if orchestrator code imports networking modules (checked by importing all of it, and by reading the source for imports, even lazy ones inside functions, and importing exactly what each names). An import disguised on purpose could get past that; at runtime an audit hook turns any socket use by Python code into a crash. Native code could get around a Python hook; the real wall is the sandbox in M3. |
| The model can't be used by other programs | `llama-server` listens on 127.0.0.1 only, behind a random API key passed through the environment (not the command line), and the worker checks that the port really is its own server's before sending a photo. |
| The model can't fetch or send anything | llama-server runs in offline mode and sees none of your own llama.cpp settings: `LLAMA_*`, `MTMD_*`, `HF_*` and `HUGGING*` variables are removed, and it gets an empty private config folder, so `~/.config/llama.cpp/config.ini` is never read. A system-wide `/etc/llama.cpp/config.ini` can't be hidden yet (M3), so while one exists As If refuses to start llama-server and `asif doctor` says so. Extra `server_args` must be on an allow-list of speed, memory, GPU and logging options; anything that downloads, loads other files, opens tunnels, runs tools or serves files is refused. |
| A bad model answer can't do damage | Output must match a strict JSON schema, and control characters are stripped, so a caption can't smuggle options into exiftool. File names are printed with control characters shown as `\x1b` and the like, so a crafted name can't send commands to your terminal. |
| Systematic bugs stop the run | More than 2 % failures in the last 200 photos stops everything. That includes the test write before the model: only exiftool's "minor" complaints about a file's own metadata count as a limit of that file. |
| Telemetry stays home | The only log is a local timings file with stage durations. No file names, paths, sizes or captions (an allow-list enforces it), and photos are processed in a random order. The copies' own file times still show the order they were made in, so treat the log as no less private than the destination folder. |

**Not yet** (honest list): the model isn't sandboxed yet (M3: no network at all). Nothing is encrypted yet (M4), so the timings log is plain text, though it holds no file names or captions. There's no battery or thermal guard yet (M7): a long run on battery keeps going until the battery is flat. There's no cloud sync (M5) and no search (M8). As If assumes your disk is encrypted and your machine is yours: `asif doctor` warns about unencrypted swap and disks.

---

## Commands

```text
asif doctor [--json]              check this machine (read-only)
asif caption SRC DEST [options]   caption a folder into copies
    -m, --model stub|local|<id>   --gguf FILE --mmproj FILE   --llama-server PATH
    --lang TAG   --compat xmp|wide   --max-edge PX   --dry-run   --limit N
    --allow-disk-workdir          (weaker) use an on-disk work dir if there's no tmpfs
asif models list | recommend      the model catalogue; a pick for this machine
asif config path | show | init    where settings live; what's in effect; a starter file
asif version
```

Exit codes: `0` all good · `1` some photos failed or were quarantined (or a folder couldn't be read, or was a link to a folder, which As If doesn't follow) · `2` bad options or setup (including a failed `asif doctor` check, a config file that isn't UTF-8, an `ASIF_HOME` that isn't set up, or a state folder As If can't write when a run starts; if the timings log fails partway through, the run carries on and a note says so) · `3` run stopped early (failure budget, full destination, model worker gone) · `130` stopped by Ctrl+C or a signal.

Settings live in `~/.config/asif/config.toml`; `asif config init` writes a commented starter file. Set `ASIF_HOME` to keep everything (config, logs, later the vault) in one folder, such as an encrypted volume, then run `asif config init` once to set it up (it leaves a small `.asif-home` file there). While that file is missing (say the volume isn't mounted and only its empty mount point is left), every command except `asif version`, `asif config path` and `asif config init` stops with "Can't start" before reading your settings, instead of using the defaults and writing to the disk underneath. As If never creates the folders `ASIF_HOME` is in.

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
uv run python tests/fixtures/make_fixtures.py /tmp/fixtures   # look at the synthetic test photos
```

The test photos are generated, never real: `tests/fixtures/make_fixtures.py` draws them with Pillow and tags them with exiftool (EXIF, GPS, rotation, motion-photo trailer, multi-picture, camera maker notes, truncated, corrupt...). `tests/fake_llama_server.py` stands in for llama-server, so the whole pipeline is tested without downloading a model.

## Licence

MIT, see [LICENSE](LICENSE). Models and tools As If talks to (llama.cpp, exiftool, the models themselves) have their own licences and are never bundled.
