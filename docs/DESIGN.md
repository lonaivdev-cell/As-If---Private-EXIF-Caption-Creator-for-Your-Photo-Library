# As If — private, offline AI captions for your photo library

> Design + build guide, v0.1 — October 2026. Working name was *quietcap*.
> Build status lives in the [README](../README.md#status); this document is the plan.

---

## 0. In one paragraph

As If does five things in order:

1. It pulls photos from your cloud drive in batches sized to fit your machine, and keeps them encrypted at rest.
2. It decrypts one photo at a time into RAM.
3. A local vision-language model, with no network access, describes the photo.
4. exiftool writes that description and a set of keywords into the photo's own metadata. As If then proves the image data was not touched.
5. It uploads the photo back and adds it to an encrypted local catalogue you can search.

Nothing about your photos ever reaches a third party, and that includes the people who wrote As If.

---

## 1. Honest scope

**What the pieces actually are.** No model "writes EXIF". A vision-language model (VLM) produces structured text, and exiftool writes that text into the file. They are two separate tools on purpose, so the model never touches your files.

**Every processed photo is a full re-upload.** Embedded metadata is part of the file's bytes, so a 10k-photo library means uploading the whole library once more. Pacing is built in (§11, §12).

**Your cloud may not search what we write.** Proton Drive's photo search today is limited to type filters, and its roadmap promises only incremental search improvements. Searchability comes from two places instead:

- As If's local catalogue (`asif search`).
- Any app that reads XMP, such as digiKam, Immich, Lightroom, or the OS file search.

The metadata is portable: it travels inside the file to wherever you move it next.

**Ente is read-only.** Ente's export is one-way, and Ente parses EXIF only at upload time. Ente can be a *source* (point As If at its export folder), but As If cannot write back into it.

**Never:**
- identify or name people,
- delete anything in your cloud (duplicates are *suggested*, never removed),
- overwrite metadata it didn't write,
- send telemetry anywhere.

---

## 2. Facts this design rests on (checked October 2026)

| Fact | Consequence for the design |
|---|---|
| Proton released an official Drive CLI (June 2026) for Windows, macOS and Linux. It is built on the Drive SDK (MIT-licensed source). Login happens in the browser, the session lives in the OS secret store, and it has `--json` output. It has **no sync engine**: it runs one command and exits. | The Proton adapter wraps the official CLI and orchestrates one-shot commands. |
| The CLI has supported Photos and albums since July 2026. | The Photos timeline is reachable, not just "My files". |
| Proton asks CLI users to transfer only what changed, and throttles accounts with unusually high traffic. | Upload pacing is a core feature. |
| The native Linux Drive app is still an announced beta, starting as a headless daemon. | Don't depend on it. |
| rclone's `protondrive` backend is reverse-engineered and uses API paths Proton has deprecated. Their continued functionality will stop being guaranteed after the SDK grace period. | Use rclone for *other* clouds; reach Proton via the official CLI. |
| Ollama can't load Qwen3.5 vision GGUFs (the projector is a separate file), while llama.cpp can. | The inference runtime is `llama-server` from llama.cpp. |
| `llama-server` accepts a JSON schema and constrains output to it. | Model output is always machine-parseable. |
| exiftool can compute `ImageDataHash`, a hash of the image data excluding metadata (MD5, SHA256 or SHA512). | This is the integrity gate (§7). |
| On CPU, image encoding dominates per-image time. One 2.2B VLM takes ~16 s per image on 6 threads, and well under a second on any modest GPU. | Input resolution is the main CPU lever. MoE only speeds up the text half (§9). |
| A recent llama.cpp issue reports ~82 s of CPU image encoding in `llama-server` with a BF16 projector for Qwen3.5-9B. | Pin the llama.cpp build, prefer F16 projectors, and always benchmark before a full pass. |

Sources are listed at the end.

**Verified while building M1** (exiftool 12.76, Ubuntu 24.04): `ImageDataHash` with `imagehashtype=SHA256` works on JPEG, PNG, WebP and TIFF (except a TIFF whose only image is JPEG-compressed strips: exiftool takes those for a preview and computes no hash, so such files are skipped); a JPEG's trailer (bytes after EOI) is *not* part of that hash, which is why the trailer gate exists; exiftool keeps an unknown JPEG trailer byte-for-byte when writing XMP; `XMP-et:OriginalImageHash` and `OriginalImageHashType` exist; bubblewrap's `--unshare-all` leaves a sandbox with loopback only (connect: "Network is unreachable", DNS fails).

---

## 3. Threat model

**Protects against:**
- **Third parties seeing photo content or derived text.** The cloud stores only what it already stored; the model has no network; telemetry never leaves the machine.
- **Model or runtime exfiltration.** The inference sandbox has no network namespace route out, and a canary test proves it before every run (§16).
- **Local leakage.** Plaintext never touches disk, snapshots (btrfs, APFS), backups, or crash leftovers. Staging is ciphertext; plaintext lives only in RAM.
- **Silent corruption of originals.** Equality gates, additive-only writes, and undo (§7).
- **Clobbering a newer edit.** The remote file must be unchanged since download, or it is skipped.

**Does not protect against:**
- A compromised OS or root account.
- Malware running as you during a run.
- Physical attacks on a running machine.

Use full-disk encryption and a machine you trust.

- **Swap:** RAM-disk pages can be swapped out. `asif doctor` warns if swap is unencrypted. zram without a writeback device, or swap inside LUKS, is fine (zram with a writeback device is only as safe as that device). On WSL2, swap is a file on the Windows drive: it's encrypted only if BitLocker or Device Encryption is on; `swap=0` in `.wslconfig` turns it off.
- *As built (M1):* the work dir must be RAM-backed or As If refuses to run (`--allow-disk-workdir` accepts the weaker option). Core dumps are switched off for the whole process tree (a crash can't write pixels to disk), and Ctrl+C, SIGTERM, SIGHUP and SIGQUIT unwind cleanly, wiping the RAM copy and stopping every child. Children also get SIGTERM if the orchestrator is SIGKILLed (Linux): exiftool is started through a tiny exec trampoline that arms the parent-death signal after fork, and the inference worker arms its own and takes llama-server down with it, so this holds however many threads the orchestrator runs. A stop signal that arrives during cleanup is acted on only after the cleanup finishes. The next run wipes any work dir whose run died without cleaning up, wherever it was; core dumps are limited to 1 byte (no core file, and no hand-off to a crash reporter), the dump filter is emptied and the orchestrator is non-dumpable.
- **Non-E2EE clouds** (plain S3, WebDAV): the provider can see your files regardless of As If. Use `rclone crypt` there.

---

## 4. Architecture

**Core principle:** network access is a capability granted to exactly two short-lived processes, the model puller and the transfer adapter. Everything else runs without it.

```
                 ┌──────────────────────── host ─────────────────────────┐
                 │  ORCHESTRATOR  (contains no networking code)          │
                 │   ├─ state machine + scheduler + guards              │
                 │   ├─ key manager (keys live in memory)               │
                 │   ├─ VAULT  (SQLCipher: catalogue, thumbs, journal)  │
                 │   └─ TELEMETRY writer ──► telemetry.db               │
                 │        (path never mounted into any sandbox)         │
                 └─────┬───────────────────┬──────────────────┬─────────┘
                  pipe │              pipe │ JSONL       pipe │
   ┌───────────────────▼──────┐  ┌─────────▼──────────────┐ ┌─▼──────────────────┐
   │ TRANSFER sandbox         │  │ INFERENCE sandbox      │ │ WRITER sandbox     │
   │ network: YES             │  │ network: NONE          │ │ network: NONE      │
   │ sees: credentials,       │  │ sees: models/ (ro),    │ │ sees: RAM work dir │
   │   staging/ (ciphertext), │  │   one decrypted image  │ │ exiftool -stay_open│
   │   RAM stream buffer      │  │   (ro)                 │ └────────────────────┘
   │ proton-drive | rclone |  │  │ llama-server on its    │
   │ rsync-over-ssh           │  │ private loopback +     │
   └──────────────────────────┘  │ worker (stdin/stdout)  │
                                 └────────────────────────┘
```

**Components:**
- **Orchestrator.** Python process that owns the state machine, keys, vault and telemetry. CI enforces "no network modules imported" for it (`tests/test_netguard.py` imports the whole orchestrator, and everything its source names even inside functions that never ran, and fails if `socket`, `ssl`, `http` or `urllib.request` got loaded). At runtime a Python audit hook additionally turns any socket use in the orchestrator into a crash (`asif/netguard.py`).
- **Transfer sandbox.** Runs the provider tool. Its download output is piped straight into an encrypting writer, so plaintext never lands on disk.
- **Inference sandbox.** Contains `llama-server`, listening on loopback *inside its own empty network namespace*, plus a tiny worker that relays JSON lines over stdin/stdout. There is no TCP path from the host to the model and no route from the model to anything.
- **Writer sandbox.** exiftool in `-stay_open` batch mode (one long-lived process; much faster than spawning one per file).
- **Vault.** Encrypted SQLite holding the catalogue, captions, keywords, hashes, thumbnails and the undo journal.
- **Telemetry store.** A separate encrypted DB with a separate key, readable only via `asif telemetry`.

**Batch-mode data flow, per batch:**
1. **List.** The transfer adapter lists the scope, and the orchestrator diffs it against the vault (skip already-captioned items).
2. **Download.** The adapter downloads the batch, encrypting into `staging/`, and records a remote fingerprint for each item.
3. **Decrypt.** For each item, the orchestrator decrypts it into the RAM work dir and computes `ImageDataHash` (H0).
4. **Describe.** The inference worker returns JSON, which is validated against the schema.
5. **Write.** The writer applies the tags to the RAM copy and computes `ImageDataHash` again (H1).
6. **Gate.** Gates run: H0 == H1, protected fields unchanged, remote unchanged (§7).
7. **Upload.** The adapter uploads the RAM copy; the orchestrator confirms the remote fingerprint.
8. **Clean up.** Thumbnail, catalogue row and undo journal go to the vault. The RAM copy is overwritten and unlinked (including the pre-write copy that exiftool's rename replaces), and the staged ciphertext is deleted.

**Linear mode** is the same loop with a batch size of 1 and **no disk staging at all**: download goes straight into RAM, then describe, write, gate, upload, wipe. It is slowest but needs almost no disk and very little RAM.

---

## 5. Per-image state machine

```
DISCOVERED → STAGED → OPENED → DESCRIBED → WRITTEN → VERIFIED → UPLOADED → CONFIRMED → CLEAN
                                   │           │          │           │
                                   ▼           ▼          ▼           ▼
                          FAILED_SCHEMA   FAILED_GATE  SKIPPED_   FAILED_UPLOAD
                          (retry ≤2 →     (never        CONFLICT   (backoff retry)
                           QUARANTINE)     uploaded)
```

- Every transition is journaled in the vault (SQLite WAL) **before** the action runs.
- After a crash or power cut, resume works from the journal. `UPLOADED` and `CONFIRMED` are re-derived by checking the remote fingerprint, never from memory. That prevents double uploads.
- **QUARANTINE** items appear in `asif review quarantine` with the reason attached (format, schema failure, gate failure).

---

## 6. Keys and storage

| Store | Default location (Linux) | Contents | Protection | Lifetime |
|---|---|---|---|---|
| Staging | `~/.cache/asif/staging/` | downloaded originals | `age` encryption with an **ephemeral session key held only in memory** | until `CONFIRMED`; leftovers wiped at next start |
| RAM work dir | private tmpfs (Linux) / RAM disk (macOS) | 1 (linear) to a few decrypted photos | RAM only; 0700, locked by its run | seconds; overwritten then unlinked; a dead run's leftovers are wiped by the next run, which also checks every parent folder an earlier run used (remembered in the state dir's `workdir-parents`; best effort; folders on network or FUSE mounts are skipped, since a dead mount would hang the run, until a run uses them again) |
| Vault | `~/.local/share/asif/vault.db` | catalogue, captions, keywords, hashes, 256 px thumbnails, undo journal | SQLCipher; key from your passphrase via Argon2id | until `asif wipe vault` |
| Telemetry | `~/.local/state/asif/telemetry.db` | timings, resources, errors — **no** names, paths or captions | SQLCipher; separate key (HKDF from master key, distinct context) | rolling, user-set retention |
| Credentials | OS secret store | Proton session (owned by the Proton CLI), rclone config password | OS (libsecret / Keychain / Credential Manager) | per provider |
| Models | `~/.local/share/asif/models/` | GGUF weights + vision projector | public files; integrity via pinned SHA-256 | until removed |

**Why the staging key is ephemeral:** if As If crashes, the staged files become unreadable garbage rather than a privacy incident. The originals are still in the cloud, so nothing is lost; As If just downloads again.

**Snapshots and backups:** staging holds only ciphertext, so snapper, Timeshift, Time Machine or restic snapshots capture nothing readable.

---

## 7. Integrity and rollback

**Gates before upload (all must pass):**

1. **Equality gate.** `ImageDataHash(original as downloaded) == ImageDataHash(processed)`.
   ```
   exiftool -api requesttags=imagedatahash -api imagehashtype=SHA256 -s3 -ImageDataHash photo.jpg
   ```
2. **Protected-fields gate.** These must be identical before and after: `DateTimeOriginal`, `CreateDate`, `GPS*`, `Orientation`, ICC profile and MakerNotes. Compare `exiftool -json -G1` dumps restricted to those tags.
   *As built:* the comparison covers **every** tag outside As If's own fields, not just the list above: `exiftool -json -n -G1:4 -a -u -b -all:all -ICC_Profile -MakerNotes`. `-n` compares raw values (a 1e-7° GPS move prints the same but is caught); `-G1:4` keeps repeated tags that plain `-G1` JSON drops; `-u` includes unknown tags; the raw ICC block and every binary tag (thumbnails, previews) are compared byte for byte. For maker notes exiftool returns its rebuilt block, so their raw byte identity is proven by the structure gate (JPEG, PNG, WebP), and TIFFs with maker notes are skipped. Bookkeeping that legitimately changes (WebP's VP8X flags, the strip, tile and thumbnail offsets, and the OtherImage/JpgFromRaw offsets whose JPEG the image-data hash covers, of every TIFF IFD and SubIFD (a TIFF whose preview exiftool would move, which the hash doesn't cover, is skipped instead), MPF's absolute image offsets) is allowed by exact tag name in `src/asif/meta/gates.py`, each with the reason. Inside As If's own fields the gate still protects the owner: XMP-dc:Subject and IPTC:Keywords may only gain values, XMP-dc:Description entries may only change if they hash to the CaptionDigest recorded before the write, and OriginalImageHash and the IPTC charset and record versions are write-once. With `compat = "wide"`, `Photoshop:IPTCDigest` may only go from in step with the IPTC (or no IPTC) to in step, and new IPTC must never leave it stale, so apps following the MWG guidelines keep showing the XMP. exiftool's warning about a digest that was already stale is the one write warning let through: As If leaves such a digest as it was.
3. **Trailer gate (JPEG).** Bytes after the image data must be identical, because Samsung and Google "motion photos" embed video there. If As If can't prove it, it skips the file.
4. **Remote-unchanged gate.** The remote fingerprint now must equal the fingerprint at download time (size + mtime, plus revision or hash when the adapter exposes it). Otherwise the item becomes `SKIPPED_CONFLICT`.
5. **Size sanity.** `processed_size − original_size < 64 KiB`, so a runaway write can't bloat files.

*Added while building M1:* a **structure gate** for JPEG, PNG and WebP. Apart from the XMP packet (and IPTC when `compat = "wide"`), every segment or chunk of the file must be byte-identical and in the same order. This catches changes `ImageDataHash` can't see, such as a rewritten MPF (multi-picture) index. For TIFF, which exiftool rebuilds, the snapshot instead proves every byte is reachable from an IFD exiftool rewrites (padding and zero filler aside). For every format, the XMP packet's own text values must all survive, read with an XML parser rather than exiftool's. Files the gates can't prove (maker-note TIFFs, unreferenced TIFF data, JPEG-compressed striped TIFFs and TIFF previews exiftool would move, markup inside XMP values including CDATA, orphaned extended XMP, metadata exiftool reads only in part, ImageMagick raw-profile PNGs, Apple iDOT PNGs without XMP ahead of the iDOT chunk, C2PA Content Credentials in any format (embedded, or kept outside the file and linked from XMP `dcterms:provenance`), unknown formats) are skipped before the model runs, as a format limitation rather than a failure. For PNG the structure gate also checks that iDOT still points at the same image data. Writes exiftool refuses with only "minor" warnings about the file's own metadata are skipped the same way.

**After upload:**

6. **Confirmation.** The remote fingerprint must match. Three verification levels:
   - `verify = "full"` re-downloads into RAM and compares the whole-file SHA-256. This doubles download traffic.
   - `verify = "sampled"` (default) does the same for 5 % of items.
   - `verify = "fingerprint"` trusts the listing.

**Additive-only rule.** As If writes only to fields that are empty or already carry As If provenance. It never removes or rewrites your data. *As built:* enforced twice, by the writer's plan and by the protected gate, which doesn't trust the writer.

The undo journal records exactly which fields were added. `asif undo --run <id>` strips only those fields and re-uploads, with all gates applied again.

**Hash inside the file.** As If also writes exiftool's `OriginalImageHash` and `OriginalImageHashType` tags once, so any tool can re-verify the pixels years later.

**Cloud-side net.** Providers with version history are a second safety net. Proton Drive keeps previous versions of files in My files; check the retention period on your plan.

---

## 8. Metadata schema

### Model output (enforced via `llama-server` JSON schema)

```json
{
  "type": "object",
  "required": ["caption", "keywords", "has_text", "people_count", "setting"],
  "properties": {
    "caption":      { "type": "string", "maxLength": 240 },
    "keywords":     { "type": "array", "minItems": 3, "maxItems": 15,
                      "items": { "type": "string", "maxLength": 40 } },
    "has_text":     { "type": "boolean" },
    "people_count": { "type": "integer", "minimum": 0, "maximum": 50 },
    "setting":      { "type": "string", "enum": ["indoor", "outdoor", "unclear"] }
  }
}
```

*As built* (`src/asif/infer/schema.json`): the same, plus `additionalProperties: false` and `minLength: 1` on the caption and each keyword.

There is no self-reported "confidence" field, because small models' confidence numbers are unreliable. Quality is measured externally instead, against a golden set (§16).

**Prompt rules** (versioned file `src/asif/infer/prompts/caption_v1.txt`):
- Describe only what is visible.
- Name brands, places or text only if they are legible in the image.
- Never guess identity, age, ethnicity or health.
- Prefer "unclear" over inventing.
- Write in the configured language.

### Field mapping

| Data | Written to | Rule |
|---|---|---|
| caption | `XMP-dc:Description` (lang-alt: `x-default` + configured language) | only if no caption exists (XMP-dc:Description, IPTC Caption-Abstract, EXIF ImageDescription; camera filler ignored) or previously written by As If |
| keywords | `XMP-dc:Subject` (+ `IPTC:Keywords` when `compat = "wide"`) | append + dedupe against XMP and IPTC; IPTC-only keywords are copied into XMP first (UTF-8 or ASCII only, else no XMP keywords are added); never remove existing. With `wide`, a `Photoshop:IPTCDigest` that matched the IPTC (or stood for no IPTC) is updated with it (JPEG); a TIFF with an IPTCDigest gets no IPTC, since exiftool can't keep it in step there, and neither does a JPEG with more than one IPTCDigest or one outside its APP13 Photoshop block (some scanners put one in EXIF), which exiftool would leave as it was; a digest that was already stale is left alone |
| provenance | `XMP-asif:*` (custom namespace via exiftool config file): tool version, model id, model SHA-256, prompt version, run id, UTC timestamp | always |
| integrity | `OriginalImageHash` / `OriginalImageHashType` | once; never overwritten |
| **never touched** | dates, GPS, orientation, ICC, MakerNotes, embedded previews, RAW data | — |

*As built:* the provenance record also says exactly what As If added — `CaptionDigest` (SHA-256 of the caption it wrote) and `CaptionLang`, `AddedKeywords`, `CopiedKeywords` (your IPTC keywords it copied into XMP), `AddedIPTCKeywords`, `WroteOriginalImageHash`, `CreatedIPTC` — so "previously written by As If" means "As If wrote it *and nobody has edited it since*", and undo works from the file alone, even without the vault. `SourceSHA256` (of the whole original) lets a re-run tell a finished copy from one made before the original was edited. An existing `XMP-x:XMPToolkit` stamp is kept. The namespace is defined in `src/asif/meta/asif_ns.config`.

### Formats

- **In-place writes (v1):** JPEG, PNG, WebP, TIFF. *As built:* a file the gates can't prove safe (see §7) is skipped and reported, not written.
- **HEIC/AVIF:** enabled only after the fixture pass proves round-trip safety on *your* exiftool version (check exiftool's supported-formats table).
- **RAW** (DNG, CR3, NEF, ARW…): sidecar `.xmp` only, never written into the file.
- **Video:** out of scope for v1.
- **OCR text:** only the `has_text` flag by default. Copying recognized text into metadata is opt-in, because screenshots of documents, prescriptions and banking apps are exactly what you don't want duplicated into another field.

---

## 9. Models

### 9.1 How to read the numbers

- **"Weights"** means the 4-bit (Q4) GGUF file.
- **Add** the vision projector (≈0.9 GB at F16 for Qwen3.5) plus ~1–2 GB for runtime and context.
- **Context needed is small:** one image plus a short prompt fits in ~4k tokens.
- **Licences:** every recommended model is Apache-2.0, which keeps the project FOSS-clean.

### 9.2 Shortlist

| Tier | Model | Architecture | Weights (Q4) | Fits | Why it's here |
|---|---|---|---|---|---|
| T0 tiny | MiniCPM-V 4.6 | dense 1.3B (SigLIP2-400M encoder + Qwen3.5-0.8B) | ~1.6 GB | any 8 GB machine, CPU | best quality per byte; **listed languages: en, zh only** |
| T0 tiny | Gemma 4 E2B | dense, 2.3B effective | ~3.1 GB | 8 GB RAM, CPU | multilingual T0 fallback |
| T1 small | Gemma 4 E4B | dense, 4.5B effective | ~5.0 GB | 16 GB RAM on CPU, or a 6 GB GPU | sensible CPU default |
| T2 GPU | Qwen3.5-9B | dense 9.6B | ~5.4–6.0 GB + ~0.9 GB projector | 10–12 GB VRAM | best accuracy that fits a mid-range GPU |
| T3 MoE | Gemma 4 26B-A4B | MoE: 25.2B total / 3.8B active | ~17 GB | 32 GB RAM | big-model knowledge at ~4B speed |
| T3 MoE | Qwen3.5-35B-A3B | MoE: 35B total / 3B active | ~20 GB + ~0.86 GB projector | 48 GB+ RAM comfortable; 32 GB tight | strongest CPU-feasible option |

**Ceiling** (rarely worth it for captions): Qwen3.5-27B or Gemma 4 31B, both dense. They need a 24 GB GPU or an Apple machine with 32 GB+.

**Excluded:**
- **Llama 4 Scout.** Too big (109B MoE), and its licence carries an EU clause, which makes it a bad default for a FOSS tool.
- **"Uncensored" or merged community fine-tunes.** Their provenance can't be verified. The manifest pins official repos or well-known quantizers (ggml-org, Unsloth).

### 9.3 MoE on CPU, the honest version

- **What MoE buys.** Text compute drops to the *active* parameters.
- **What it doesn't.** Memory still holds *every* expert: 26B-A4B needs all ~17 GB resident.
- **The vision encoder is dense** and runs once per image whatever the language model is. On CPU it often dominates, so the **#1 CPU lever is input resolution**:
  - cap the longest edge at 768–1024 px;
  - Gemma 4 exposes visual-token budgets from 70 to 1120.
- **Hybrid setups** (8–16 GB GPU + ≥32 GB RAM): keep attention and shared layers on the GPU and the experts in RAM. Use llama.cpp's MoE CPU-offload flags; check `llama-server --help` in the build you pin.

### 9.4 Speed — PLANNING ESTIMATES, not measurements

`asif models bench` replaces these with your real numbers in about 5 minutes.

| Setup | Model | Est. seconds/image | 10k photos |
|---|---|---|---|
| GPU 10–12 GB | Qwen3.5-9B | 2–5 | 6–14 h |
| GPU 6–8 GB | Gemma 4 E4B | 1–3 | 3–8 h |
| Hybrid 12 GB VRAM + 32 GB RAM | T3 MoE | 5–15 | 14–42 h |
| CPU, 8-core desktop | MiniCPM-V 4.6 | 5–15 | 14–42 h |
| CPU, 8-core desktop | Gemma 4 E4B | 15–40 | 2–5 days |
| CPU, 8-core desktop | T3 MoE | 20–60 | 2.5–7 days |

**Anchors behind these guesses:**
- About 16 s/image for a 2.2B VLM on 6 CPU threads (cold start).
- About 32 s for image encoding *alone* on a 16-core Zen 2 with an older Qwen2-VL encoder (~1 MP image).

Transfer time comes on top and depends on provider throttling.

### 9.5 Accuracy

Accuracy is measured, not assumed (§16 golden set). Expect small models to hallucinate fine detail: brand names, small text, counts. The prompt and schema push toward conservative, generic-but-true keywords ("dog", "beach", "sunset"), which is what search needs anyway.

---

## 10. Spec detection → recommendation

**Detect (read-only):**
- CPU model, core count and flags (AVX2 / AVX-512)
- RAM total and free
- GPU vendor and VRAM (`nvidia-smi`, `rocm-smi`, Metal)
- free disk at the staging path
- swap type (zram / encrypted / plain)
- AC vs battery
- OS (and whether it's WSL)
- configured caption language

**Pick:**

```text
candidates = [m in manifest where m.languages covers cfg.caption_lang]

if apple_silicon:   pick largest tier whose (weights + overhead) <= 0.65 * unified_memory
elif vram >= 10 GB: T2 on GPU
elif vram >= 6 GB:  T1 on GPU; offer hybrid T3 if ram >= 28 GB
elif ram >= 44 GB:  T3 Qwen3.5-35B-A3B on CPU
elif ram >= 28 GB:  T3 Gemma 4 26B-A4B on CPU
elif ram >= 12 GB:  T1 on CPU
else:               T0

bench 20 bundled images → projected hours for the library
if projected > cfg.deadline: offer one tier down, and show the quality trade-off
```

**Always print the reason**, e.g. "picked Qwen3.5-9B because 12 GB VRAM; Qwen3.5-35B-A3B would need ~22 GB". When a bigger model would fit but isn't a GPU pick (an MoE on a 19 GB+ GPU), the reason says so instead of quoting a memory figure.

*As built:* RAM is compared as the OS reports it (Linux `MemTotal` is always a little under what's installed), so the thresholds sit below the nominal sizes: 7 GB for an "8 GB" machine, 12 for "16 GB", 28 for "32 GB", 44 for "48 GB". They live in the manifest (`min_ram_gb`), and the hybrid offer uses the T3 model's own threshold. GPUs report a few MiB under their size, so VRAM within 64 MiB of a whole GB counts as that GB.

---

## 11. Batching, linear mode, long-run guards

### Batch sizing

```text
p95       = 95th-percentile file size from the listing
            (8 MB default when the adapter doesn't report sizes)
staging   = min(cfg.staging_cap, free_disk(staging_dir) - max(10 GB, 10 %))
batch_n   = floor(staging / (p95 * 1.02))            # age overhead is tiny
ram_work  = min(cfg.ram_scratch, 0.25 * (free_ram_after_model_load))
in_flight = max(1, floor(ram_work / p95))            # decrypted photos alive at once
double_buffer = staging >= 2 * batch_n * p95         # download batch k+1 while processing k
```

### Linear mode

`batch_n = 1` with no disk staging. Pick it when the machine is weak, disk is tiny, or you want the smallest possible plaintext footprint.

### Guards for multi-day runs

- **Thermal.** Pause when the 60-second rolling median of CPU package temperature stays above the limit for `thermal_window_s` (default 300 s). Resume 10 °C below the limit.
  - **Never** trigger on a single spike; transient spikes are normal.
- **Power.** Refuse to run on battery by default.
  - Inhibit sleep for the duration: `systemd-inhibit` on Linux, `caffeinate -i` on macOS, the execution-state API on Windows.
- **Failure budget.** Abort if more than 2 % of the last 200 items failed. A systematic bug should stop the run, not chew through the library.
- **Pacing.** Cap uploads per hour and bandwidth, to stay inside provider fair use.

---

## 12. Transfer adapters and the login wizard

```python
class Transfer(Protocol):
    def login(self) -> None: ...
    def list(self, scope: str) -> Iterator[RemoteItem]:  # id, path, size?, mtime?, revision?, hash?
        ...
    def fingerprint(self, item: RemoteItem) -> Fingerprint: ...
    def download(self, item: RemoteItem, sink: BinaryIO) -> None:  # sink = encrypting writer
        ...
    def upload_replace(self, item: RemoteItem, src: Path) -> RemoteItem: ...
```

### `proton` — official Proton Drive CLI

**Install.** As If does **not** redistribute the binary. It downloads it from Proton and verifies it against Proton's published checksums.

**Wizard:**
1. Check or install the CLI.
2. `proton-drive auth login` opens the browser; the session goes into the OS secret store.
3. Test with `proton-drive filesystem list /my-files --json`.
4. Choose a scope: a folder, or Photos.
5. `asif plan` shows counts, sizes, batch size and ETA without writing anything.

**Headless servers.** Some wrappers use the CLI's plaintext session-file mode. As If allows it only behind an explicit `--unsafe-session-file` flag, with the file stored on an encrypted volume.

**Photos-timeline caveat.** The CLI exposes photos differently from regular files: third-party wrappers report no folders and no sizes in listings. It is **unverified** whether re-uploading a photo creates a new revision or a duplicate, and whether album membership and favourites survive.
- The cloud canary (§16) tests this on a throwaway photo first.
- Until it passes, write-back to the Photos timeline uses `dest = "mirror:<album>"`, never in-place.

### `rclone` — every other cloud

Covers Nextcloud/WebDAV, S3, SFTP, OneDrive, Google Drive, SMB, and more.

**Wizard:**
1. Run a guided `rclone config`.
2. Force a config password and store it in the OS secret store.
3. The password reaches the transfer sandbox only, via the environment.

**Not for Proton.** The wizard refuses `protondrive` remotes and points to the `proton` adapter instead (deprecated API paths, see §2).

### `ssh` — rsync over SSH (NAS, self-hosted)

- Key-based authentication only.
- The wizard pins the host key on first connect.
- `rsync --checksum` for fingerprints.

### `local` — a folder on this machine (M1)

Folder in, captioned copies out (`asif caption SRC DEST`). The "remote" is the source folder, so the remote-unchanged gate compares the source file's size, mtime and SHA-256. M1 only writes copies into a separate folder, never in place.

---

## 13. Sandboxing per platform

### Linux — bubblewrap

Sketch; the real command is generated by code.

```
bwrap --unshare-all --die-with-parent --new-session \
  --ro-bind /usr /usr --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/bin /bin \
  --proc /proc --dev /dev --tmpfs /tmp \
  --ro-bind /path/to/models /models \
  --ro-bind /path/to/ramwork /work \
  /usr/bin/llama-server -m /models/model.gguf --mmproj /models/mmproj.gguf --host 127.0.0.1 --port 8089
```

- `--unshare-all` creates a fresh network namespace that contains only loopback.
- **GPU** needs explicit device nodes (`--dev-bind` for `/dev/nvidia*` or `/dev/dri/*`) plus the driver libraries; the doctor detects which.
- **llama.cpp settings:** recent llama.cpp builds read `/etc/llama.cpp/config.ini` and `~/.config/llama.cpp/config.ini` before anything As If passes. The sandbox must never expose either, and keeps the worker's environment rules (only its own `LLAMA_*` variables, offline mode under both names, a private empty `XDG_CONFIG_HOME`). Until M3, the worker refuses to start while the system-wide file exists.
- **Limits:** wrap the whole thing in `systemd-run --user --scope -p MemoryMax=… -p CPUQuota=…` to bound memory and CPU.
- On Debian/Ubuntu `/usr/bin/python3` is a symlink into `/etc/alternatives`, so a sandbox that runs Python needs `/etc` (or at least that link) bound too.

### macOS

- **No network namespaces exist**, so isolation relies on a `sandbox-exec` profile that denies all network except loopback.
- `sandbox-exec` is deprecated but still present. **Verify on the current macOS during M10.**
- If the isolation canary fails, As If refuses to run inference unless you pass `--accept-weaker-isolation`, and that choice is recorded in telemetry.

### Windows — WSL2 (the whole tool runs inside Linux)

- WSL2 is a real Linux kernel, so the Linux path, including bubblewrap, applies. Verify bubblewrap in M10.
- Keep **all** As If data inside the WSL filesystem, not under `/mnt/c`. That's faster, and Windows indexers and antivirus don't touch it.

---

## 14. Telemetry — local, isolated, yours

**What it records:**
- per-stage durations
- throughput
- peak RSS and VRAM
- temperatures
- retries and failures (with error class)
- gate outcomes
- model id
- batch sizes

**What it never records:** filenames, paths, captions, keywords, thumbnails, credentials.

**Identity:** each image appears only as a random per-run id (`i_000123`). The mapping lives in the vault, so telemetry alone can't be tied to your files.

**Isolation:**
- The orchestrator is the only writer.
- The telemetry path is never bind-mounted into the transfer, inference or writer sandboxes.
- The module is import-linted to forbid `socket`, `http`, `urllib`, `requests` and friends.

**Encryption:** separate SQLCipher key derived from your passphrase.

**Viewing and export:**
- `asif telemetry report [--since 7d]` prints summaries.
- `asif telemetry export <file>` writes a local file, only on explicit request.

**Example event:**
```json
{"ts":"2026-11-02T03:14:07Z","run":"r_7f3a","img":"i_000123","stage":"infer",
 "ms":4210,"ok":true,"model":"qwen3.5-9b-q4km","rss_mb":9120,"vram_mb":8740,"cpu_c":71.5}
```

*Until M9* a minimal, unencrypted timings log stands in for this (`~/.local/state/asif/timings.jsonl`): same event shape, same "no names, paths or captions" rule, enforced by an allow-list of keys. It records no file sizes either, and photos are processed in a random order, so an id alone says nothing about which file it was. The copies' change times do record the order they were made in, so anyone who can read the destination folder can line the log up with it: the log is no more private than that folder. Durations and outcomes could still loosely correlate with a file someone already knows about; the encrypted store replaces this log.

---

## 15. Duplicates, thumbnails, review, search

**Thumbnails.** 256 px on the longest edge, WebP, stored as BLOBs in the vault, so they're encrypted with it.

**Exact duplicates.** Same `ImageDataHash`.

**Near duplicates.** 64-bit perceptual hash (dHash or pHash) with a Hamming distance ≤ 6 bits to start; tune it on your library.

**The ephemeral list.** During a run, candidate pairs live in an in-memory SQLite database (`:memory:`).
- `asif review dupes` shows them side by side as thumbnails.
- Your decisions (keep both / mark duplicate) are saved to the vault, and the candidate list itself is discarded.
- **Nothing is ever deleted remotely** by As If.

**Search.** `asif search "cachorro na praia"` runs SQLite FTS5 over captions and keywords. It prints thumbnails inline on terminals that support images, or opens the selected photo from the cloud.

**Semantic search (v2).** Optional image embeddings (SigLIP2/CLIP family) stored in the vault, for "find photos like this one".

---

## 16. Test passes ("prove it before the full pass")

`asif run` performs passes 1–4 automatically before touching the library.

1. **Preflight.** Tool versions are pinned and present, plus disk, RAM, swap type, AC power and the clock. (*As built:* `asif doctor` never runs llama-server, since running it outside the sandbox applies its settings files; its version will come from the M3 sandboxed run.)
2. **Isolation canary.** Inside the inference and writer sandboxes, try to open a TCP connection to a public IP and resolve a DNS name.
   - **Both must fail**, or the run aborts.
   - Inference must still work afterwards, proving the sandbox is isolated and not just broken.
3. **Fixture round-trip.** Bundled synthetic images run through describe → write → gates on local copies. The set covers:
   - JPEG with and without EXIF
   - PNG, WebP
   - a motion-photo JPEG with a trailer
   - rotated (Orientation ≠ 1)
   - GPS-tagged
   - truncated / corrupt
   - a 50 MP panorama

   Every gate must behave as expected, including *failing* on the corrupt file.

   *As built:* the set lives in `tests/fixtures/make_fixtures.py` (generated, never real photos) and also covers a progressive JPEG, an MPF multi-picture JPEG, PNG with alpha, lossless WebP, TIFF, and files whose owner already wrote captions, keywords or Latin-1 IPTC.
4. **Cloud canary.** 10 real photos are processed and written to a scratch destination, never in-place. Then:
   - re-download and compare the whole-file SHA-256;
   - you eyeball the captions in `asif review canary`;
   - on the Proton Photos timeline, also check revision-vs-duplicate behaviour (§12).
5. **Golden quality set** (first run per model). Label 30–50 of your own photos by accepting or rejecting the suggested keywords. As If computes keyword precision and recall and shows them next to the bench speed, so you choose the model on evidence.
6. **Full pass.** Continuous invariants: every item passes the gates, the failure budget holds (§11), the thermal guard is active, and the run can resume at any point.

---

## 17. CLI and configuration

```text
asif doctor                          # specs, dependencies, swap, isolation canary
asif models list | pull <id> | verify | bench <id>
asif login proton | rclone | ssh
asif plan --scope /my-files/Photos   # counts, sizes, batch size, ETA — writes nothing
asif test --canary 10                # passes 1–4 only
asif run --mode batch|linear [--dry-run]
asif review dupes | quarantine | canary
asif search "dog on the beach"
asif undo --run <run-id>
asif telemetry report [--since 7d]
asif wipe staging | vault | telemetry | all
```

*Available now (M0–M1):* `asif doctor`, `asif caption SRC DEST` (local folder in, captioned copies out; `--dry-run`, `--limit`, `--lang`, `--compat`, `--max-edge`, `--allow-disk-workdir`), `asif models list | recommend`, `asif config path | show | init`, `asif version`. See the README.

```toml
# ~/.config/asif/config.toml
[general]
caption_lang = "pt-BR"          # BCP-47; filters models by language support
compat       = "xmp"            # "xmp" | "wide" (also mirrors to IPTC)

[source]
adapter = "proton"              # "local" | "proton" | "rclone" | "ssh"
scope   = "/my-files/Photos/2019"
dest    = "in-place"            # or "mirror:/my-files/Photos-captioned"

[resources]
mode           = "batch"        # "batch" | "linear"
staging_dir    = "~/.cache/asif/staging"
staging_cap_gb = 50
ram_scratch_mb = 1024
max_edge_px    = 1024
# allow_disk_workdir = false    # true = on-disk work dir when there's no tmpfs (weaker)

[model]
id     = "auto"                 # or a manifest id, e.g. "qwen3.5-9b-q4km"; "local"; "stub"
device = "auto"                 # "auto" | "cpu" | "gpu" | "hybrid"
# Until `asif models pull` (M2), point at model files you downloaded yourself:
# gguf         = "~/models/model.Q4_K_M.gguf"
# mmproj       = "~/models/mmproj-F16.gguf"
# llama_server = "llama-server"  # or a script that ends with: exec llama-server "$@"
# server_args  = []              # allow-listed speed/memory/GPU/logging options only

[upload]
verify               = "sampled" # "full" | "sampled" | "fingerprint"
sample_rate          = 0.05
max_uploads_per_hour = 300
bandwidth            = "10M"

[guards]
cpu_temp_limit_c  = 90
thermal_window_s  = 300
max_failure_rate  = 0.02
require_ac_power  = true          # M7; not enforced yet (doctor only warns)

[telemetry]
enabled        = true
retention_days = 90
```

**Model manifest entry** (`src/asif/models/manifest.toml`; revision, names and hashes filled in when pinning):

```toml
[[model]]
id           = "qwen3.5-9b-q4km"
name         = "Qwen3.5-9B"
tier         = "T2"
arch         = "dense"
params       = "9.6B"
weights_gb   = 6.0
projector_gb = 0.9
min_vram_gb  = 10.0
min_ram_gb   = 12.0
languages    = ["*"]               # or primary subtags, e.g. ["en", "zh"]
licence      = "Apache-2.0"
repo         = "unsloth/Qwen3.5-9B-GGUF"
revision     = "<pinned commit>"
notes        = "Best accuracy that fits a mid-range GPU."
files = [
    { role = "weights",   name = "<model>.Q4_K_M.gguf", sha256 = "<…>" },
    { role = "projector", name = "<mmproj>-F16.gguf",   sha256 = "<…>" },
]
```

A catalogue id is only written into `XMP-asif:ModelId` when the file's SHA-256 matches its pinned hash; otherwise the copy says `local:<file name>`.

---

## 18. Stack and repository layout

**Language:** Python 3.12+, managed with `uv`. Rationale: subprocess orchestration, exiftool bindings and SQLite are all first-class in Python, and it's easy for contributors.

| Concern | Choice | Note |
|---|---|---|
| CLI / wizard | Typer + Rich (Textual TUI later) | |
| Schemas | pydantic | validates model JSON |
| Inference | llama.cpp `llama-server` (pinned release) | CUDA / ROCm / Vulkan / Metal builds |
| Metadata | exiftool in `-stay_open` mode | our own small wrapper (`src/asif/meta/exiftool.py`) |
| Staging encryption | `age` (binary, or a maintained binding — evaluate) | streaming, simple, audited format |
| Vault | SQLCipher (evaluate `sqlcipher3`) | FTS5 for search |
| Images | Pillow (+ pillow-heif) | thumbnails, perceptual hashes |
| Sandboxing | bubblewrap / sandbox-exec / WSL2 + bubblewrap | |

```
as-if/
├── pyproject.toml
├── src/asif/
│   ├── cli.py
│   ├── config.py, paths.py      # typed config.toml, XDG / macOS locations
│   ├── netguard.py              # audit-hook tripwire: no sockets in the orchestrator
│   ├── workdir.py               # the RAM work dir
│   ├── doctor.py                # spec detection, preflight
│   ├── models/                  # manifest.toml, manifest.py, recommend.py (pull/verify/bench: M2)
│   ├── transfer/                # base.py, proton_cli.py, rclone.py, ssh_rsync.py   (M5)
│   ├── sandbox/                 # linux_bwrap.py, macos_seatbelt.py, canary.py     (M3)
│   ├── crypto/                  # keys.py, staging.py, vault.py                     (M4)
│   ├── pipeline/                # states.py, local.py, timings.py (journal/batch/guards: M7)
│   ├── infer/                   # worker.py (runs inside sandbox), client.py, preprocess.py, prompts/, schema.json
│   ├── meta/                    # exiftool.py, writer.py, gates.py, structure.py, asif_ns.config (undo: M6)
│   ├── dedupe/                  # hashes.py, review.py                              (M8)
│   ├── search/                  # fts.py                                            (M8)
│   └── telemetry/               # writer.py, report.py   (no network modules)       (M9)
├── tests/
│   ├── fixtures/                # make_fixtures.py: synthetic images (§16 pass 3)
│   └── test_*.py
└── docs/
```

---

## 19. Build plan — one milestone at a time, each with a checkpoint

| # | Milestone | Verification checkpoint |
|---|---|---|
| M0 | Skeleton + `doctor` | doctor's output matches `lscpu`, `free -h`, `nvidia-smi` on your machine |
| M1 | **Local-only core**: folder in → copies out, no cloud, no sandbox, no encryption; minimal timings log | 100 photos captioned; `ImageDataHash` equal on all 100; dates and GPS byte-identical |
| M2 | Model manifest, pull, verify, bench | a deliberately corrupted model file is rejected; bench prints s/image |
| M3 | Inference + writer sandboxes, isolation canary | canary connect **fails**, captioning still **works** |
| M4 | Crypto: staging, RAM work dir, vault | searching `staging/` for JPEG magic bytes finds none; `kill -9` mid-run leaves no plaintext on disk |
| M5 | Transfer adapters (local/SFTP via rclone first, then Proton) + login wizard | cloud canary to a scratch folder; re-download hash equals the local processed file |
| M6 | Write-back gates + undo | edit a remote file mid-run → `SKIPPED_CONFLICT`; `undo` restores byte-identical image data |
| M7 | Batch scheduler, linear mode, resume, guards | simulated power cut → resume with zero duplicate uploads |
| M8 | Dupes, thumbnails, review, search | known duplicate pairs in fixtures are all suggested; nothing is deleted |
| M9 | Telemetry hardening | telemetry DB contains no filenames (`strings` check); report renders |
| M10 | macOS + Windows/WSL2 ports | the full test-pass suite green on each platform |
| M11 | Release: reproducible builds, signed releases, docs | a stranger installs it from the README in under 15 minutes |

**Why M1 comes first, without cloud or crypto:** the riskiest logic is the metadata writer and its gates. Prove those on local copies before anything can touch a real library.

---

## 20. Platform notes

### Linux (primary)

- **Dependencies:** llama.cpp (a release build matching your GPU backend), exiftool, bubblewrap, age, rclone (optional). Package names differ between distros, so check your package manager.
- **Swap:** prefer zram or encrypted swap; the doctor warns otherwise.
- **Long runs:** run under `systemd-inhibit`. For a headless server, consider a `systemd --user` service with lingering enabled.

### macOS

- **Install:** `brew install llama.cpp exiftool age` (and `rclone` if needed).
- **Proton CLI:** download the binary, `chmod +x` it, and clear the quarantine attribute if Gatekeeper blocks it.
- **Memory:** Metal acceleration works, and unified memory makes T3 MoE models practical on 32 GB+ Macs. Leave headroom, because macOS reserves part of unified memory.
- **RAM work dir:** create a RAM disk (`hdiutil` + `diskutil`). Run under `caffeinate -i`.

### Windows (via WSL2)

- **Install:** `wsl --install -d Ubuntu-24.04`, then follow the Linux path inside it.
- **GPU (NVIDIA):** install only the Windows driver; CUDA is exposed inside WSL. Other GPU vendors: check current WSL support before promising it.
- **Memory:** WSL caps memory below your total RAM by default. Set `memory=`, `processors=` and `swap=` in `%UserProfile%\.wslconfig` so a T3 model fits.
- **Login:** if `proton-drive auth login` can't open a browser from WSL, set `BROWSER` to a Windows-bridging opener (e.g. `wslview`).
- **Secret store:** WSL has no Secret Service by default. Either run a headless keyring daemon inside WSL, or run the transfer adapter Windows-side (`proton-drive.exe` uses Credential Manager) while inference stays in WSL. Decide in M10.

---

## 21. Risks and open questions

1. **Proton Photos write-back semantics** (revision vs duplicate, album membership). The cloud canary decides; mirror mode is used until then.
2. **HEIC/AVIF in-place writes.** The fixture pass decides per exiftool version.
3. **Motion-photo trailers.** The trailer gate skips anything it can't prove intact.
4. **Fair-use limits** for ~10k re-uploads. Pacing defaults are conservative; spread the work over days.
5. **macOS isolation** depends on a deprecated mechanism.
6. **WSL secret storage** friction.
7. **Hallucination rate** on personal photos. The golden set measures it; conservative prompts reduce it.
8. **Model churn.** New VLMs ship monthly, so the manifest plus bench plus golden set make swapping cheap.
9. **Ente has no write path.** Out of scope unless Ente ships a metadata API.
10. **XMP namespace URI.** `XMP-asif` uses the provisional URI `http://ns.as-if.invalid/1.0/`. Pick the permanent one before 1.0; tags written under the old URI would need migrating.

---

## 22. Licence and distribution

- **App licence:** the repository currently carries the **MIT** licence (`LICENSE`). The original plan weighed GPL-3.0-or-later (keeps forks open) against Apache-2.0 (permissive); settle it before the first public release.
- **Don't redistribute** Proton's CLI, rclone or model weights. Download them from upstream and verify pinned checksums. This is the same approach existing Proton CLI wrappers take.
- **Reproducible builds** + signed releases + an SBOM, so people can check that the privacy claims match the binary.

---

## Sources (checked October 2026)

- Proton — Introducing Proton Drive CLI: https://proton.me/blog/proton-drive-cli
- Proton — Drive roadmap (CLI photos & albums): https://proton.me/drive/roadmap
- Proton — Autumn/winter 2026 roadmaps (search, Linux beta): https://proton.me/blog/2026-autumn-winter-roadmaps
- Proton — Photo filters in Drive: https://proton.me/support/drive-photo-filters
- rclone forum — Proton engineer on rclone and deprecated endpoints: https://forum.rclone.org/t/proton-drive-x-rclone/53609
- Protondrive-for-Linux (wrapper; checksum-verified CLI download): https://github.com/ColinMario/Protondrive-for-Linux
- protondrive-gui releases (CLI photo listing behaviour): https://github.com/al3xg0r/protondrive-gui/releases
- Ente help — export, search, EXIF parsing: https://help.ente.io/photos/features/backup-and-sync/export · https://help.ente.io/photos/faq/misc
- Unsloth — Qwen3.5 (sizes; Ollama vision limitation): https://unsloth.ai/docs/models/qwen3.5.md
- Qwen3.5-9B VLM GGUF (sizes): https://huggingface.co/jc-builds/Qwen3.5-9B-VLM-Q4_K_M-GGUF
- Qwen3.5-35B-A3B quant sizes: https://huggingface.co/mudler/Qwen3.5-35B-A3B-APEX-GGUF
- Gemma 4 specs and GGUF sizes: https://huggingface.co/google/gemma-4-E4B · https://www.dsebastien.net/gemma-4/
- MiniCPM-V 4.6: https://huggingface.co/ggml-org/MiniCPM-V-4.6-GGUF · https://www.datalearner.com/en/ai-models/pretrained-models/minicpm-v-4-6
- CPU vision-encode timings: https://huggingface.co/kelnei/slugvision-2.2b · https://zenn.dev/syoyo/articles/ced4c9bfeefeb2
- llama.cpp issue — slow CPU image encoding in llama-server: https://github.com/ggml-org/llama.cpp/issues/22582
- exiftool forum — ImageDataHash and OriginalImageHash: https://exiftool.org/forum/index.php?msg=79216 · https://exiftool.org/forum/index.php?msg=91746
