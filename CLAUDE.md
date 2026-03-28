# CLAUDE.md — Home Brain Project

This file is the **single source of truth** for Claude Code working on this project.
Read it fully before making any change. Every section is a non-negotiable constraint.

---

## What This Project Is

**Home Brain** is a self-hosted home intelligence system built around a CP Plus CP-E44Q
Wi-Fi security camera. It pulls footage via RTSP, detects motion, extracts meaningful
clips, analyses them with Claude Vision (mood + activity), transcribes audio via Whisper,
and delivers a daily summary to Telegram. Old footage archives to Backblaze B2 after 30 days.

**The north star:** Turn raw surveillance footage into a warm, searchable memory of home life —
eventually a plugin in a Second Brain system.

**Hardware constraint:** Always-on laptop. No cloud compute. No Docker. Self-hosted only.

---

## Architecture: DDD + Hexagonal (Ports & Adapters)

This is non-negotiable. Every piece of code must honour this structure.

### The Rule
```
Domain → knows nothing about infrastructure
Ports  → interfaces the domain defines
Adapters → implement ports, live outside the domain
Container → the only place that wires adapters to ports
```

### Bounded Contexts (one folder per context)

| Context | Folder | Responsibility |
|---|---|---|
| Surveillance | `surveillance/` | RTSP → motion → clip extraction |
| Intelligence | `intelligence/` | Clip → mood + transcript + observation |
| Memory | `memory/` | Summary → Telegram + B2 archive |

### Dependency Rule (strict)
```
shared/        ← imported by everyone (config, logger only)
domain/        ← imports nothing outside its own bounded context
adapters/      ← imports domain ports + external libraries
container.py   ← imports everything, wires it all together
__main__.py    ← imports container only
```

**Never:** import an adapter inside a domain service.
**Never:** import between bounded contexts directly — use domain events or the container.
**Never:** put infrastructure terms (ffmpeg, sqlite, boto3) inside domain entities or services.

---

## Ubiquitous Language

Use these exact terms in code, comments, and variable names. No synonyms.

| Term | Meaning |
|---|---|
| `Stream` | The live RTSP feed from the camera |
| `Recording` | A saved full-day video file (raw, unprocessed) |
| `MotionWindow` | A time range where significant activity was detected |
| `Clip` | A meaningful video segment extracted from a MotionWindow |
| `Observation` | Claude's analysis of a Clip (mood + activity + description) |
| `Transcript` | Whisper's audio-to-text output for a Clip |
| `Mood` | Emotional category: happy / sad / angry / neutral / calm |
| `DailySummary` | Claude's narrative of a full day's Observations |
| `Highlight` | A top-ranked Clip selected for the daily summary notification |
| `Archive` | Cold storage of clips older than 30 days (Backblaze B2) |
| `Notification` | Telegram message delivered to the owner |

---

## Current State — Phase 1 (COMPLETE)

### What exists
- `shared/config.py` — all environment config via `.env`
- `shared/logger.py` — structured logging
- `surveillance/domain/entities/clip.py` — `Clip`, `MotionWindow`, `ClipStatus`
- `surveillance/domain/ports/ports.py` — `IStreamIngestionPort`, `IMotionDetectionPort`, `IStreamRecorderPort`, `IClipStoragePort`
- `surveillance/domain/services/stream_ingestion_service.py` — full pipeline orchestration
- `surveillance/adapters/ffmpeg_adapter.py` — RTSP recording + scene detection
- `surveillance/adapters/sqlite_clip_storage_adapter.py` — persistence
- `container.py` — DI wiring
- `__main__.py` — CLI: `run`, `test`, `list`, `schedule`

### SQLite schema (in sqlite_clip_storage_adapter.py)
```sql
clips(id, recorded_at, duration_s, file_path, status, size_bytes, archive_url, created_at)
```

### CLI commands
```bash
python -m home_brain test          # Record 60s live → extract clips
python -m home_brain run           # Extract for yesterday
python -m home_brain run 2025-12-01
python -m home_brain list [date]
python -m home_brain schedule      # Daily cron loop
```

---

## Roadmap — What to Build Next

### Phase 2 — Intelligence Domain
New bounded context: `intelligence/`

**Entities to create:**
- `Observation(id, clip_id, mood, activity, description, confidence, observed_at)`
- `Transcript(id, clip_id, text, language, observed_at)`
- `DailySummary(id, date, summary_text, mood_counts, highlight_clip_ids, notified)`

**Ports to create:**
- `IObservationStorePort` — save/query Observations
- `IVideoAnalysisPort` — outbound: send clip frames → get Observation back
- `ITranscriptionPort` — outbound: send audio → get Transcript back
- `IClipQueryPort` — outbound: get pending clips from Surveillance domain

**Adapters to create:**
- `ClaudeVisionAdapter` — implements `IVideoAnalysisPort`
  - Use `claude-haiku-4-5-20251001` for per-clip analysis (cost: ~₹2–3/clip)
  - Use `claude-sonnet-4-6` for daily summary (once/day)
  - Extract 3–5 key frames per clip via ffmpeg before sending
  - Prompt must return structured JSON: `{mood, activity, description, confidence}`
- `WhisperTranscriptionAdapter` — implements `ITranscriptionPort`
  - Use `faster-whisper` (CPU-optimised, runs on laptop)
  - Model: `small` or `base` — fast enough, good accuracy for home audio
- `SQLiteObservationAdapter` — implements `IObservationStorePort`
  - Same SQLite file: `home_brain.sqlite`
  - New tables: `observations`, `transcripts`, `daily_summaries`

**Handoff from Phase 1:**
`IClipStoragePort.find_pending()` returns all `status=pending` clips.
Intelligence domain processes them, then calls `clip.mark_analysed()` via the storage port.

### Phase 3 — Memory Domain
New bounded context: `memory/`

**Ports to create:**
- `INotificationPort` — outbound: send Telegram message + clips
- `IArchivePort` — outbound: upload to Backblaze B2, update clip record

**Adapters to create:**
- `TelegramNotificationAdapter` — `python-telegram-bot` library
  - Send daily summary text
  - Attach top 3 Highlight clips as video messages
- `BackblazeB2ArchiveAdapter` — `boto3` with B2 endpoint
  - Trigger: clips older than `RETENTION_DAYS` (default 30)
  - After upload: set `clip.archive_url`, delete local file
  - Keep SQLite record forever (it's tiny, it's your index)

---

## Hard Constraints — Never Violate

1. **No Docker.** Use system Python, cron, and local processes only.
2. **No cloud compute.** Anthropic API for AI only. Everything else runs on the laptop.
3. **No framework DI containers** (no FastAPI DI, no inject library). Manual wiring in `container.py`.
4. **No cross-domain adapter imports.** Domains communicate via ports or domain events only.
5. **No hardcoded credentials.** All secrets via `.env` → `config.py`. Never in code.
6. **SQLite is the primary database.** Do not introduce Postgres, Redis, or any server DB.
7. **ffmpeg is the only video tool.** Do not add OpenCV unless ffmpeg truly cannot do the job.
8. **Backblaze B2 for cold archive.** Do not use AWS S3, GCP, or Azure.
9. **Telegram for notifications.** Do not add email, push notifications, or other channels.
10. **Claude Haiku for per-clip analysis** (cost-sensitive). Claude Sonnet for daily summaries only.

---

## Cost Budget (monthly target: < ₹150)

| Item | Tool | Target |
|---|---|---|
| Per-clip AI analysis | Claude Haiku | ₹30–50 |
| Daily summary | Claude Sonnet | ₹15–20 |
| Transcription | Whisper (local) | ₹0 |
| 30-day video archive | Backblaze B2 | ₹50 |
| Notifications | Telegram Bot | ₹0 |
| **Total** | | **< ₹120** |

When adding AI calls: always use Haiku first. Only escalate to Sonnet when Haiku genuinely
can't do the job. Always log token counts so cost is visible.

---

## Code Standards

### File naming
```
entities/     → noun.py           (clip.py, observation.py)
ports/        → ports.py          (one file per bounded context)
services/     → noun_verb_service.py  (stream_ingestion_service.py)
adapters/     → library_noun_adapter.py  (ffmpeg_adapter.py, sqlite_clip_storage_adapter.py)
```

### Class naming
```
Entities:  PascalCase noun         → Clip, Observation, DailySummary
Ports:     I + PascalCase + Port   → IClipStoragePort, IVideoAnalysisPort
Services:  PascalCase + Service    → StreamIngestionService
Adapters:  Library + Noun + Adapter → ClaudeVisionAdapter, SQLiteObservationAdapter
```

### Every new file must have
1. A module-level docstring explaining: what it is, what it knows, what it doesn't know.
2. Logger: `logger = get_logger("boundedcontext.ClassName")`
3. Type hints on all public methods.
4. No `print()` statements — use `logger.info/debug/error`.

### Adding a new Bounded Context checklist
- [ ] Create `context_name/` folder with `__init__.py`
- [ ] Create `domain/entities/`, `domain/ports/`, `domain/services/`
- [ ] Create `adapters/`
- [ ] Define all ports as ABCs in `ports/ports.py` before writing any adapter
- [ ] Add adapters to `container.py` — never import them elsewhere
- [ ] Add any new tables to SQLite with `CREATE TABLE IF NOT EXISTS` in the adapter

---

## Environment Variables Reference

```bash
# Camera
CAMERA_IP          # Static IP of CP Plus CP-E44Q on local network
CAMERA_PORT        # Default: 554
CAMERA_USER        # Default: admin
CAMERA_PASS        # Your camera password
CAMERA_CHANNEL     # Default: 1
CAMERA_SUBTYPE     # 0=main(4MP), 1=sub(360p, recommended)

# Storage
DATA_DIR           # Default: ./data
RETENTION_DAYS     # Default: 30

# Motion detection
SCENE_THRESHOLD    # 0.1–0.5, default 0.3
MIN_CLIP_DURATION  # seconds, default 5
MAX_CLIP_DURATION  # seconds, default 120
MERGE_GAP_S        # seconds, default 10

# Dev
RECORD_DURATION_S  # Default: 86400 (24h). Set to 60 for testing.

# Phase 2 (not yet in use)
ANTHROPIC_API_KEY  # For Claude Vision + Sonnet summary
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
B2_APPLICATION_KEY_ID
B2_APPLICATION_KEY
B2_BUCKET_NAME
```

---

## Camera Technical Notes

- **Model:** CP Plus CP-E44Q, 4MP Wi-Fi
- **RTSP URL (sub-stream):** `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=1`
- **RTSP URL (main stream):** `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0`
- **Protocol:** Use `-rtsp_transport tcp` in ffmpeg (more stable on home WiFi)
- **WiFi:** 2.4GHz only — camera cannot connect to 5GHz
- **Cloud:** ezykam+ cloud subscription is separate from this system. This pipeline is fully independent of the ezykam+ app/cloud.
- **Audio:** Camera has a microphone. Audio is included in the RTSP stream. Whisper will use this.

---

## Data Lifecycle

```
Day 0–30:   Clips live at  ./data/clips/YYYY-MM-DD/*.mp4
            DB lives at    ./data/home_brain.sqlite

Day 31+:    Nightly archive job (MemoryService):
            1. Compress clip (ffmpeg -crf 28)
            2. Upload to Backblaze B2
            3. Set clip.archive_url in SQLite
            4. Delete local .mp4
            5. SQLite record stays forever (it's your searchable index)

Quarterly:  SQLite snapshot uploaded to B2 as well
```
