"""
Home Brain — Entry Point

What it is: CLI entry point and command dispatcher.
What it knows: available commands, how to parse argv, how to call the container.
What it doesn't know: domain logic, adapters — it only touches the container and ports.

Usage (run from the repo root):
  python -m home_brain run              # extract clips for yesterday (production)
  python -m home_brain run 2025-12-01  # extract clips for a specific date
  python -m home_brain test            # record 60s right now + extract (dev)
  python -m home_brain list [date]     # list stored clips for a date
  python -m home_brain schedule        # daily cron-style scheduler loop
  python -m home_brain analyse [date]  # analyse pending clips (Phase 2)
  python -m home_brain summarise [date] # generate daily summary (Phase 2)
"""
import os
import sys
from datetime import date, datetime, timedelta
from typing import Optional

from home_brain.container import build_container
from home_brain.shared.logger import get_logger

logger = get_logger("home_brain.main")


def cmd_run(target_date: Optional[date] = None) -> None:
    """Run extraction for a specific date (default: yesterday)."""
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    logger.info(f"=== Home Brain | Daily Extraction | {target_date} ===")
    container = build_container()
    clips = container.ingestion_service.run_daily_extraction(target_date)

    print(f"\nExtracted {len(clips)} clips for {target_date}")
    total_mb = sum((c.size_bytes or 0) for c in clips) / 1024 / 1024
    total_s = sum(c.duration_seconds for c in clips)
    print(f"   Total activity: {total_s:.0f}s | Disk used: {total_mb:.1f} MB")
    print(f"\n   Clips:")
    for clip in clips:
        print(
            f"   [{clip.recorded_at.strftime('%H:%M:%S')}] "
            f"{clip.duration_seconds:.0f}s | "
            f"{(clip.size_bytes or 0)/1024:.0f} KB | "
            f"{clip.file_path.name}"
        )


def cmd_test() -> None:
    """
    Dev mode: record 60s from the live stream right now, then extract clips.
    Overrides RECORD_DURATION_S=60 so config picks it up.
    """
    os.environ["RECORD_DURATION_S"] = "60"
    logger.info("=== TEST MODE: Recording 60s from live stream ===")
    cmd_run(target_date=date.today())


def cmd_list(target_date: Optional[date] = None) -> None:
    """List clips stored for a given date."""
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    container = build_container()
    clips = container.clip_storage.find_by_date(target_date)

    if not clips:
        print(f"No clips found for {target_date}")
        return

    print(f"\nClips for {target_date} ({len(clips)} total):\n")
    for clip in clips:
        size = f"{clip.size_bytes/1024:.0f}KB" if clip.size_bytes else "unknown"
        print(
            f"  [{clip.recorded_at.strftime('%H:%M:%S')}] "
            f"{clip.duration_seconds:.0f}s | {size} | "
            f"status={clip.status.value} | {clip.file_path.name}"
        )


def cmd_analyse(target_date: Optional[date] = None) -> None:
    """Analyse all pending clips with Claude Vision + Whisper (Phase 2)."""
    if target_date is None:
        target_date = date.today()

    logger.info(f"=== Home Brain | Clip Analysis | {target_date} ===")
    container = build_container()
    observations = container.clip_analysis_service.analyse_pending_clips()

    print(f"\nAnalysed {len(observations)} clip(s).")
    for obs in observations:
        print(
            f"  [{obs.clip_id[:8]}] mood={obs.mood.value} "
            f"confidence={obs.confidence:.2f} | {obs.activity}"
        )


def cmd_summarise(target_date: Optional[date] = None) -> None:
    """Generate a DailySummary from all Observations for a date (Phase 2)."""
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    logger.info(f"=== Home Brain | Daily Summary | {target_date} ===")
    container = build_container()
    try:
        summary = container.clip_analysis_service.generate_daily_summary(target_date)
        print(f"\nSummary for {target_date}:")
        print(f"  {summary.summary_text}")
        print(f"  Moods: {summary.mood_counts}")
        print(f"  Highlights: {summary.highlight_clip_ids}")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_schedule() -> None:
    """
    Runs as a daily scheduler.
    Triggers extraction every day at 06:00 AM for the previous day.
    For production, prefer system cron instead:
      0 6 * * * cd /path/to/project && python -m home_brain run
    """
    import time
    logger.info("Scheduler started. Will run extraction daily at 06:00.")

    while True:
        now = datetime.now()
        next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)

        wait_seconds = (next_run - now).total_seconds()
        logger.info(f"Next extraction at {next_run} (in {wait_seconds/3600:.1f}h)")
        time.sleep(wait_seconds)

        try:
            cmd_run()
        except Exception as e:
            logger.error(f"Scheduled extraction failed: {e}")


def main() -> None:
    args = sys.argv[1:]
    command = args[0] if args else "run"

    if command == "run":
        target = date.fromisoformat(args[1]) if len(args) > 1 else None
        cmd_run(target)

    elif command == "test":
        cmd_test()

    elif command == "list":
        target = date.fromisoformat(args[1]) if len(args) > 1 else None
        cmd_list(target)

    elif command == "schedule":
        cmd_schedule()

    elif command == "analyse":
        target = date.fromisoformat(args[1]) if len(args) > 1 else None
        cmd_analyse(target)

    elif command == "summarise":
        target = date.fromisoformat(args[1]) if len(args) > 1 else None
        cmd_summarise(target)

    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m home_brain [run|test|list|schedule|analyse|summarise] [date]")
        sys.exit(1)


if __name__ == "__main__":
    main()
