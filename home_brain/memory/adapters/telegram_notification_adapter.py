"""
Memory Adapters — Telegram Notification.

What it is: Outbound notification adapter that sends a DailySummary and highlight
            clip videos to the owner via Telegram.
What it knows: python-telegram-bot async API, file size limits, message formatting.
What it doesn't know: domain rules, SQLite, B2, ffmpeg, or any other adapter.

python-telegram-bot v20+ is async-only. The public interface is kept synchronous
via asyncio.run() so callers (MemoryService) don't need to be async.
"""

import asyncio
from pathlib import Path
from typing import List

import telegram

from home_brain.intelligence.domain.entities.observation import DailySummary
from home_brain.memory.domain.ports.ports import INotificationPort
from home_brain.shared.logger import get_logger

logger = get_logger("memory.adapters.telegram")

_MAX_VIDEO_BYTES = 50 * 1024 * 1024  # 50 MB — Telegram bot API limit


class TelegramNotificationAdapter(INotificationPort):
    """
    Sends daily summary text + up to 3 highlight clip videos via Telegram Bot API.

    Validates credentials on construction. Large (>50 MB) or missing clip files
    are skipped gracefully with a warning log — no exception raised.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        if not bot_token or not bot_token.strip():
            raise ValueError("bot_token must not be empty")
        if not chat_id or not chat_id.strip():
            raise ValueError("chat_id must not be empty")
        self._bot_token = bot_token
        self._chat_id = chat_id

    def send_daily_summary(
        self,
        summary: DailySummary,
        clip_paths: List[Path],
    ) -> None:
        asyncio.run(self._async_send(summary, clip_paths))

    async def _async_send(
        self,
        summary: DailySummary,
        clip_paths: List[Path],
    ) -> None:
        text = self._format_message(summary)

        async with telegram.Bot(token=self._bot_token) as bot:
            await bot.send_message(chat_id=self._chat_id, text=text)
            logger.info(f"Sent daily summary text for {summary.date}")

            for path in clip_paths:
                if not path.exists():
                    logger.warning(f"Clip file not found, skipping: {path}")
                    continue
                size = path.stat().st_size
                if size > _MAX_VIDEO_BYTES:
                    logger.warning(
                        f"Clip {path.name} is {size / 1024 / 1024:.1f} MB > 50 MB, skipping"
                    )
                    continue
                await bot.send_video(chat_id=self._chat_id, video=path)
                logger.info(f"Sent video: {path.name}")

    def _format_message(self, summary: DailySummary) -> str:
        mood_line = ", ".join(
            f"{mood}: {count}" for mood, count in summary.mood_counts.items()
        )
        return (
            f"Home Brain — Daily Summary\n"
            f"Date: {summary.date}\n\n"
            f"{summary.summary_text}\n\n"
            f"Moods: {mood_line}"
        )
