"""
Tests for TelegramNotificationAdapter — INotificationPort.

Uses unittest.mock to patch the telegram library. No real Telegram API calls.
Negative tests first, then positive.
"""

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from home_brain.intelligence.domain.entities.observation import DailySummary
from home_brain.memory.adapters.telegram_notification_adapter import (
    TelegramNotificationAdapter,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_summary(notified: bool = False) -> DailySummary:
    return DailySummary(
        id="sum-1",
        date=date(2026, 3, 28),
        summary_text="A calm day at home.",
        mood_counts={"calm": 3, "happy": 1},
        highlight_clip_ids=["clip-1", "clip-2"],
        notified=notified,
    )


def _make_adapter() -> TelegramNotificationAdapter:
    return TelegramNotificationAdapter(
        bot_token="test-token-123",
        chat_id="99999999",
    )


# ---------------------------------------------------------------------------
# NEGATIVE TESTS
# ---------------------------------------------------------------------------


class TestTelegramNotificationAdapterNegative:
    def test_empty_bot_token_raises_value_error(self):
        with pytest.raises(ValueError, match="bot_token"):
            TelegramNotificationAdapter(bot_token="", chat_id="123")

    def test_whitespace_bot_token_raises_value_error(self):
        with pytest.raises(ValueError, match="bot_token"):
            TelegramNotificationAdapter(bot_token="   ", chat_id="123")

    def test_empty_chat_id_raises_value_error(self):
        with pytest.raises(ValueError, match="chat_id"):
            TelegramNotificationAdapter(bot_token="token-abc", chat_id="")

    def test_large_video_skipped_gracefully(self, tmp_path):
        """Clips > 50 MB are skipped — no error raised."""
        clip_path = tmp_path / "big.mp4"
        clip_path.write_bytes(b"x")  # small file on disk

        adapter = _make_adapter()
        summary = _make_summary()

        with patch(
            "home_brain.memory.adapters.telegram_notification_adapter.telegram"
        ) as mock_tg:
            mock_bot = AsyncMock()
            mock_bot.send_message = AsyncMock()
            mock_bot.send_video = AsyncMock()
            mock_tg.Bot.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
            mock_tg.Bot.return_value.__aexit__ = AsyncMock(return_value=False)

            # Patch stat to return a size > 50 MB
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 51 * 1024 * 1024
                adapter.send_daily_summary(summary, [clip_path])

        # send_video should NOT have been called
        mock_bot.send_video.assert_not_called()
        # But send_message SHOULD have been called once (text message)
        mock_bot.send_message.assert_called_once()

    def test_nonexistent_clip_path_skipped_gracefully(self, tmp_path):
        """Missing clip file is silently skipped."""
        missing = tmp_path / "ghost.mp4"
        adapter = _make_adapter()
        summary = _make_summary()

        with patch(
            "home_brain.memory.adapters.telegram_notification_adapter.telegram"
        ) as mock_tg:
            mock_bot = AsyncMock()
            mock_bot.send_message = AsyncMock()
            mock_bot.send_video = AsyncMock()
            mock_tg.Bot.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
            mock_tg.Bot.return_value.__aexit__ = AsyncMock(return_value=False)

            adapter.send_daily_summary(summary, [missing])

        mock_bot.send_video.assert_not_called()
        mock_bot.send_message.assert_called_once()


# ---------------------------------------------------------------------------
# POSITIVE TESTS
# ---------------------------------------------------------------------------


class TestTelegramNotificationAdapterPositive:
    def test_no_clips_sends_one_text_message(self):
        adapter = _make_adapter()
        summary = _make_summary()

        with patch(
            "home_brain.memory.adapters.telegram_notification_adapter.telegram"
        ) as mock_tg:
            mock_bot = AsyncMock()
            mock_bot.send_message = AsyncMock()
            mock_bot.send_video = AsyncMock()
            mock_tg.Bot.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
            mock_tg.Bot.return_value.__aexit__ = AsyncMock(return_value=False)

            adapter.send_daily_summary(summary, [])

        mock_bot.send_message.assert_called_once()
        mock_bot.send_video.assert_not_called()

    def test_text_message_contains_summary_text(self):
        adapter = _make_adapter()
        summary = _make_summary()

        with patch(
            "home_brain.memory.adapters.telegram_notification_adapter.telegram"
        ) as mock_tg:
            mock_bot = AsyncMock()
            mock_bot.send_message = AsyncMock()
            mock_bot.send_video = AsyncMock()
            mock_tg.Bot.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
            mock_tg.Bot.return_value.__aexit__ = AsyncMock(return_value=False)

            adapter.send_daily_summary(summary, [])

        call_kwargs = mock_bot.send_message.call_args
        text_arg = call_kwargs[1].get("text") or call_kwargs[0][1]
        assert "A calm day at home." in text_arg

    def test_text_message_contains_date(self):
        adapter = _make_adapter()
        summary = _make_summary()

        with patch(
            "home_brain.memory.adapters.telegram_notification_adapter.telegram"
        ) as mock_tg:
            mock_bot = AsyncMock()
            mock_bot.send_message = AsyncMock()
            mock_bot.send_video = AsyncMock()
            mock_tg.Bot.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
            mock_tg.Bot.return_value.__aexit__ = AsyncMock(return_value=False)

            adapter.send_daily_summary(summary, [])

        call_kwargs = mock_bot.send_message.call_args
        text_arg = call_kwargs[1].get("text") or call_kwargs[0][1]
        assert "2026-03-28" in text_arg

    def test_three_small_clips_sends_three_videos(self, tmp_path):
        clips = []
        for i in range(3):
            p = tmp_path / f"clip{i}.mp4"
            p.write_bytes(b"x")
            clips.append(p)

        adapter = _make_adapter()
        summary = _make_summary()

        with patch(
            "home_brain.memory.adapters.telegram_notification_adapter.telegram"
        ) as mock_tg:
            mock_bot = AsyncMock()
            mock_bot.send_message = AsyncMock()
            mock_bot.send_video = AsyncMock()
            mock_tg.Bot.return_value.__aenter__ = AsyncMock(return_value=mock_bot)
            mock_tg.Bot.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 1 * 1024 * 1024  # 1 MB
                adapter.send_daily_summary(summary, clips)

        assert mock_bot.send_video.call_count == 3
        mock_bot.send_message.assert_called_once()
