"""
Tests for intelligence domain entities: Observation, Transcript, DailySummary, Mood.

Negative tests first (RED before implementation), then positive tests.
"""

import pytest
from datetime import date

from home_brain.intelligence.domain.entities.observation import (
    DailySummary,
    Mood,
    Observation,
    Transcript,
)


# ---------------------------------------------------------------------------
# NEGATIVE TESTS — all should raise before implementation exists
# ---------------------------------------------------------------------------


class TestObservationNegative:
    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            Observation.create(
                clip_id="clip-1",
                mood=Mood.NEUTRAL,
                activity="person walking",
                description="A person walks through the hallway.",
                confidence=-0.1,
            )

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            Observation.create(
                clip_id="clip-1",
                mood=Mood.NEUTRAL,
                activity="person walking",
                description="A person walks through the hallway.",
                confidence=1.1,
            )

    def test_empty_activity_raises(self):
        with pytest.raises(ValueError, match="activity"):
            Observation.create(
                clip_id="clip-1",
                mood=Mood.NEUTRAL,
                activity="",
                description="A person walks through the hallway.",
                confidence=0.9,
            )

    def test_whitespace_only_activity_raises(self):
        with pytest.raises(ValueError, match="activity"):
            Observation.create(
                clip_id="clip-1",
                mood=Mood.NEUTRAL,
                activity="   ",
                description="A person walks through the hallway.",
                confidence=0.9,
            )

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            Observation.create(
                clip_id="clip-1",
                mood=Mood.NEUTRAL,
                activity="person walking",
                description="",
                confidence=0.9,
            )

    def test_whitespace_only_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            Observation.create(
                clip_id="clip-1",
                mood=Mood.NEUTRAL,
                activity="person walking",
                description="   ",
                confidence=0.9,
            )

    def test_invalid_mood_string_raises(self):
        with pytest.raises(ValueError):
            Observation.create(
                clip_id="clip-1",
                mood="excited",  # not a valid Mood value
                activity="person walking",
                description="A person walks.",
                confidence=0.9,
            )


class TestTranscriptNegative:
    def test_empty_language_raises(self):
        with pytest.raises(ValueError, match="language"):
            Transcript.create(
                clip_id="clip-1",
                text="Hello there.",
                language="",
            )

    def test_whitespace_only_language_raises(self):
        with pytest.raises(ValueError, match="language"):
            Transcript.create(
                clip_id="clip-1",
                text="Hello there.",
                language="   ",
            )


class TestDailySummaryNegative:
    def test_empty_summary_text_raises(self):
        with pytest.raises(ValueError, match="summary_text"):
            DailySummary.create(
                date=date(2026, 3, 28),
                summary_text="",
                mood_counts={"neutral": 3},
                highlight_clip_ids=["clip-1"],
            )

    def test_whitespace_only_summary_text_raises(self):
        with pytest.raises(ValueError, match="summary_text"):
            DailySummary.create(
                date=date(2026, 3, 28),
                summary_text="   ",
                mood_counts={"neutral": 3},
                highlight_clip_ids=["clip-1"],
            )

    def test_more_than_three_highlights_raises(self):
        with pytest.raises(ValueError, match="highlight_clip_ids"):
            DailySummary.create(
                date=date(2026, 3, 28),
                summary_text="A quiet day at home.",
                mood_counts={"neutral": 4},
                highlight_clip_ids=["clip-1", "clip-2", "clip-3", "clip-4"],
            )


# ---------------------------------------------------------------------------
# POSITIVE TESTS
# ---------------------------------------------------------------------------


class TestObservationPositive:
    def test_create_sets_uuid_id(self):
        obs = Observation.create(
            clip_id="clip-1",
            mood=Mood.HAPPY,
            activity="child playing in living room",
            description="A child is playing with toys on the floor.",
            confidence=0.95,
        )
        assert obs.id
        assert len(obs.id) == 36  # UUID string length

    def test_create_sets_observed_at(self):
        obs = Observation.create(
            clip_id="clip-1",
            mood=Mood.CALM,
            activity="cat sleeping on sofa",
            description="The cat is napping peacefully.",
            confidence=0.8,
        )
        assert obs.observed_at is not None

    def test_create_stores_all_fields(self):
        obs = Observation.create(
            clip_id="clip-42",
            mood=Mood.NEUTRAL,
            activity="person walking",
            description="A person walks through the hallway.",
            confidence=0.75,
        )
        assert obs.clip_id == "clip-42"
        assert obs.mood == Mood.NEUTRAL
        assert obs.activity == "person walking"
        assert obs.description == "A person walks through the hallway."
        assert obs.confidence == 0.75

    def test_confidence_boundary_zero(self):
        obs = Observation.create(
            clip_id="clip-1",
            mood=Mood.SAD,
            activity="empty room",
            description="The room is empty.",
            confidence=0.0,
        )
        assert obs.confidence == 0.0

    def test_confidence_boundary_one(self):
        obs = Observation.create(
            clip_id="clip-1",
            mood=Mood.ANGRY,
            activity="loud argument",
            description="Two people are arguing loudly.",
            confidence=1.0,
        )
        assert obs.confidence == 1.0

    def test_mood_string_accepted_as_mood_enum(self):
        obs = Observation.create(
            clip_id="clip-1",
            mood=Mood.HAPPY,
            activity="celebration",
            description="People are celebrating.",
            confidence=0.9,
        )
        assert obs.mood == Mood.HAPPY

    def test_two_observations_have_different_ids(self):
        kwargs = dict(
            clip_id="clip-1",
            mood=Mood.NEUTRAL,
            activity="person standing",
            description="A person stands by the door.",
            confidence=0.6,
        )
        obs1 = Observation.create(**kwargs)
        obs2 = Observation.create(**kwargs)
        assert obs1.id != obs2.id


class TestMoodEnum:
    def test_all_five_moods_exist(self):
        assert Mood.HAPPY.value == "happy"
        assert Mood.SAD.value == "sad"
        assert Mood.ANGRY.value == "angry"
        assert Mood.NEUTRAL.value == "neutral"
        assert Mood.CALM.value == "calm"

    def test_mood_from_string(self):
        assert Mood("happy") == Mood.HAPPY
        assert Mood("calm") == Mood.CALM


class TestTranscriptPositive:
    def test_create_sets_uuid_id(self):
        tr = Transcript.create(clip_id="clip-1", text="Hello world.", language="en")
        assert tr.id
        assert len(tr.id) == 36

    def test_create_stores_all_fields(self):
        tr = Transcript.create(clip_id="clip-99", text="Namaste.", language="hi")
        assert tr.clip_id == "clip-99"
        assert tr.text == "Namaste."
        assert tr.language == "hi"

    def test_empty_text_is_allowed(self):
        # Silence is valid — no speech detected
        tr = Transcript.create(clip_id="clip-1", text="", language="en")
        assert tr.text == ""

    def test_create_sets_observed_at(self):
        tr = Transcript.create(clip_id="clip-1", text="Hi.", language="en")
        assert tr.observed_at is not None

    def test_two_transcripts_have_different_ids(self):
        tr1 = Transcript.create(clip_id="clip-1", text="Hi.", language="en")
        tr2 = Transcript.create(clip_id="clip-1", text="Hi.", language="en")
        assert tr1.id != tr2.id


class TestDailySummaryPositive:
    def test_create_defaults_notified_to_false(self):
        ds = DailySummary.create(
            date=date(2026, 3, 28),
            summary_text="A quiet day at home.",
            mood_counts={"neutral": 3, "happy": 1},
            highlight_clip_ids=["clip-1", "clip-2"],
        )
        assert ds.notified is False

    def test_create_sets_uuid_id(self):
        ds = DailySummary.create(
            date=date(2026, 3, 28),
            summary_text="A quiet day at home.",
            mood_counts={},
            highlight_clip_ids=[],
        )
        assert ds.id
        assert len(ds.id) == 36

    def test_exactly_three_highlights_allowed(self):
        ds = DailySummary.create(
            date=date(2026, 3, 28),
            summary_text="Busy day.",
            mood_counts={"happy": 3},
            highlight_clip_ids=["clip-1", "clip-2", "clip-3"],
        )
        assert len(ds.highlight_clip_ids) == 3

    def test_zero_highlights_allowed(self):
        ds = DailySummary.create(
            date=date(2026, 3, 28),
            summary_text="No clips today.",
            mood_counts={},
            highlight_clip_ids=[],
        )
        assert ds.highlight_clip_ids == []

    def test_mark_notified_returns_new_instance_with_notified_true(self):
        ds = DailySummary.create(
            date=date(2026, 3, 28),
            summary_text="A quiet day at home.",
            mood_counts={"neutral": 2},
            highlight_clip_ids=["clip-1"],
        )
        notified_ds = ds.mark_notified()
        assert notified_ds.notified is True
        assert ds.notified is False  # original unchanged

    def test_mark_notified_preserves_all_fields(self):
        ds = DailySummary.create(
            date=date(2026, 3, 28),
            summary_text="A quiet day.",
            mood_counts={"calm": 5},
            highlight_clip_ids=["clip-1"],
        )
        notified_ds = ds.mark_notified()
        assert notified_ds.id == ds.id
        assert notified_ds.date == ds.date
        assert notified_ds.summary_text == ds.summary_text
        assert notified_ds.mood_counts == ds.mood_counts
        assert notified_ds.highlight_clip_ids == ds.highlight_clip_ids
