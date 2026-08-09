from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.dataset.caption_utils import read_caption_file


def test_plain_caption_keeps_existing_trim_behavior(tmp_path: Path):
    caption_path = tmp_path / "clip.txt"
    caption_path.write_text("  first line\nsecond  ", encoding="utf-8")

    assert read_caption_file(str(caption_path)) == "first line\nsecond"


def test_numbered_srt_cues_are_joined_in_file_order(tmp_path: Path):
    caption_path = tmp_path / "clip.srt"
    caption_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,500\nFirst cue.\n\n2\n00:00:01,500 --> 00:00:03,000\nSecond cue.\n",
        encoding="utf-8",
    )

    assert read_caption_file(str(caption_path)) == "First cue. Second cue."


def test_srt_bom_is_removed_when_first_cue_has_no_sequence_number(tmp_path: Path):
    caption_path = tmp_path / "clip.srt"
    caption_path.write_text(
        "00:00:00,000 --> 00:00:01,000\nCaption without a number.\n",
        encoding="utf-8-sig",
    )

    assert read_caption_file(str(caption_path)) == "Caption without a number."


def test_srt_handles_crlf_multiline_text_and_timing_settings(tmp_path: Path):
    caption_path = tmp_path / "clip.srt"
    caption_path.write_bytes(b"1\r\n00:00:00.000 --> 00:00:02.000 position:50% align:middle\r\n  first line  \r\nsecond  line\r\n")

    assert read_caption_file(str(caption_path)) == "first line second  line"


def test_uppercase_srt_path_uses_srt_parser(tmp_path: Path):
    caption_path = tmp_path / "clip.SRT"
    caption_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nUppercase extension.\n", encoding="utf-8")

    assert read_caption_file(str(caption_path)) == "Uppercase extension."


def test_empty_srt_returns_empty_caption(tmp_path: Path):
    caption_path = tmp_path / "clip.srt"
    caption_path.write_text("", encoding="utf-8")

    assert read_caption_file(str(caption_path)) == ""


def test_malformed_srt_reports_cue_and_path(tmp_path: Path):
    caption_path = tmp_path / "clip.srt"
    caption_path.write_text(
        "title mistakenly placed before timing\n00:00:00,000 --> 00:00:01,000\nCaption.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        read_caption_file(str(caption_path))

    assert "cue block 1" in str(error.value)
    assert str(caption_path) in str(error.value)
