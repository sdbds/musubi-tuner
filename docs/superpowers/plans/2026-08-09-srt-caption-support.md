# SRT Caption Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse `.srt` caption files into one ordered plain-text caption while preserving existing plain-text and JSONL dataset behavior.

**Architecture:** Add a focused caption-file reader under `musubi_tuner.dataset`. It will select SRT parsing from the already-resolved path, and both directory-backed data sources will call it instead of duplicating file I/O. Caption discovery and JSONL handling remain unchanged.

**Tech Stack:** Python 3.10+, standard library `os` and `re`, pytest, Ruff

## Global Constraints

- Add no runtime dependency.
- Read SRT files with `utf-8-sig`; retain `utf-8` and `strip()` behavior for all other caption files.
- Join trimmed, non-empty cue text lines and cues with one ASCII space; preserve whitespace inside each text line.
- Accept a timing line first or after an ASCII decimal sequence number.
- Do not select cues by frame range, deduplicate text, remove inline tags, change caption discovery, or change JSONL captions.
- A malformed non-empty cue block must raise a path-aware `ValueError` without wrapping decoding or I/O exceptions.

---

### Task 1: Caption File Reader

**Files:**
- Create: `src/musubi_tuner/dataset/caption_utils.py`
- Create: `tests/test_caption_utils.py`

**Interfaces:**
- Produces: `parse_srt_caption(content: str, source: str) -> str`
- Produces: `read_caption_file(caption_path: str) -> str`

- [ ] **Step 1: Write failing reader and parser tests**

Create `tests/test_caption_utils.py` with real temporary files and literal expected captions:

```python
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
        "1\n00:00:00,000 --> 00:00:01,500\nFirst cue.\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\nSecond cue.\n",
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
    caption_path.write_bytes(
        b"1\r\n00:00:00.000 --> 00:00:02.000 position:50% align:middle\r\n"
        b"  first line  \r\nsecond  line\r\n"
    )

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
        "title mistakenly placed before timing\n"
        "00:00:00,000 --> 00:00:01,000\nCaption.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        read_caption_file(str(caption_path))

    assert "cue block 1" in str(error.value)
    assert str(caption_path) in str(error.value)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_caption_utils.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'musubi_tuner.dataset.caption_utils'` because the production module does not exist.

- [ ] **Step 3: Add the minimal caption reader**

Create `src/musubi_tuner/dataset/caption_utils.py`:

```python
import os
import re


_SRT_SEQUENCE_NUMBER = re.compile(r"[0-9]+")
_SRT_TIMESTAMP = r"[0-9]{2,}:[0-5][0-9]:[0-5][0-9][,.][0-9]{3}"
_SRT_TIMING_LINE = re.compile(rf"{_SRT_TIMESTAMP}[ \t]*-->[ \t]*{_SRT_TIMESTAMP}(?:[ \t]+.*)?")
_SRT_CUE_SEPARATOR = re.compile(r"(?:[ \t]*\r?\n){2,}")


def parse_srt_caption(content: str, source: str) -> str:
    content = content.strip()
    if not content:
        return ""

    caption_lines: list[str] = []
    for block_index, block in enumerate(_SRT_CUE_SEPARATOR.split(content), start=1):
        lines = block.splitlines()
        timing_index = 1 if _SRT_SEQUENCE_NUMBER.fullmatch(lines[0].strip()) else 0
        if timing_index >= len(lines) or not _SRT_TIMING_LINE.fullmatch(lines[timing_index].strip()):
            raise ValueError(
                f"Invalid SRT cue block {block_index} in {source!r}: "
                "expected a timing line after an optional decimal sequence number"
            )

        caption_lines.extend(line.strip() for line in lines[timing_index + 1 :] if line.strip())

    return " ".join(caption_lines)


def read_caption_file(caption_path: str) -> str:
    is_srt = os.path.splitext(caption_path)[1].lower() == ".srt"
    encoding = "utf-8-sig" if is_srt else "utf-8"
    with open(caption_path, "r", encoding=encoding) as caption_file:
        content = caption_file.read()

    return parse_srt_caption(content, caption_path) if is_srt else content.strip()
```

- [ ] **Step 4: Run the reader tests and verify GREEN**

Run: `pytest -q tests/test_caption_utils.py`

Expected: `7 passed`.

- [ ] **Step 5: Run focused lint and format checks**

Run: `& ..\.venv\Scripts\ruff.exe check --no-cache src/musubi_tuner/dataset/caption_utils.py tests/test_caption_utils.py`

Expected: exit 0 with no lint errors.

Run: `& ..\.venv\Scripts\ruff.exe format --no-cache --check src/musubi_tuner/dataset/caption_utils.py tests/test_caption_utils.py`

Expected: exit 0 with both files already formatted.

- [ ] **Step 6: Commit the reader**

```powershell
git add -- src/musubi_tuner/dataset/caption_utils.py tests/test_caption_utils.py
git commit -m "feat: parse SRT caption files"
```

### Task 2: Directory Data Source Integration

**Files:**
- Modify: `tests/test_caption_utils.py`
- Modify: `src/musubi_tuner/dataset/datasources.py:9-10,243-249,635-641`

**Interfaces:**
- Consumes: `read_caption_file(caption_path: str) -> str` from Task 1
- Produces: parsed SRT captions through `ImageDirectoryDatasource.get_caption` and `VideoDirectoryDatasource.get_caption`

- [ ] **Step 1: Add failing directory data source tests**

Append to `tests/test_caption_utils.py` and add the imports shown below:

```python
from musubi_tuner.dataset.datasources import ImageDirectoryDatasource, VideoDirectoryDatasource


def test_image_directory_datasource_reads_srt_caption(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    image_path.touch()
    (tmp_path / "frame.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nImage caption.\n",
        encoding="utf-8",
    )

    datasource = ImageDirectoryDatasource(str(tmp_path), caption_extension=".srt")

    assert datasource.get_caption(0) == (str(image_path), "Image caption.")


def test_video_directory_datasource_reads_srt_caption(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    video_path.touch()
    (tmp_path / "clip.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nVideo caption.\n",
        encoding="utf-8",
    )

    datasource = VideoDirectoryDatasource(str(tmp_path), caption_extension=".srt")

    assert datasource.get_caption(0) == (str(video_path), "Video caption.")
```

- [ ] **Step 2: Run integration tests and verify RED**

Run: `pytest -q tests/test_caption_utils.py::test_image_directory_datasource_reads_srt_caption tests/test_caption_utils.py::test_video_directory_datasource_reads_srt_caption`

Expected: both tests fail because the data sources return raw SRT content containing sequence numbers and timing lines.

- [ ] **Step 3: Route directory caption files through the shared reader**

Add the import near the other dataset imports in `src/musubi_tuner/dataset/datasources.py`:

```python
from musubi_tuner.dataset.caption_utils import read_caption_file
```

Replace the duplicated `open` blocks in both directory data source `get_caption` methods with:

```python
caption = read_caption_file(caption_path)
```

Do not change caption path construction, globbing, or JSONL data sources.

- [ ] **Step 4: Run integration and regression tests and verify GREEN**

Run: `pytest -q tests/test_caption_utils.py tests/test_audio_dataset_seam.py`

Expected: `30 passed`.

- [ ] **Step 5: Run focused lint and format checks**

Run: `& ..\.venv\Scripts\ruff.exe check --no-cache src/musubi_tuner/dataset/caption_utils.py src/musubi_tuner/dataset/datasources.py tests/test_caption_utils.py`

Expected: exit 0 with no lint errors.

Run: `& ..\.venv\Scripts\ruff.exe format --no-cache --check src/musubi_tuner/dataset/caption_utils.py src/musubi_tuner/dataset/datasources.py tests/test_caption_utils.py`

Expected: exit 0 with all files already formatted.

- [ ] **Step 6: Commit data source integration**

```powershell
git add -- src/musubi_tuner/dataset/datasources.py tests/test_caption_utils.py
git commit -m "feat: load SRT captions in directory datasets"
```

### Task 3: User Documentation and Final Verification

**Files:**
- Modify: `docs/dataset_config.md:171-188`

**Interfaces:**
- Documents: `caption_extension = ".srt"` for video directory datasets

- [ ] **Step 1: Document SRT behavior in English**

After the English paragraph describing `video1.mp4` and `video1.txt`, add:

```markdown
You can also set `caption_extension = ".srt"`. For SRT captions, cue numbers and time ranges are removed, and all cue text is joined in order into one caption for the entire video. SRT timing is not matched to frames selected by `frame_extraction`.
```

- [ ] **Step 2: Document SRT behavior in Japanese**

After the corresponding Japanese caption paragraph, add:

```markdown
`caption_extension = ".srt"`も使用できます。SRTキャプションでは、字幕番号と時間範囲が取り除かれ、すべての字幕本文が順番に結合されて動画全体の1つのキャプションになります。SRTの時間情報は、`frame_extraction`で選択されたフレームとの対応付けには使用されません。
```

- [ ] **Step 3: Run the focused test suite**

Run: `pytest -q tests/test_caption_utils.py tests/test_audio_dataset_seam.py`

Expected: `30 passed`.

- [ ] **Step 4: Run the full test suite**

Run: `pytest -q`

Expected: all collected tests pass with zero failures.

- [ ] **Step 5: Run final static checks**

Run: `& ..\.venv\Scripts\ruff.exe check --no-cache src/musubi_tuner/dataset/caption_utils.py src/musubi_tuner/dataset/datasources.py tests/test_caption_utils.py`

Expected: exit 0.

Run: `& ..\.venv\Scripts\ruff.exe format --no-cache --check src/musubi_tuner/dataset/caption_utils.py src/musubi_tuner/dataset/datasources.py tests/test_caption_utils.py`

Expected: exit 0.

Run: `git diff --check`

Expected: exit 0 with no whitespace errors.

- [ ] **Step 6: Commit documentation**

```powershell
git add -- docs/dataset_config.md
git commit -m "docs: document SRT dataset captions"
```
