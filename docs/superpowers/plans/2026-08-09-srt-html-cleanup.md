# SRT HTML Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove HTML markup from SRT captions while preserving rendered subtitle text.

**Architecture:** Keep cleanup inside `caption_utils.py`, immediately after cue metadata is removed and before cleaned lines are joined. A small standard-library `HTMLParser` subclass will collect character data, discard tags, insert a separator for `<br>`, and decode character references without changing the plain-text caption path.

**Tech Stack:** Python standard library `html.parser`, pytest, Ruff

## Global Constraints

- Add no runtime dependency; use Python's built-in `html.parser`.
- Apply HTML cleanup only to `.srt` caption files.
- Remove all HTML tags while preserving their text content.
- Treat `<br>` case-insensitively as a text separator.
- Decode HTML character references through `HTMLParser(convert_charrefs=True)`.
- Preserve the existing `.txt` read and trim behavior.

---

### Task 1: Clean HTML markup from SRT cue text

**Files:**
- Modify: `tests/test_caption_utils.py`
- Modify: `src/musubi_tuner/dataset/caption_utils.py`

**Interfaces:**
- Consumes: `parse_srt_caption(content: str, source: str) -> str`
- Produces: `_strip_srt_html(text: str) -> str`, used only by the SRT parser

- [ ] **Step 1: Write failing SRT cleanup and plain-text isolation tests**

```python
def test_srt_removes_html_markup_but_keeps_rendered_text(tmp_path: Path):
    caption_path = tmp_path / "clip.srt"
    caption_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "<font color='green' title='2 > 1'><i>Green &amp; bright</i></font><BR>Next\n"
        "<font color='green'></font>\n",
        encoding="utf-8",
    )

    assert read_caption_file(str(caption_path)) == "Green & bright Next"


def test_plain_caption_keeps_html_like_text(tmp_path: Path):
    caption_path = tmp_path / "clip.txt"
    caption_path.write_text("<font color='green'>Keep me</font>", encoding="utf-8")

    assert read_caption_file(str(caption_path)) == "<font color='green'>Keep me</font>"
```

- [ ] **Step 2: Run the focused test and verify the SRT assertion fails**

Run: `..\.venv\Scripts\python.exe -m pytest -q tests/test_caption_utils.py::test_srt_removes_html_markup_but_keeps_rendered_text tests/test_caption_utils.py::test_plain_caption_keeps_html_like_text`

Expected: one failure because the current SRT parser returns literal tags and `&amp;`; the plain-text isolation test passes.

- [ ] **Step 3: Add the standard-library HTML text extractor**

```python
from html.parser import HTMLParser


class _SRTTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "br":
            self.parts.append(" ")


def _strip_srt_html(text: str) -> str:
    parser = _SRTTextExtractor()
    parser.feed(text)
    parser.close()
    return "".join(parser.parts)
```

In `parse_srt_caption`, pass every cue text line through `_strip_srt_html`, trim the result, and append it only when non-empty.

- [ ] **Step 4: Run the focused tests and the complete caption test module**

Run: `..\.venv\Scripts\python.exe -m pytest -q tests/test_caption_utils.py`

Expected: all tests pass, including HTML cleanup and unchanged `.txt` behavior.

- [ ] **Step 5: Run integration regression tests and static checks**

Run: `..\.venv\Scripts\python.exe -m pytest -q tests/test_caption_utils.py tests/test_audio_dataset_seam.py tests/test_uniform_shift_timesteps.py`

Run: `..\.venv\Scripts\ruff.exe check --no-cache src/musubi_tuner/dataset/caption_utils.py tests/test_caption_utils.py`

Run: `..\.venv\Scripts\ruff.exe format --no-cache --check src/musubi_tuner/dataset/caption_utils.py tests/test_caption_utils.py`

Expected: all tests and both Ruff commands pass.

- [ ] **Step 6: Commit the implementation**

```powershell
git add src/musubi_tuner/dataset/caption_utils.py tests/test_caption_utils.py docs/superpowers/plans/2026-08-09-srt-html-cleanup.md
git commit -m "feat: strip HTML markup from SRT captions"
```
