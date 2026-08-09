# SRT Caption Support Design

## Problem

Directory-backed datasets accept any value for `caption_extension`, so an SRT file can already be discovered by setting `caption_extension = ".srt"`. The current reader treats every caption file as plain text, however. SRT cue numbers and time ranges therefore reach the text encoder together with the subtitle text.

## Goal

When a directory-backed dataset uses an `.srt` caption file, read the cues in file order, remove cue numbers and time ranges, normalize whitespace, and return the combined cue text as one caption.

Existing plain-text caption behavior must not change.

## Scope

The parser will support:

- UTF-8 files, including an optional byte-order mark
- LF and CRLF line endings
- standard numbered SRT cues
- cues without sequence numbers
- multi-line cue text
- optional settings after the end timestamp
- case-insensitive SRT parser dispatch after the caption path is resolved

The reader will trim leading and trailing whitespace from each cue text line and join non-empty lines and consecutive cues with one ASCII space. Whitespace inside a text line will remain unchanged. Empty cues will not contribute text. Subtitle text, punctuation, repeated cues, and inline formatting tags will otherwise remain unchanged.

Existing caption path discovery will not change. On case-sensitive file systems, the configured `caption_extension` must therefore use the same letter case as the caption filename.

The following are out of scope:

- selecting cues for a sampled video time range
- deduplicating rolling or repeated subtitles
- removing HTML, ASS, or other inline formatting
- accepting non-SRT subtitle formats
- changing JSONL caption handling

## Design

Add a small caption-reading module under `musubi_tuner.dataset` with two responsibilities:

1. Read plain captions as UTF-8 text and SRT captions as UTF-8 with an optional byte-order mark (`utf-8-sig`).
2. Dispatch `.srt` files to an SRT parser while returning other files with the current `strip()` behavior.

The SRT parser will split the document into cue blocks on blank lines. For each non-empty block it will accept either a timing line first or a decimal sequence number followed by a timing line. A timing line must contain two valid SRT timestamps separated by `-->`; end-timestamp settings are allowed. The parser will discard the sequence number and timing line, trim the remaining text lines as described above, and append non-empty text to the result. A non-numeric line before a timing line is malformed rather than an identifier.

Both image and video directory data sources will use the shared reader. This keeps `caption_extension` behavior consistent and removes the existing duplicate file-reading logic. JSONL data sources will remain unchanged because their captions are already stored as strings rather than caption files.

## Error Handling

A non-empty cue block without a recognizable timing line will raise `ValueError`. The error will identify the caption path and cue number so malformed files fail during dataset loading instead of sending timestamps or unrelated content to the text encoder.

An empty SRT file, or one containing only empty cues, will return an empty caption, matching the existing behavior for an empty text caption file.

File decoding and I/O errors will continue to propagate with their original exception types.

## Tests

Tests will exercise the real caption reader and directory data source behavior:

- plain-text captions retain their current content and trimming behavior
- numbered SRT cues are combined in file order
- BOM combined with a first cue that has no sequence number is handled
- CRLF, multi-line cue text, and end-timestamp settings are handled
- an already resolved `.SRT` path uses the SRT parser
- malformed non-empty cue blocks raise a path-aware `ValueError`
- image directory data sources return parsed SRT text from `get_caption`
- video directory data sources return parsed SRT text from `get_caption`

Tests will be written and observed failing before production code is added.

## Documentation

Update the video directory dataset documentation in English and Japanese. Document that `caption_extension = ".srt"` removes sequence numbers and time ranges and combines all cue text into one video caption. State that captions are not selected according to sampled frame ranges.
