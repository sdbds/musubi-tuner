from html.parser import HTMLParser
import os
import re


_SRT_SEQUENCE_NUMBER = re.compile(r"[0-9]+")
_SRT_TIMESTAMP = r"[0-9]{2,}:[0-5][0-9]:[0-5][0-9][,.][0-9]{3}"
_SRT_TIMING_LINE = re.compile(rf"{_SRT_TIMESTAMP}[ \t]*-->[ \t]*{_SRT_TIMESTAMP}(?:[ \t]+.*)?")
_SRT_CUE_SEPARATOR = re.compile(r"(?:[ \t]*\r?\n){2,}")


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
                f"Invalid SRT cue block {block_index} in {source}: expected a timing line after an optional decimal sequence number"
            )

        for line in lines[timing_index + 1 :]:
            cleaned_line = _strip_srt_html(line).strip()
            if cleaned_line:
                caption_lines.append(cleaned_line)

    return " ".join(caption_lines)


def read_caption_file(caption_path: str) -> str:
    is_srt = os.path.splitext(caption_path)[1].lower() == ".srt"
    encoding = "utf-8-sig" if is_srt else "utf-8"
    with open(caption_path, "r", encoding=encoding) as caption_file:
        content = caption_file.read()

    return parse_srt_caption(content, caption_path) if is_srt else content.strip()
