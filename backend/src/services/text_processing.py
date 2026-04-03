"""Utility helpers for normalizing agent generated text."""

from __future__ import annotations

from html import escape
from html.parser import HTMLParser
import re


_SAFE_TAGS = {
    "p",
    "br",
    "strong",
    "em",
    "code",
    "pre",
    "ul",
    "ol",
    "li",
    "blockquote",
    "a",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
}

_SAFE_ATTRS = {
    "a": {"href", "title", "target", "rel"},
    "code": {"class"},
}


def strip_tool_calls(text: str) -> str:
    """移除文本中的工具调用标记。"""

    if not text:
        return text

    pattern = re.compile(r"\[TOOL_CALL:[^\]]+\]")
    return pattern.sub("", text)


class _MarkdownHtmlSanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _SAFE_TAGS:
            self._parts.append(escape(self.get_starttag_text() or ""))
            return

        allowed = _SAFE_ATTRS.get(tag, set())
        clean_attrs: list[str] = []
        for key, value in attrs:
            if key not in allowed:
                continue
            if value is None:
                continue
            clean_attrs.append(f'{key}="{escape(value, quote=True)}"')

        attr_text = f" {' '.join(clean_attrs)}" if clean_attrs else ""
        self._parts.append(f"<{tag}{attr_text}>")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SAFE_TAGS:
            self._parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self._parts.append(escape(data))

    def handle_entityref(self, name: str) -> None:
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._parts.append(f"&#{name};")

    def sanitized(self) -> str:
        return "".join(self._parts)


def sanitize_markdown_html(text: str) -> str:
    """Sanitize inline HTML in markdown with a strict whitelist."""

    if not text:
        return text

    sanitizer = _MarkdownHtmlSanitizer()
    sanitizer.feed(text)
    sanitizer.close()
    return sanitizer.sanitized()

