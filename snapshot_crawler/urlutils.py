"""URL normalization, same-origin policy and HTML link extraction."""
from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit


def normalize_url(base: str, href: str) -> str | None:
    """Resolve ``href`` against ``base`` and drop the fragment.

    Query strings are preserved. Returns None for unsupported schemes.
    """
    joined = urljoin(base, href)
    without_frag, _frag = urldefrag(joined)
    parts = urlsplit(without_frag)
    if parts.scheme not in ("http", "https"):
        return None
    if not parts.hostname:
        return None
    return urlunsplit(parts)


def origin_of(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return parts.scheme, parts.hostname.lower(), port


def same_origin(url: str, root_url: str) -> bool:
    """Same protocol + host + port only; path/query do not matter."""
    try:
        return origin_of(url) == origin_of(root_url)
    except ValueError:
        return False


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in ("a", "area", "link"):
            return
        for name, value in attrs:
            if name in ("href",) and value:
                self.links.append(value)


def extract_links(html: str, base_url: str) -> list[str]:
    parser = _LinkParser()
    try:
        parser.feed(html)
    except Exception:
        pass
    resolved = []
    seen = set()
    for href in parser.links:
        url = normalize_url(base_url, href)
        if url and url not in seen:
            seen.add(url)
            resolved.append(url)
    return resolved
