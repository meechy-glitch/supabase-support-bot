import time
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

from app.config import settings

USER_AGENT = "supabase-support-bot/0.1 (+sales demo; contact: bobblash.eb@gmail.com)"
REQUEST_TIMEOUT = 20
POLITE_DELAY_SECONDS = 0.3


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_doc_urls(session: requests.Session | None = None) -> list[str]:
    """Pull all /docs URLs from the Supabase sitemap, capped at MAX_PAGES."""
    s = session or _session()
    resp = s.get(settings.docs_sitemap, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml-xml")
    urls = [loc.text.strip() for loc in soup.find_all("loc")]
    docs = [u for u in urls if "/docs" in u]
    return docs[: settings.max_pages]


def _normalize_doc_url(href: str, base: str) -> str | None:
    """Resolve href against base, strip fragment + query, and keep only Supabase
    /docs URLs. Returns the canonical absolute URL or None if it isn't a docs page.
    """
    abs_url = urljoin(base, href)
    parts = urlsplit(abs_url)
    if parts.scheme not in ("http", "https") or parts.netloc != "supabase.com":
        return None
    path = parts.path
    if not (path == "/docs" or path.startswith("/docs/")):
        return None
    # Drop fragments/queries and any trailing slash so equivalent links dedupe.
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/") or "/", "", ""))


def _fetch_html(url: str, session: requests.Session) -> str | None:
    """Fetch raw HTML for a URL, returning None on errors or non-200 (e.g. 404)."""
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return resp.text


def _extract_doc_links(html: str, base: str) -> list[str]:
    """Extract every <a href> on the page that resolves to a Supabase /docs URL."""
    soup = BeautifulSoup(html, "lxml")
    found: set[str] = set()
    for a in soup.find_all("a", href=True):
        norm = _normalize_doc_url(a["href"], base)
        if norm is not None:
            found.add(norm)
    return sorted(found)


def crawl_doc_urls(
    session: requests.Session | None = None,
    depth: int | None = None,
    max_pages: int | None = None,
    seeds: list[str] | None = None,
) -> list[str]:
    """Bounded breadth-first crawl of the Supabase docs.

    Starts from the sitemap pages (depth 0), follows in-page /docs links to
    discover deeper pages the shallow sitemap omits, and stops at `depth` levels
    or once `max_pages` URLs have been discovered, whichever comes first. Returns
    a deduplicated, discovery-ordered list of /docs URLs (seeds first).
    """
    s = session or _session()
    depth = settings.depth if depth is None else depth
    max_pages = settings.max_pages if max_pages is None else max_pages
    seed_urls = seeds if seeds is not None else fetch_doc_urls(s)

    seen: set[str] = set()
    discovered: list[str] = []
    frontier: list[str] = []
    for u in seed_urls:
        norm = _normalize_doc_url(u, u) or u
        if norm not in seen:
            seen.add(norm)
            discovered.append(norm)
            frontier.append(norm)

    for _ in range(depth):
        if len(discovered) >= max_pages:
            break
        next_frontier: list[str] = []
        for url in frontier:
            if len(discovered) >= max_pages:
                break
            html = _fetch_html(url, s)
            time.sleep(POLITE_DELAY_SECONDS)
            if html is None:
                continue
            for link in _extract_doc_links(html, url):
                if link not in seen:
                    seen.add(link)
                    discovered.append(link)
                    next_frontier.append(link)
                    if len(discovered) >= max_pages:
                        break
        frontier = next_frontier

    return discovered[:max_pages]


def fetch_page_text(url: str, session: requests.Session | None = None) -> tuple[str, str] | None:
    """Fetch a docs page and extract (title, article_text). Returns None if the page is empty/unusable."""
    s = session or _session()
    resp = s.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    article = soup.select_one("article")
    if article is None:
        return None
    text = article.get_text("\n", strip=True)
    if len(text) < 100:
        return None
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    return title, text


def iter_pages():
    """Yield (url, title, text) for each doc page, with a polite delay between requests."""
    s = _session()
    urls = fetch_doc_urls(s)
    for url in urls:
        page = fetch_page_text(url, s)
        time.sleep(POLITE_DELAY_SECONDS)
        if page is None:
            continue
        title, text = page
        yield url, title, text
