import time

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
