"""SSRF-resistant website retrieval and relevant business-text extraction."""

from __future__ import annotations

import ipaddress
import ssl
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx
from bs4 import BeautifulSoup
from httpcore._backends.anyio import AnyIOBackend

from app.utils.validators import (
    UnsafeWebsiteURLError,
    resolve_public_addresses,
    validate_resolved_url,
    validate_url_syntax,
)

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_NOISE_SELECTORS = (
    "nav",
    "header",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "iframe",
    "dialog",
    "[aria-hidden='true']",
    "[role='navigation']",
    "[role='menu']",
    "[role='banner']",
    "[role='contentinfo']",
    ".cookie",
    ".cookie-banner",
    ".navbar",
    ".navigation",
    ".menu",
    ".breadcrumb",
    ".breadcrumbs",
    ".social-share",
)


class WebsiteFetchError(Exception):
    """Base class for website extraction failures."""


class WebsiteTimeoutError(WebsiteFetchError):
    """Raised when the target website does not respond before the timeout."""


class WebsiteHTTPError(WebsiteFetchError):
    """Raised when a website returns an error or an unsupported document."""


class EmptyWebsiteContentError(WebsiteFetchError):
    """Raised when the page contains no useful business text."""


@dataclass(frozen=True)
class WebsiteContent:
    """Structured content extracted from the submitted landing page."""

    url: str
    title: str
    meta_description: str
    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    relevant_links: list[dict[str, str]] = field(default_factory=list)
    service_descriptions: list[str] = field(default_factory=list)

    def to_prompt_text(self, max_chars: int) -> str:
        """Render only bounded, relevant source content for the analysis prompt."""
        sections = [
            f"Page URL: {self.url}",
            f"Page title: {self.title}",
            f"Meta description: {self.meta_description}",
            "Headings:\n" + "\n".join(f"- {item}" for item in self.headings),
            "Service/product descriptions:\n"
            + "\n".join(f"- {item}" for item in self.service_descriptions),
            "Relevant internal links:\n"
            + "\n".join(f"- {item['text']}: {item['url']}" for item in self.relevant_links),
            "Page text:\n" + "\n".join(self.paragraphs),
        ]
        return "\n\n".join(sections)[:max_chars]


class _PublicDNSNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolve and pin each connection to a vetted public IP before connecting."""

    def __init__(self) -> None:
        self._backend = AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = await resolve_public_addresses(host, port)
        address = ipaddress.ip_address(addresses[0])
        pinned_host = address.compressed
        return await self._backend.connect_tcp(
            host=pinned_host,
            port=port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise UnsafeWebsiteURLError("Unix socket connections are not allowed.")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class SSRFProtectedHTTPTransport(httpx.AsyncHTTPTransport):
    """HTTPX transport whose TCP backend pins connections to public DNS results."""

    def __init__(self) -> None:
        super().__init__(verify=True, trust_env=False, retries=0)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl.create_default_context(),
            max_connections=12,
            max_keepalive_connections=4,
            keepalive_expiry=5,
            network_backend=_PublicDNSNetworkBackend(),
        )


class WebsiteExtractor:
    """Fetch and parse one public HTTP(S) website with strict resource limits."""

    def __init__(
        self,
        timeout_seconds: float = 12,
        max_response_bytes: int = 2_000_000,
        max_content_chars: int = 18_000,
        max_redirects: int = 5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.max_content_chars = max_content_chars
        self.max_redirects = max_redirects
        self._transport = transport

    async def extract(self, url: str) -> WebsiteContent:
        """Fetch, validate, and extract business-relevant content from a URL."""
        validate_url_syntax(url)
        timeout = httpx.Timeout(self.timeout_seconds)
        transport = self._transport or SSRFProtectedHTTPTransport()
        async with httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "AI-Link-Analysis/1.0 (+website text analysis)"},
        ) as client:
            response_url = url
            for redirect_count in range(self.max_redirects + 1):
                try:
                    await validate_resolved_url(response_url)
                except UnsafeWebsiteURLError as exc:
                    if redirect_count > 0:
                        raise WebsiteHTTPError(
                            "Website redirected to an unsafe network destination."
                        ) from exc
                    raise
                try:
                    async with client.stream("GET", response_url) as response:
                        if response.status_code in _REDIRECT_STATUSES:
                            if redirect_count >= self.max_redirects:
                                raise WebsiteHTTPError("Website exceeded the maximum number of redirects.")
                            location = response.headers.get("location")
                            if not location:
                                raise WebsiteHTTPError("Website returned a redirect without a destination.")
                            response_url = urljoin(response_url, location)
                            try:
                                validate_url_syntax(response_url)
                            except ValueError as exc:
                                raise WebsiteHTTPError(
                                    "Website redirected to an invalid or unsafe URL."
                                ) from exc
                            continue
                        if response.status_code >= 400:
                            raise WebsiteHTTPError(
                                f"Website returned HTTP {response.status_code}."
                            )
                        content_type = response.headers.get("content-type", "").lower()
                        if content_type and not (
                            "text/html" in content_type or "application/xhtml+xml" in content_type
                        ):
                            raise WebsiteHTTPError("The URL did not return an HTML page.")
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > self.max_response_bytes:
                                raise WebsiteHTTPError(
                                    "Website response exceeds the configured size limit."
                                )
                        return self.parse_html(bytes(body), str(response.url))
                except httpx.TimeoutException as exc:
                    raise WebsiteTimeoutError("Website request timed out.") from exc
                except httpx.HTTPError as exc:
                    raise WebsiteHTTPError("Website could not be reached.") from exc
            raise WebsiteHTTPError("Website redirect handling failed.")

    def parse_html(self, html: bytes, page_url: str) -> WebsiteContent:
        """Extract and de-duplicate the title, metadata, headings, and page text."""
        soup = BeautifulSoup(html, "html.parser")
        title = _clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
        description_tag = soup.find("meta", attrs={"name": "description"})
        meta_description = _clean_text(description_tag.get("content", "")) if description_tag else ""

        for node in soup.select(", ".join(_NOISE_SELECTORS)):
            node.decompose()

        headings = _unique_text(
            _clean_text(node.get_text(" ", strip=True))
            for node in soup.find_all(["h1", "h2", "h3"])
        )
        paragraphs = _unique_text(
            _clean_text(node.get_text(" ", strip=True))
            for node in soup.find_all(["p", "li"])
        )

        parsed_base = urlsplit(page_url)
        links: list[dict[str, str]] = []
        seen_links: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            text = _clean_text(anchor.get_text(" ", strip=True))
            target = urljoin(page_url, anchor["href"].strip())
            parsed_target = urlsplit(target)
            if (
                not text
                or parsed_target.scheme not in {"http", "https"}
                or parsed_target.hostname != parsed_base.hostname
                or target in seen_links
            ):
                continue
            seen_links.add(target)
            links.append({"text": text[:200], "url": target[:2048]})
            if len(links) >= 30:
                break

        service_descriptions: list[str] = []
        for heading in soup.find_all(["h2", "h3"]):
            text = _clean_text(heading.get_text(" ", strip=True))
            if not text:
                continue
            parent = heading.parent
            description = _clean_text(parent.get_text(" ", strip=True)) if parent else text
            if description and description not in service_descriptions:
                service_descriptions.append(description[:800])
            if len(service_descriptions) >= 12:
                break

        result = WebsiteContent(
            url=page_url,
            title=title[:500],
            meta_description=meta_description[:1000],
            headings=headings[:60],
            paragraphs=paragraphs[:200],
            relevant_links=links,
            service_descriptions=service_descriptions,
        )
        if not any((result.title, result.meta_description, *result.headings, *result.paragraphs)):
            raise EmptyWebsiteContentError("Website contained no readable business content.")
        if len(result.to_prompt_text(self.max_content_chars).strip()) < 30:
            raise EmptyWebsiteContentError("Website contained too little useful content to analyze.")
        return result


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _unique_text(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if len(value) < 2:
            continue
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(value[:2000])
    return unique
