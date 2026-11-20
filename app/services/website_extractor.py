"""SSRF-resistant website retrieval and relevant business-text extraction."""

from __future__ import annotations

import ipaddress
import json
import re
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

    def __init__(
        self, message: str, content: WebsiteContent | None = None
    ) -> None:
        super().__init__(message)
        self.content = content


@dataclass(frozen=True)
class WebsiteContent:
    """Normalized content extracted from a website or public social profile."""

    url: str
    title: str
    meta_description: str
    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    relevant_links: list[dict[str, str]] = field(default_factory=list)
    service_descriptions: list[str] = field(default_factory=list)
    platform: str = "website"
    content_type: str = "page"
    extraction_status: str = "success"
    source: str = "public_html"
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_prompt_text(self, max_chars: int) -> str:
        """Render only bounded, relevant source content for the analysis prompt."""
        sections = [
            f"Platform: {self.platform}",
            f"Content type: {self.content_type}",
            f"Extraction status: {self.extraction_status}",
            f"Source: {self.source}",
            "Limitations:\n" + "\n".join(f"- {item}" for item in self.limitations),
            f"Metadata: {json.dumps(self.metadata, ensure_ascii=False)}",
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
        apify_api_token: str = "",
        apify_instagram_actor_id: str = "",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.max_content_chars = max_content_chars
        self.max_redirects = max_redirects
        self._transport = transport
        self.apify_api_token = apify_api_token
        self.apify_instagram_actor_id = apify_instagram_actor_id

    async def extract(self, url: str) -> WebsiteContent:
        """Fetch, validate, and normalize content from a website or social platform."""
        validate_url_syntax(url)
        platform = _detect_platform(url)
        if platform != "website":
            return await self._extract_social(url, platform)
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

    async def _request_public(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
    ) -> httpx.Response:
        """Fetch a bounded response while validating every redirect destination."""
        current_url = url
        for redirect_count in range(self.max_redirects + 1):
            try:
                await validate_resolved_url(current_url)
            except UnsafeWebsiteURLError as exc:
                if redirect_count:
                    raise WebsiteHTTPError(
                        "Website redirected to an unsafe network destination."
                    ) from exc
                raise

            async with client.stream("GET", current_url, headers=headers) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    if redirect_count >= self.max_redirects:
                        raise WebsiteHTTPError(
                            "Website exceeded the maximum number of redirects."
                        )
                    location = response.headers.get("location")
                    if not location:
                        raise WebsiteHTTPError(
                            "Website returned a redirect without a destination."
                        )
                    current_url = urljoin(current_url, location)
                    try:
                        validate_url_syntax(current_url)
                    except ValueError as exc:
                        raise WebsiteHTTPError(
                            "Website redirected to an invalid or unsafe URL."
                        ) from exc
                    continue

                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.max_response_bytes:
                        raise WebsiteHTTPError(
                            "Website response exceeds the configured size limit."
                        )
                response_headers = response.headers.copy()
                for header in ("content-encoding", "content-length", "transfer-encoding"):
                    response_headers.pop(header, None)
                return httpx.Response(
                    response.status_code,
                    headers=response_headers,
                    content=bytes(body),
                    request=response.request,
                )
        raise WebsiteHTTPError("Website redirect handling failed.")

    async def _extract_social(self, url: str, platform: str) -> WebsiteContent:
        """Extract public social metadata; return explicit limitations when blocked."""
        timeout = httpx.Timeout(self.timeout_seconds)
        transport = self._transport or SSRFProtectedHTTPTransport()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

        if platform == "youtube":
            oembed_url = str(
                httpx.URL("https://www.youtube.com/oembed")
                .copy_add_param("url", url)
                .copy_add_param("format", "json")
            )
            data: dict[str, Any] = {}
            title = ""
            description = ""
            source = "youtube_oembed"
            async with httpx.AsyncClient(transport=transport, timeout=timeout, trust_env=False) as client:
                try:
                    response = await self._request_public(client, oembed_url, headers)
                    if response.status_code < 400:
                        parsed = response.json()
                        if isinstance(parsed, dict):
                            data = parsed
                except (httpx.TimeoutException, WebsiteHTTPError):
                    data = {}
                except (httpx.HTTPError, json.JSONDecodeError, ValueError):
                    data = {}

                if not data:
                    try:
                        page = await self._request_public(client, url, headers)
                        if page.status_code >= 400:
                            detail = (
                                f"YouTube returned HTTP {page.status_code}; public metadata is inaccessible."
                            )
                            if page.status_code >= 500:
                                raise WebsiteHTTPError(detail)
                            raise EmptyWebsiteContentError(
                                detail,
                                content=WebsiteContent(
                                    url=url,
                                    title="",
                                    meta_description="",
                                    platform="youtube",
                                    content_type=_social_content_type(url, platform),
                                    extraction_status="limited",
                                    source="public_html",
                                    limitations=[detail],
                                ),
                            )
                        soup = BeautifulSoup(page.content, "html.parser")
                        public_metadata = _social_metadata(soup)
                        title = public_metadata.get("title", "")
                        description = public_metadata.get("description", "")
                        source = "public_html"
                    except httpx.TimeoutException as exc:
                        raise WebsiteTimeoutError("YouTube metadata request timed out.") from exc
                    except httpx.HTTPError as exc:
                        raise WebsiteHTTPError("YouTube public metadata could not be reached.") from exc
                    except WebsiteHTTPError:
                        raise
                    except UnsafeWebsiteURLError as exc:
                        raise WebsiteHTTPError(
                            "YouTube redirected to an unsafe network destination."
                        ) from exc
            title = _clean_text(str(data.get("title", ""))) or _clean_text(title)
            author = _clean_text(str(data.get("author_name", "")))
            if not title:
                detail = "YouTube returned no usable public metadata."
                raise EmptyWebsiteContentError(
                    detail,
                    content=WebsiteContent(
                        url=url,
                        title="",
                        meta_description="",
                        platform="youtube",
                        content_type=_social_content_type(url, platform),
                        extraction_status="limited",
                        source=source,
                        limitations=[detail],
                    ),
                )
            content_type = _social_content_type(url, platform)
            limitations: list[str] = []
            if source == "youtube_oembed" and content_type == "video":
                limitations.extend(
                    [
                        "Only public oEmbed metadata was retrieved; the video content and transcript were not retrieved.",
                        "Engagement statistics were not retrieved.",
                    ]
                )
            elif content_type in {"profile", "channel_or_video"}:
                limitations.append("Channel statistics, recent posts, and transcripts may require another authorized source.")
            elif content_type == "video":
                limitations.append("Transcript and engagement data were not retrieved from public HTML.")
            else:
                limitations.append("This is a YouTube landing page; channel-specific details were not retrieved.")
            metadata = {
                key: value
                for key, value in {
                    "author_name": author,
                    "thumbnail_url": str(data.get("thumbnail_url", "")),
                }.items()
                if value
            }
            return WebsiteContent(
                url=url,
                title=title,
                meta_description=_clean_text(description)[:1000],
                headings=[title],
                platform="youtube",
                content_type=content_type,
                extraction_status="success" if source == "public_html" and description else "limited",
                source=source,
                limitations=limitations,
                metadata=metadata,
            )

        async with httpx.AsyncClient(transport=transport, timeout=timeout, trust_env=False) as client:
            try:
                response = await self._request_public(client, url, headers)
            except httpx.TimeoutException as exc:
                raise WebsiteTimeoutError("Social platform request timed out.") from exc
            except httpx.HTTPError as exc:
                raise WebsiteHTTPError("Social platform could not be reached.") from exc
            except UnsafeWebsiteURLError:
                raise

        if response.status_code >= 500:
            raise WebsiteHTTPError(
                f"The {platform} public page returned HTTP {response.status_code}."
            )
        if response.status_code >= 400:
            return _limited_social_content(url, platform, response.status_code)

        content_type = response.headers.get("content-type", "").lower()
        if content_type and "text/html" not in content_type:
            raise WebsiteHTTPError("Social URL did not return an HTML page.")
        soup = BeautifulSoup(response.content, "html.parser")
        metadata = _social_metadata(soup)
        title = metadata.pop("title", "") or _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
        description = metadata.pop("description", "")

        if not description:
            meta_description = soup.find("meta", attrs={"name": "description"})
            description = _clean_text(meta_description.get("content", "")) if meta_description else ""

        html_text = response.text
        login_wall = bool(re.search(r'log in|login|sign up|password', html_text, re.IGNORECASE))

        source = "public_html"
        fallback_limitations: list[str] = []
        if platform == "instagram" and (login_wall or not title or not description) and self.apify_api_token and self.apify_instagram_actor_id:
            apify_data = await self._extract_instagram_apify(url)
            if apify_data:
                title = _clean_text(str(apify_data.get("title") or apify_data.get("name") or apify_data.get("username") or ""))
                description = _clean_text(str(apify_data.get("description") or apify_data.get("biography") or apify_data.get("bio") or ""))
                metadata.update({key: _clean_text(str(value)) for key, value in apify_data.items() if isinstance(value, (str, int, float)) and key not in {"description", "biography", "bio", "title", "name"}})
                login_wall = False
                metadata["scraper"] = "apify"
                source = "apify"
            else:
                fallback_limitations.append(
                    "The configured Instagram fallback did not return usable public profile data."
                )
        username = _platform_username(url, platform)
        limitations: list[str] = fallback_limitations

        if platform == "instagram":
            limitations.append("Public HTML may omit profile bio, follower counts, and posts; login or bot protection can require another authorized source.")
        elif platform == "facebook":
            limitations.append("Facebook may require login or may omit public profile and post details.")
        elif platform == "x":
            if _social_content_type(url, platform) == "public_post":
                limitations.append("Only this post's extracted public text and metadata are available; profile bio and account statistics were not retrieved.")
            else:
                limitations.append("X may block automated requests or omit post and profile details.")

        if login_wall:
            limitations.insert(0, f"The {platform} public page appears to contain a login or access restriction.")
            extraction_status = "limited"
        elif description or (
            title
            and title.casefold()
            not in {platform.casefold(), f"{platform.title()} - log in or sign up".casefold()}
        ):
            extraction_status = "success"
        else:
            extraction_status = "limited"

        fallback_title = title or (username and f"@{username}" if platform in {"instagram", "x"} else username)
        fallback_title = fallback_title or platform.title()
        if username:
            metadata["username"] = username
            metadata["username_source"] = "submitted_url_path"
        return WebsiteContent(
            url=url,
            title=_clean_text(fallback_title)[:500],
            meta_description=_clean_text(description)[:1000],
            headings=[_clean_text(fallback_title)] if fallback_title else [],
            paragraphs=[_clean_text(description)] if description else [],
            platform=platform,
            content_type=_social_content_type(url, platform),
            extraction_status=extraction_status,
            source=source,
            limitations=limitations,
            metadata=metadata,
        )

    async def _extract_instagram_apify(self, url: str) -> dict[str, Any] | None:
        """Call a configured Apify Instagram actor and use its first profile result."""
        actor = self.apify_instagram_actor_id.strip("/")
        endpoint = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
        payload = {"directUrls": [url], "resultsLimit": 1}
        headers = {"Authorization": f"Bearer {self.apify_api_token}"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(max(20, self.timeout_seconds), trust_env=False), trust_env=False) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            return None
        except httpx.HTTPError:
            return None
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            return None
        return data[0]

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
        evidence = " ".join(
            [
                result.title,
                result.meta_description,
                *result.headings,
                *result.paragraphs,
                *result.service_descriptions,
            ]
        ).strip()
        if not evidence:
            raise EmptyWebsiteContentError(
                "Website contained no readable business content.", content=result
            )
        if len(evidence) < 20:
            raise EmptyWebsiteContentError(
                "Website contained too little useful content to analyze.",
                content=result,
            )
        return result


def _social_content_type(url: str, platform: str) -> str:
    """Classify common profile, post, video, and homepage URL forms."""
    parsed = urlsplit(url)
    parts = [part.casefold() for part in parsed.path.split("/") if part]
    if platform == "youtube":
        if parsed.hostname == "youtu.be" or parsed.path.startswith(("/watch", "/shorts/")):
            return "video"
        if not parts:
            return "homepage"
        return "profile" if parts[0].startswith("@") or parts[0] in {"channel", "c", "user"} else "channel_or_video"
    if platform == "instagram":
        if not parts:
            return "homepage"
        if parts[0] in {"p", "reel", "reels", "tv", "stories"}:
            return "public_post"
        return "profile"
    if platform == "x":
        if not parts or parts[0] in {"home", "explore", "search", "i", "settings", "messages", "notifications"}:
            return "homepage"
        return "public_post" if "status" in parts else "profile"
    if platform == "facebook":
        if not parts:
            return "homepage"
        return "public_post" if parts[0] in {"posts", "videos", "watch", "photo"} else "profile"
    return "page"


def _limited_social_content(
    url: str, platform: str, status_code: int
) -> WebsiteContent:
    """Represent an upstream denial without labeling URL-derived values as content."""
    if platform == "facebook" and status_code == 400:
        limitation = (
            "Facebook returned HTTP 400; public HTML content was not accessible."
        )
    elif status_code in {403, 429}:
        limitation = (
            f"{platform} returned HTTP {status_code}; automated access was denied or rate limited."
        )
    elif status_code in {404, 410}:
        limitation = (
            f"{platform} returned HTTP {status_code}; the requested public content was not found or is unavailable."
        )
    else:
        limitation = (
            f"{platform} returned HTTP {status_code}; the requested public content was inaccessible."
        )

    username = _platform_username(url, platform)
    metadata = (
        {"requested_handle": username, "requested_handle_source": "submitted_url_path"}
        if username
        else {}
    )
    return WebsiteContent(
        url=url,
        title="",
        meta_description="",
        platform=platform,
        content_type=_social_content_type(url, platform),
        extraction_status="limited",
        source="public_html",
        limitations=[limitation],
        metadata=metadata,
    )


def _detect_platform(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    if host == "youtube.com" or host.endswith(".youtube.com") or host == "youtu.be":
        return "youtube"
    if host == "instagram.com" or host.endswith(".instagram.com"):
        return "instagram"
    if host == "facebook.com" or host.endswith(".facebook.com"):
        return "facebook"
    if host in {"x.com", "twitter.com"} or host.endswith(".x.com") or host.endswith(".twitter.com"):
        return "x"
    return "website"


def _social_metadata(soup: BeautifulSoup) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for meta in soup.find_all("meta"):
        key = meta.get("property") or meta.get("name")
        value = _clean_text(str(meta.get("content", "")))
        if not key or not value:
            continue
        normalized = key.lower()
        if normalized in {"og:title", "twitter:title"} and "title" not in metadata:
            metadata["title"] = value
        elif normalized in {"og:description", "twitter:description"} and "description" not in metadata:
            metadata["description"] = value
        elif normalized in {"og:image", "twitter:image"}:
            metadata.setdefault("image_url", value)
        elif normalized in {"og:url", "twitter:site"}:
            metadata.setdefault("canonical_url", value)
    return metadata


def _platform_username(url: str, platform: str) -> str:
    path_parts = [part for part in urlsplit(url).path.split("/") if part]
    if not path_parts:
        return ""
    if platform == "youtube":
        if urlsplit(url).hostname == "youtu.be" or path_parts[0] in {"watch", "shorts"}:
            return ""
        return path_parts[-1]
    if platform == "instagram":
        if path_parts[0] in {"p", "reel", "reels", "tv", "stories"}:
            return ""
        return path_parts[0]
    if platform == "x":
        return path_parts[0]
    if platform == "facebook":
        if path_parts[0] in {"posts", "videos", "watch", "photo"}:
            return ""
        return path_parts[0]
    return ""


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
