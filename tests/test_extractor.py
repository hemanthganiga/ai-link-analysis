"""Unit tests for URL validation, SSRF protections, and HTML extraction."""

from __future__ import annotations

import gzip
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.services import website_extractor as extractor_module
from app.services.website_extractor import (
    EmptyWebsiteContentError,
    _detect_platform,
    WebsiteExtractor,
    WebsiteHTTPError,
    WebsiteTimeoutError,
)
from app.utils.validators import (
    UnsafeWebsiteURLError,
    resolve_public_addresses,
    validate_resolved_url,
    validate_url_syntax,
)


@pytest.mark.parametrize(
    ("url", "platform"),
    [
        ("https://www.instagram.com/crestspot", "instagram"),
        ("https://youtube.com/@YouTube", "youtube"),
        ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
        ("https://facebook.com/Meta", "facebook"),
        ("https://x.com/openai", "x"),
        ("https://example.com", "website"),
    ],
)
def test_social_platform_detection(url: str, platform: str) -> None:
    assert _detect_platform(url) == platform


@pytest.mark.asyncio
async def test_instagram_login_wall_is_reported_as_limited() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html><head><title>Instagram</title></head><body><button>Log in</button></body></html>",
        )

    extractor = WebsiteExtractor(transport=httpx.MockTransport(handler))
    with patch("app.services.website_extractor.validate_resolved_url", new=AsyncMock(return_value=("instagram.com", 443, ["31.13.65.36"]))):
        result = await extractor.extract("https://www.instagram.com/crestspot")

    assert result.platform == "instagram"
    assert result.content_type == "profile"
    assert result.extraction_status == "limited"
    assert result.metadata["username"] == "crestspot"
    assert any("login" in limitation.lower() for limitation in result.limitations)


@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_youtube_channel_falls_back_to_public_html_metadata() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if "oembed" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html><head><meta property='og:title' content='YouTube'><meta property='og:description' content='Official YouTube channel'></head></html>",
        )

    extractor = WebsiteExtractor(transport=httpx.MockTransport(handler))
    with patch("app.services.website_extractor.validate_resolved_url", new=AsyncMock(return_value=("youtube.com", 443, ["142.250.1.1"]))):
        result = await extractor.extract("https://www.youtube.com/@YouTube")

    assert result.platform == "youtube"
    assert result.title == "YouTube"
    assert result.source == "public_html"
    assert result.extraction_status == "success"


@pytest.mark.asyncio
async def test_instagram_apify_fallback_is_used_when_configured() -> None:
    html_transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html><head><title>Instagram</title></head><body>Log in</body></html>",
        )
    )
    extractor = WebsiteExtractor(
        transport=html_transport,
        apify_api_token="test-token",
        apify_instagram_actor_id="apify/instagram-profile-scraper",
    )

    async def fake_apify(_: str) -> dict[str, object]:
        return {"username": "crestspot", "name": "CrestSpot", "description": "Creative visual studio"}

    extractor._extract_instagram_apify = fake_apify  # type: ignore[method-assign]
    with patch("app.services.website_extractor.validate_resolved_url", new=AsyncMock(return_value=("instagram.com", 443, ["31.13.65.36"]))):
        result = await extractor.extract("https://www.instagram.com/crestspot")

    assert result.title == "CrestSpot"
    assert result.meta_description == "Creative visual studio"
    assert result.extraction_status == "success"
    assert result.source == "apify"


@pytest.mark.asyncio
async def test_social_http_status_is_exposed() -> None:
    extractor = WebsiteExtractor(
        transport=httpx.MockTransport(lambda _: httpx.Response(403, content=b"blocked"))
    )
    with patch("app.services.website_extractor.validate_resolved_url", new=AsyncMock(return_value=("instagram.com", 443, ["31.13.65.36"]))):
        result = await extractor.extract("https://www.instagram.com/crestspot")
    assert result.platform == "instagram"
    assert result.extraction_status == "limited"
    assert "403" in result.limitations[0]


@pytest.mark.asyncio
async def test_social_redirect_to_private_host_is_rejected() -> None:
    requested_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    extractor = WebsiteExtractor(transport=httpx.MockTransport(handler))
    safe_dns = AsyncMock(return_value=("instagram.com", 443, ["31.13.65.36"]))
    with patch("app.services.website_extractor.validate_resolved_url", new=safe_dns):
        with pytest.raises(WebsiteHTTPError, match="invalid or unsafe URL"):
            await extractor.extract("https://www.instagram.com/crestspot")

    assert requested_urls == ["https://www.instagram.com/crestspot"]
    safe_dns.assert_awaited_once()


@pytest.mark.asyncio
async def test_social_response_size_limit_is_enforced() -> None:
    extractor = WebsiteExtractor(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"x" * 512,
            )
        ),
        max_response_bytes=100,
    )
    with patch(
        "app.services.website_extractor.validate_resolved_url",
        new=AsyncMock(return_value=("instagram.com", 443, ["31.13.65.36"])),
    ):
        with pytest.raises(WebsiteHTTPError, match="size limit"):
            await extractor.extract("https://www.instagram.com/crestspot")


@pytest.mark.asyncio
async def test_social_gzip_response_is_decoded_once() -> None:
    html = (
        b"<html><head><title>CrestSpot</title>"
        b"<meta property='og:description' content='CrestSpot is an independent creative studio for growing brands.'>"
        b"</head></html>"
    )
    extractor = WebsiteExtractor(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/html", "content-encoding": "gzip"},
                content=gzip.compress(html),
            )
        )
    )
    with patch(
        "app.services.website_extractor.validate_resolved_url",
        new=AsyncMock(return_value=("instagram.com", 443, ["31.13.65.36"])),
    ):
        result = await extractor.extract("https://www.instagram.com/crestspot")

    assert result.title == "CrestSpot"
    assert "independent creative studio" in result.meta_description
    assert result.extraction_status == "success"


def test_valid_http_and_https_urls_are_accepted() -> None:
    assert validate_url_syntax("https://example.com") == "https://example.com"
    assert validate_url_syntax("http://www.example.org/page") == "http://www.example.org/page"


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "javascript:alert(1)",
        "example.com",
        "https://user:pass@example.com",
        "https://example.com:99999",
        "https://example.com/path with spaces",
    ],
)
def test_invalid_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        validate_url_syntax(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://10.0.0.8",
        "http://192.168.1.5",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]",
        "http://localhost",
        "http://service.internal",
    ],
)
def test_private_and_internal_urls_are_rejected(url: str) -> None:
    with pytest.raises(UnsafeWebsiteURLError):
        validate_url_syntax(url)


@pytest.mark.asyncio
async def test_dns_results_reject_private_addresses() -> None:
    class FakeLoop:
        async def getaddrinfo(self, *_: object, **__: object) -> list[tuple[object, ...]]:
            return [(0, 0, 0, "", ("127.0.0.1", 443))]

    with patch("app.utils.validators.asyncio.get_running_loop", return_value=FakeLoop()):
        with pytest.raises(UnsafeWebsiteURLError):
            await resolve_public_addresses("example.com", 443)


def test_extractor_removes_navigation_noise_and_deduplicates_content() -> None:
    extractor = WebsiteExtractor(max_content_chars=1000)
    html = b"""
    <html><head><title>Acme | Solutions</title>
    <meta name="description" content="Reliable Acme solutions"></head>
    <body><nav>Menu noise</nav><script>bad()</script><main>
      <h1>Acme</h1><h2>Cloud services</h2><p>We deliver cloud services.</p>
      <p>We deliver cloud services.</p><a href="/services">Services</a>
      <a href="https://other.example/x">External</a>
    </main></body></html>
    """
    content = extractor.parse_html(html, "https://example.com")
    assert content.title == "Acme | Solutions"
    assert content.meta_description == "Reliable Acme solutions"
    assert content.headings == ["Acme", "Cloud services"]
    assert content.paragraphs == ["We deliver cloud services."]
    assert content.relevant_links == [{"text": "Services", "url": "https://example.com/services"}]
    assert "Menu noise" not in content.to_prompt_text(1000)


def test_empty_or_invalid_html_is_reported() -> None:
    extractor = WebsiteExtractor()
    with pytest.raises(EmptyWebsiteContentError):
        extractor.parse_html(b"<html><body><script>code()</script></body></html>", "https://example.com")
    content = extractor.parse_html(b"not valid html <p>Readable company information.</p>", "https://example.com")
    assert content.paragraphs == ["Readable company information."]


def test_title_only_site_is_not_treated_as_usable_content() -> None:
    with pytest.raises(EmptyWebsiteContentError, match="too little useful content"):
        WebsiteExtractor().parse_html(b"<html><title>Acme</title></html>", "https://example.com")


@pytest.mark.asyncio
async def test_fetch_follows_only_validated_redirects() -> None:
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path == "/":
            return httpx.Response(302, headers={"location": "https://example.com/about"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<title>Company</title><p>Useful business content for analysis.</p>",
        )

    transport = httpx.MockTransport(handler)
    extractor = WebsiteExtractor(transport=transport)
    with patch("app.services.website_extractor.validate_resolved_url", new=AsyncMock(return_value=("example.com", 443, ["93.184.216.34"]))):
        result = await extractor.extract("https://example.com/")
    assert len(requests) == 2
    assert result.title == "Company"


@pytest.mark.asyncio
async def test_default_transport_is_recreated_for_each_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transports: list[httpx.MockTransport] = []

    def make_transport() -> httpx.MockTransport:
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"<title>Company</title><p>Useful business content for analysis.</p>",
            )
        )
        transports.append(transport)
        return transport

    monkeypatch.setattr(extractor_module, "SSRFProtectedHTTPTransport", make_transport)
    extractor = WebsiteExtractor()
    safe_dns = AsyncMock(return_value=("example.com", 443, ["93.184.216.34"]))
    with patch("app.services.website_extractor.validate_resolved_url", new=safe_dns):
        await extractor.extract("https://example.com/")
        await extractor.extract("https://example.com/")
    assert len(transports) == 2


@pytest.mark.asyncio
async def test_private_redirect_is_rejected_before_fetch() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    extractor = WebsiteExtractor(transport=httpx.MockTransport(handler))
    with patch("app.services.website_extractor.validate_resolved_url", new=AsyncMock(return_value=("example.com", 443, ["93.184.216.34"]))):
        with pytest.raises(WebsiteHTTPError, match="unsafe URL"):
            await extractor.extract("https://example.com/")


@pytest.mark.asyncio
async def test_timeout_and_http_failure_have_clear_errors() -> None:
    async def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("late")

    extractor = WebsiteExtractor(transport=httpx.MockTransport(timeout))
    with patch("app.services.website_extractor.validate_resolved_url", new=AsyncMock(return_value=("example.com", 443, ["93.184.216.34"]))):
        with pytest.raises(WebsiteTimeoutError, match="timed out"):
            await extractor.extract("https://example.com/")

    async def not_found(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    extractor = WebsiteExtractor(transport=httpx.MockTransport(not_found))
    with patch("app.services.website_extractor.validate_resolved_url", new=AsyncMock(return_value=("example.com", 443, ["93.184.216.34"]))):
        with pytest.raises(WebsiteHTTPError, match="404"):
            await extractor.extract("https://example.com/")


@pytest.mark.asyncio
async def test_unreachable_website_is_reported() -> None:
    async def unreachable(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    extractor = WebsiteExtractor(transport=httpx.MockTransport(unreachable))
    with patch("app.services.website_extractor.validate_resolved_url", new=AsyncMock(return_value=("example.com", 443, ["93.184.216.34"]))):
        with pytest.raises(WebsiteHTTPError, match="could not be reached"):
            await extractor.extract("https://example.com/")


@pytest.mark.asyncio
async def test_network_backend_pins_the_vetted_public_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBackend:
        connected_host: str | None = None

        async def connect_tcp(self, **kwargs: object) -> object:
            self.connected_host = str(kwargs["host"])
            return object()

    fake_backend = FakeBackend()
    backend = extractor_module._PublicDNSNetworkBackend()
    backend._backend = fake_backend

    async def resolve(_: str, __: int) -> list[str]:
        return ["93.184.216.34"]

    monkeypatch.setattr(extractor_module, "resolve_public_addresses", resolve)
    await backend.connect_tcp("example.com", 443)
    assert fake_backend.connected_host == "93.184.216.34"
