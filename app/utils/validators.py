"""Syntactic URL validation and DNS/IP address safety checks."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit


class UnsafeWebsiteURLError(ValueError):
    """Raised when a URL could target a local or non-public network address."""


def validate_url_syntax(url: str) -> str:
    """Validate an absolute HTTP(S) URL without performing network access."""
    candidate = url.strip()
    if candidate != url or any(char.isspace() for char in candidate):
        raise ValueError("URL must not contain surrounding or embedded whitespace.")
    if not candidate or len(candidate) > 2048:
        raise ValueError("URL must be between 1 and 2048 characters.")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL is malformed.") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")
    if not parsed.hostname or not parsed.netloc:
        raise ValueError("URL must include a valid host name.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing user credentials are not allowed.")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("URL port must be between 1 and 65535.")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(
        (".localhost", ".local", ".internal", ".test", ".invalid")
    ):
        raise UnsafeWebsiteURLError("Local and internal host names are not allowed.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            host.encode("idna")
        except UnicodeError as exc:
            raise ValueError("URL host name is invalid.") from exc
    else:
        _ensure_public_address(address)
    return candidate


def _ensure_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if not address.is_global:
        raise UnsafeWebsiteURLError("Private, reserved, and non-routable IP addresses are not allowed.")


async def resolve_public_addresses(host: str, port: int) -> list[str]:
    """Resolve a host and reject the entire result if any address is non-public."""
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            results = await asyncio.wait_for(
                asyncio.get_running_loop().getaddrinfo(
                    host,
                    port,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                ),
                timeout=5,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise UnsafeWebsiteURLError("Website host could not be resolved safely.") from exc
        addresses = list(dict.fromkeys(item[4][0] for item in results))
        if not addresses:
            raise UnsafeWebsiteURLError("Website host did not resolve to an IP address.")
    else:
        addresses = [str(literal)]

    for raw_address in addresses:
        address = ipaddress.ip_address(raw_address)
        _ensure_public_address(address)
    return addresses


async def validate_resolved_url(url: str) -> tuple[str, int, list[str]]:
    """Validate a URL and resolve its host before an outbound request."""
    validate_url_syntax(url)
    parsed = urlsplit(url)
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeWebsiteURLError("Website URL must include a host name.")
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    addresses = await resolve_public_addresses(hostname, port)
    return hostname, port, addresses
