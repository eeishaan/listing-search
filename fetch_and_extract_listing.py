#!/usr/bin/env python3
"""
Library + CLI for downloading TRREB listings and extracting structured data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from custom_parser import _extract_details
from download_listing_with_playwright import fetch_listing
from extract_gallery_images import extract_gallery_image_urls


class ListingExtractionParams(BaseModel):
    """Parameters for extracting listing details."""

    url: str = Field(
        description="The URL of the TRREB listing to extract details from."
    )


class ListingExtractionResult(BaseModel):
    """Structured details of a real estate listing."""

    source_url: str = Field(description="The URL of the listing.")
    price: Optional[str] = Field(description="The listed price of the property.")
    taxes: Optional[str] = Field(description="Annual property taxes.")
    tax_year: Optional[str] = Field(description="The tax year for the reported taxes.")
    address: Optional[str] = Field(description="The full address of the property.")
    house_type: Optional[str] = Field(
        description="Type of the property (e.g., Detached)."
    )
    description: Optional[str] = Field(description="Description of the property.")
    images: List[str] = Field(
        default_factory=list, description="List of image URLs for the property."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a TRREB listing and save the parsed details as JSON."
    )
    parser.add_argument("url", help="Listing URL to download and parse.")
    parser.add_argument(
        "--output",
        "-o",
        default="listing_details.json",
        help="Path to the JSON file to write (default: %(default)s).",
    )
    parser.add_argument(
        "--html-output",
        help=(
            "Optional path to also store the downloaded HTML. "
            "If omitted, a temporary file is used and removed afterwards."
        ),
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Launch the browser in headed mode while downloading.",
    )
    return parser.parse_args()


def _base_url_from_listing(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid listing URL: {url}")
    return f"{parsed.scheme}://{parsed.netloc}"


async def download_listing_html(
    url: str, *, headed: bool = False, html_output: Path | None = None
) -> str:
    """
    Download the full HTML for a listing URL via Playwright.

    Returns the HTML text; optionally persists it to html_output.
    """
    if html_output:
        target = Path(html_output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        await fetch_listing(url, target, headed=headed)
        return target.read_text(encoding="utf-8")

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "listing.html"
        await fetch_listing(url, tmp_path, headed=headed)
        return tmp_path.read_text(encoding="utf-8")


def download_listing_html_sync(
    url: str, *, headed: bool = False, html_output: Path | None = None
) -> str:
    """Synchronous helper that wraps download_listing_html."""
    return asyncio.run(
        download_listing_html(url, headed=headed, html_output=html_output)
    )


def parse_listing_details(html_text: str, *, source_url: str) -> Dict[str, Any]:
    """Convert listing HTML into structured data."""
    details = _extract_details(html_text)
    base_url = _base_url_from_listing(source_url)
    images = extract_gallery_image_urls(html_text, base_url=base_url)
    return {
        "source_url": source_url,
        "price": details.get("price"),
        "taxes": details.get("taxes"),
        "tax_year": details.get("tax_year"),
        "address": details.get("address"),
        "house_type": details.get("house_type"),
        "description": details.get("description"),
        "images": images,
    }


async def extract_listing_details_async(
    url: str, *, headed: bool = False, html_output: Path | None = None
) -> Dict[str, Any]:
    """Async API that downloads a listing and returns parsed details."""
    html_text = await download_listing_html(url, headed=headed, html_output=html_output)
    return parse_listing_details(html_text, source_url=url)


class Cache:
    def __init__(self, cache_dir: Path = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, url: str) -> Dict[str, Any]:
        listing_num = url.split("/")[-1].strip()
        print("searching cache for", listing_num)
        cache_file = self.cache_dir / f"{listing_num}.json"
        if cache_file.exists():
            print("cache hit")
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return None

    def set(self, url: str, details: Dict[str, Any]):
        listing_num = url.split("/")[-1]
        cache_file = self.cache_dir / f"{listing_num}.json"
        cache_file.write_text(json.dumps(details, indent=2), encoding="utf-8")


GLOBAL_CACHE = Cache()


def extract_listing_details(
    url: str, *, headed: bool = False, html_output: Path | None = None
) -> Dict[str, Any]:
    """Sync wrapper returning listing details."""
    result = GLOBAL_CACHE.get(url)
    if result:
        return result
    result = asyncio.run(
        extract_listing_details_async(url, headed=headed, html_output=html_output)
    )
    GLOBAL_CACHE.set(url, result)
    return result


def save_listing_details(details: Dict[str, Any], output_path: Path) -> None:
    """Persist listing details to JSON."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(details, indent=2), encoding="utf-8")


# =============================================================================
# Tool Definition for AI Agents
# =============================================================================

TOOL_DEFINITION = {
    "name": "extract_listing_details",
    "description": "Extracts detailed information from a specific TRREB listing URL, including price, address, description, and images.",
    "input_schema": ListingExtractionParams.model_json_schema(),
    "output_schema": ListingExtractionResult.model_json_schema(),
}


def main() -> None:
    args = _parse_args()
    html_output_path = (
        Path(args.html_output).expanduser().resolve() if args.html_output else None
    )
    details = extract_listing_details(
        args.url,
        headed=args.headed,
        html_output=html_output_path,
    )
    save_listing_details(details, Path(args.output))


if __name__ == "__main__":
    main()
