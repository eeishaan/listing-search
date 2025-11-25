#!/usr/bin/env python3
"""
Download a TRREB listing URL, extract gallery images and key details,
and persist everything to a JSON file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from custom_parser import _extract_details
from download_listing_with_playwright import fetch_listing
from extract_gallery_images import extract_gallery_image_urls


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a TRREB listing and save the parsed details as JSON."
    )
    parser.add_argument(
        "url",
        help="Listing URL to download and parse.",
    )
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


async def _download_listing(url: str, output_path: Path, headed: bool) -> None:
    await fetch_listing(url, output_path, headed=headed)


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    html_path: Path | None = None

    if args.html_output:
        html_path = Path(args.html_output).expanduser().resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(_download_listing(args.url, html_path, headed=args.headed))
    else:
        with TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / "listing.html"
            asyncio.run(_download_listing(args.url, html_path, headed=args.headed))
            html_text = html_path.read_text(encoding="utf-8")
            _persist_details(
                args.url,
                html_text,
                output_path=output_path,
            )
            return

    html_text = html_path.read_text(encoding="utf-8")
    _persist_details(
        args.url,
        html_text,
        output_path=output_path,
    )


def _persist_details(url: str, html_text: str, output_path: Path) -> None:
    details = _extract_details(html_text)
    base_url = _base_url_from_listing(url)
    images = extract_gallery_image_urls(html_text, base_url=base_url)

    payload = {
        "source_url": url,
        "price": details.get("price"),
        "taxes": details.get("taxes"),
        "tax_year": details.get("tax_year"),
        "address": details.get("address"),
        "house_type": details.get("house_type"),
        "description": details.get("description"),
        "images": images,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

