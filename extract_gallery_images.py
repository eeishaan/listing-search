#!/usr/bin/env python3
"""
Utility to extract all gallery image URLs from a downloaded TRREB listing page.

Example:
    python extract_gallery_images.py page_source.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

DEFAULT_HTML = Path("page_source.html")
DEFAULT_BASE_URL = "https://onlistings.trreb.ca"


def _uniquify(sequence: Iterable[str]) -> list[str]:
    """Return items in their first-seen order without duplicates."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in sequence:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def extract_gallery_image_urls(html: str, base_url: str | None = None) -> list[str]:
    """Parse the listing HTML and return all image URLs in the gallery."""
    soup = BeautifulSoup(html, "html.parser")
    photo_list = soup.select_one(".listing-full .photos-slideshow") or soup.select_one(
        ".photos-slideshow"
    )

    nodes: list[Tag] = []
    if photo_list:
        nodes = photo_list.find_all("li")
    else:
        nodes = [img for img in soup.select("img.listing-photo")]

    urls: list[str] = []
    for node in nodes:
        url: str | None = None
        if node.has_attr("data-src"):
            url = node["data-src"]
        else:
            img = node if node.name == "img" else node.find("img")
            if img:
                url = img.get("data-src") or img.get("src")

        if not url:
            continue

        url = url.strip()
        if not url:
            continue

        if base_url:
            url = urljoin(base_url, url)

        urls.append(url)

    return _uniquify(urls)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract gallery image URLs from a TRREB listing HTML document."
    )
    parser.add_argument(
        "html_path",
        nargs="?",
        default=str(DEFAULT_HTML),
        help="Path to the saved listing HTML (default: %(default)s).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=(
            "Base URL to join with relative image paths "
            "(default: %(default)s). Use an empty string to keep paths relative."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to write the URLs. Defaults to stdout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the results as JSON instead of newline-delimited text.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    html_path = Path(args.html_path).expanduser().resolve()
    html_text = html_path.read_text(encoding="utf-8")

    base_url: str | None = args.base_url
    if base_url is not None:
        base_url = base_url.strip() or None

    urls = extract_gallery_image_urls(html_text, base_url=base_url)

    if args.json:
        output_text = json.dumps(urls, indent=2)
    else:
        output_text = "\n".join(urls)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
