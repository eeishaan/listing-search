#!/usr/bin/env python3
"""
Fetches a TRREB listing page with Playwright, handles the cookie consent
prompt when present, and writes the fully loaded HTML to disk.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

LISTING_URL = "https://onlistings.trreb.ca/searchlistings#search/6b6aa7df9bdf684ea2e44f88/listing/TREB-N11980487"
DEFAULT_OUTPUT = Path("data/trreb_listing.html")
ACCEPT_TIMEOUT_MS = 8000
ACCEPT_SELECTORS = (
    "#consumerAgreementPopoutButton",
    "#consumerAgreementPopoutHeader button",
    "#consumerAgreementPopoutHeader",
    "role=button[name=/accept/i]",
    "button:has-text('Accept')",
    "text=/\\baccept\\b/i",
)


async def maybe_accept_terms(page) -> None:
    """Click the cookie/terms consent button in the main frame or any child frame."""

    async def _click_in_scope(scope) -> bool:
        for selector in ACCEPT_SELECTORS:
            candidate = scope.locator(selector).first
            try:
                await candidate.wait_for(state="visible", timeout=ACCEPT_TIMEOUT_MS)
                await candidate.click()
                await page.wait_for_timeout(750)
                return True
            except PlaywrightTimeoutError:
                continue
        return False

    # The main page plus any iframes that might host the popup.
    scopes = [page, *page.frames]
    for scope in scopes:
        if await _click_in_scope(scope):
            return


async def fetch_listing(url: str, output_path: Path, headed: bool) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        await maybe_accept_terms(page)
        await page.wait_for_load_state("networkidle")

        html = await page.content()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        await context.close()
        await browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a TRREB listing with Playwright."
    )
    parser.add_argument("--url", default=LISTING_URL, help="Listing URL to download.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to store the HTML source.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Launch browser in headed mode for debugging.",
    )
    return parser.parse_args()


async def _async_main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    await fetch_listing(args.url, output_path, headed=args.headed)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
