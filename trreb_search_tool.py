#!/usr/bin/env python3
"""
TRREB Listing Search Tool

A Playwright-based tool that sets search filters on the TRREB listing search page
and extracts listing results (listing number, address, URL).
"""

from __future__ import annotations

import asyncio
import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from playwright.async_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

# =============================================================================
# Schemas
# =============================================================================


class TREBSearchParams(BaseModel):
    """Input parameters for the TRREB listing search tool."""

    # Basic search
    location: str | None = Field(
        default=None,
        description="City, postal code, address, or MLS# to search for",
    )
    listing_type: Literal["sale", "lease"] = Field(
        default="sale",
        description="Type of listing: 'sale' or 'lease'",
    )

    # Property categories (can select multiple)
    property_categories: list[Literal["freehold", "condo", "commercial"]] | None = (
        Field(
            default=None,
            description="Property categories to include: freehold, condo, commercial",
        )
    )

    # Price range
    price_min: int | None = Field(
        default=None,
        description="Minimum price (e.g., 600000 for $600,000)",
    )
    price_max: int | None = Field(
        default=None,
        description="Maximum price (e.g., 1000000 for $1,000,000)",
    )

    # Bedrooms (multi-select: 0, 1, 2, 3, 4, 5+)
    bedrooms: list[Literal["0", "1", "2", "3", "4", "5+"]] | None = Field(
        default=None,
        description="Number of bedrooms to filter by (can select multiple)",
    )
    bedrooms_plus: Literal["any", "yes", "no"] | None = Field(
        default=None,
        description="Filter for additional bedrooms (den): any, yes, or no",
    )

    # Bathrooms (multi-select: 0, 1, 2, 3, 4+)
    bathrooms: list[Literal["0", "1", "2", "3", "4+"]] | None = Field(
        default=None,
        description="Number of bathrooms to filter by (can select multiple)",
    )

    # Property types (Freehold)
    freehold_types: list[str] | None = Field(
        default=None,
        description=(
            "Freehold property types: Detached, Semi-Detached, "
            "Attached/Row House/Townhouse, Duplex, Triplex, Fourplex, "
            "Multiplex, Link, Farm, Cottage, Vacant Land, etc."
        ),
    )

    # Condo types
    condo_types: list[str] | None = Field(
        default=None,
        description=(
            "Condo property types: Condo Apartment, Condo Townhouse, "
            "Co-Op Apartment, Detached Condo, Locker, Parking Space, etc."
        ),
    )

    # Style
    styles: list[str] | None = Field(
        default=None,
        description=(
            "Property styles: 2-Storey, 3-Storey, Bungalow, Backsplit, "
            "Sidesplit, Apartment, Bachelor/Studio, Loft, etc."
        ),
    )

    # Square footage
    sqft_ranges: list[str] | None = Field(
        default=None,
        description=(
            "Square footage ranges: '< 700', '700-1,100', '1,100-1,500', "
            "'1,500-2,000', '2,000-2,500', '2,500-3,000', '3,000-3,500', "
            "'3,500-5,000', '5,000+'"
        ),
    )

    # Special filters
    has_open_house: bool | None = Field(
        default=None,
        description="Filter for properties with upcoming open houses",
    )
    has_live_stream: bool | None = Field(
        default=None,
        description="Filter for properties with live stream open houses",
    )


class ListingResult(BaseModel):
    """A single listing result from the search."""

    listing_number: str = Field(
        description="The MLS listing number (e.g., 'C12410955')",
    )
    address: str = Field(
        description="The property address (e.g., '20 Fashion Roseway 408, Toronto')",
    )
    url: str = Field(
        description="The full URL to the listing detail page",
    )


class SearchResults(BaseModel):
    """Results from a TRREB listing search."""

    listings: list[ListingResult] = Field(
        default_factory=list,
        description="List of matching listings",
    )
    total_count: int = Field(
        default=0,
        description="Total number of listings found",
    )
    search_url: str = Field(
        default="",
        description="The URL of the search results page",
    )


# =============================================================================
# Constants
# =============================================================================

SEARCH_URL = "https://onlistings.trreb.ca/searchlistings"
ACCEPT_TIMEOUT_MS = 300
PAGE_LOAD_TIMEOUT_MS = 3000
FILTER_WAIT_MS = 300
ELEMENT_TIMEOUT_MS = 200

ACCEPT_SELECTORS = (
    "#consumerAgreementPopoutButton",
    "#consumerAgreementPopoutHeader button",
    "#consumerAgreementPopoutHeader",
    "role=button[name=/accept/i]",
    "button:has-text('Accept')",
    "text=/\\baccept\\b/i",
)

# Price dropdown values mapping
PRICE_OPTIONS_SALE = [
    0,
    25000,
    50000,
    75000,
    100000,
    125000,
    150000,
    175000,
    200000,
    225000,
    250000,
    275000,
    300000,
    325000,
    350000,
    375000,
    400000,
    425000,
    450000,
    475000,
    500000,
    550000,
    600000,
    650000,
    700000,
    750000,
    800000,
    850000,
    900000,
    950000,
    1000000,
    1100000,
    1200000,
    1300000,
    1400000,
    1500000,
    1600000,
    1700000,
    1800000,
    1900000,
    2000000,
    2500000,
    3000000,
    4000000,
    5000000,
    7500000,
    10000000,
    20000000,
    30000000,
    40000000,
    50000000,
]


async def switch_to_list_view(page: Page) -> None:
    """Switch to list view. Fails fast if element not found."""
    list_view = page.locator("label.btn-table").filter(has_text="List")
    await list_view.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
    await list_view.click()
    await page.wait_for_timeout(FILTER_WAIT_MS)


# =============================================================================
# Filter Setting Functions
# =============================================================================


async def maybe_accept_terms(page: Page) -> None:
    """Click the cookie/terms consent button if present."""

    async def _click_in_scope(scope) -> bool:
        for selector in ACCEPT_SELECTORS:
            candidate = scope.locator(selector).first
            try:
                await candidate.wait_for(state="visible", timeout=ACCEPT_TIMEOUT_MS)
                await candidate.click()
                await page.wait_for_timeout(300)
                return True
            except PlaywrightTimeoutError:
                continue
        return False

    scopes = [page, *page.frames]
    for scope in scopes:
        if await _click_in_scope(scope):
            return


async def set_listing_type(page: Page, listing_type: Literal["sale", "lease"]) -> None:
    """Switch between Sale and Lease tabs. Fails fast if element not found."""
    if listing_type == "sale":
        sale_tab = page.locator("#saleOrRent-sale")
        await sale_tab.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        await sale_tab.click()
    else:
        lease_tab = page.locator("#saleOrRent-rent")
        await lease_tab.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        await lease_tab.click()
    await page.wait_for_timeout(FILTER_WAIT_MS)


async def set_location(page: Page, location: str) -> None:
    """Set the location search field. Fails fast if element not found."""
    # Use the actual input with id="loc" (not the readonly hint input)
    location_input = page.locator("input#loc")
    await location_input.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
    await location_input.click()
    await location_input.fill(location)
    await page.wait_for_timeout(FILTER_WAIT_MS)
    await location_input.press("Enter")
    await page.wait_for_timeout(500)


async def set_property_categories(
    page: Page, categories: list[Literal["freehold", "condo", "commercial"]]
) -> None:
    """Set property category checkboxes based on hidden values. Fails fast."""
    category_map = {
        "freehold": "FREE",
        "condo": "CONDO",
        "commercial": "COM",
    }

    for category, value in category_map.items():
        checkbox = page.locator(f"input[name='class'][value='{value}']")
        await checkbox.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        is_checked = await checkbox.is_checked()
        should_be_checked = category in categories

        if should_be_checked and not is_checked:
            await checkbox.check()
            await page.wait_for_timeout(FILTER_WAIT_MS)
        elif not should_be_checked and is_checked:
            await checkbox.uncheck()
            await page.wait_for_timeout(FILTER_WAIT_MS)


def _find_closest_price(target: int, options: list[int]) -> int:
    """Find the closest price option to the target value."""
    return min(options, key=lambda x: abs(x - target))


def _price_option_value(amount: int, bound: Literal["min", "max"]) -> str:
    """Convert a numeric amount into the select option value."""
    if amount == 0:
        return ""
    prefix = ">=" if bound == "min" else "<="
    return f"{prefix}{amount}"


async def set_price_range(
    page: Page, price_min: int | None, price_max: int | None
) -> None:
    """Set the price range dropdowns. Fails fast if element not found."""
    # Find the Price section and its comboboxes
    price_label = page.locator("label:has-text('Price')").first
    await price_label.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
    price_section = price_label.locator("..")

    if price_min is not None:
        closest_min = _find_closest_price(price_min, PRICE_OPTIONS_SALE)
        option_value = _price_option_value(closest_min, "min")

        # First combobox is "From" - use select_option for native select elements
        from_combo = price_section.get_by_role("combobox").first
        await from_combo.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        await from_combo.select_option(value=option_value)
        await page.wait_for_timeout(FILTER_WAIT_MS)

    if price_max is not None:
        closest_max = _find_closest_price(price_max, PRICE_OPTIONS_SALE)
        option_value = _price_option_value(closest_max, "max")

        # Second combobox is "To"
        to_combo = price_section.get_by_role("combobox").nth(1)
        await to_combo.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        await to_combo.select_option(value=option_value)
        await page.wait_for_timeout(FILTER_WAIT_MS)


async def set_bedrooms(
    page: Page,
    bedrooms: list[str] | None,
    bedrooms_plus: Literal["any", "yes", "no"] | None,
) -> None:
    """Set bedroom filter checkboxes. Fails fast if element not found."""
    if bedrooms:
        # Use specific input name="bedrooms" to target bedroom checkboxes
        for bed_count in ["0", "1", "2", "3", "4", "5+"]:
            value = ">=5" if bed_count == "5+" else bed_count
            input_locator = page.locator(f"input[name='bedrooms'][value='{value}']")
            await input_locator.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
            label_locator = input_locator.locator("..")
            is_checked = await input_locator.is_checked()
            should_be_checked = bed_count in bedrooms

            if should_be_checked and not is_checked:
                await label_locator.click()
                await page.wait_for_timeout(FILTER_WAIT_MS)
            elif not should_be_checked and is_checked:
                await label_locator.click()
                await page.wait_for_timeout(FILTER_WAIT_MS)

    if bedrooms_plus:
        # Find the Bedroom+ radio buttons
        plus_map = {
            "any": "Any",
            "yes": "Ye",
            "no": "No",
        }  # "Yes" appears as "Ye" in snapshot
        radio_name = plus_map.get(bedrooms_plus, "Any")

        bedroom_plus_section = page.locator("label:has-text('Bedroom')").locator(
            "text=/\\+/"
        )
        await bedroom_plus_section.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        parent = bedroom_plus_section.locator("..").locator("..")
        radio = parent.get_by_role("radio", name=radio_name)
        await radio.check()
        await page.wait_for_timeout(FILTER_WAIT_MS)


async def set_bathrooms(page: Page, bathrooms: list[str]) -> None:
    """Set bathroom filter checkboxes. Fails fast if element not found."""
    # Use specific input name="bathrooms" to target bathroom checkboxes
    for bath_count in ["0", "1", "2", "3", "4+"]:
        value = ">=4" if bath_count == "4+" else bath_count
        input_locator = page.locator(f"input[name='bathrooms'][value='{value}']")
        await input_locator.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        label_locator = input_locator.locator("..")
        is_checked = await input_locator.is_checked()
        should_be_checked = bath_count in bathrooms

        if should_be_checked and not is_checked:
            await label_locator.click()
            await page.wait_for_timeout(FILTER_WAIT_MS)
        elif not should_be_checked and is_checked:
            await label_locator.click()
            await page.wait_for_timeout(FILTER_WAIT_MS)


async def set_property_types(page: Page, types: list[str], section_label: str) -> None:
    """Set property type checkboxes in a multi-select dropdown. Fails fast."""
    # Find the Type section with the given label context
    type_sections = page.locator(f"label:has-text('{section_label}')").locator(
        ".. >> label:has-text('Type')"
    )
    await type_sections.first.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)

    # Click the "Any" button to open the dropdown
    any_button = type_sections.first.locator(".. >> button:has-text('Any')")
    await any_button.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
    await any_button.click()
    await page.wait_for_timeout(300)

    # Check the specified types
    for prop_type in types:
        checkbox = page.get_by_role("checkbox", name=prop_type, exact=True)
        await checkbox.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        await checkbox.check()
        await page.wait_for_timeout(200)

    # Close dropdown by clicking elsewhere
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(FILTER_WAIT_MS)


async def set_styles(page: Page, styles: list[str]) -> None:
    """Set style filter checkboxes. Fails fast if element not found."""
    # Find the Style section
    style_section = page.locator("label:has-text('Style')").first
    await style_section.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)

    # Click the "Any" button to open the dropdown
    any_button = style_section.locator(".. >> button:has-text('Any')")
    await any_button.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
    await any_button.click()
    await page.wait_for_timeout(300)

    for style in styles:
        checkbox = page.get_by_role("checkbox", name=style, exact=True)
        await checkbox.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        await checkbox.check()
        await page.wait_for_timeout(200)

    await page.keyboard.press("Escape")
    await page.wait_for_timeout(FILTER_WAIT_MS)


async def set_sqft_ranges(page: Page, ranges: list[str]) -> None:
    """Set square footage range checkboxes. Fails fast if element not found."""
    sqft_section = page.locator("label:has-text('Approximate Square Footage')").first
    await sqft_section.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)

    any_button = sqft_section.locator(".. >> button:has-text('Any')")
    await any_button.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
    await any_button.click()
    await page.wait_for_timeout(300)

    for sqft_range in ranges:
        checkbox = page.get_by_role("checkbox", name=sqft_range, exact=True)
        await checkbox.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        await checkbox.check()
        await page.wait_for_timeout(200)

    await page.keyboard.press("Escape")
    await page.wait_for_timeout(FILTER_WAIT_MS)


async def set_open_house_filter(page: Page, has_open_house: bool) -> None:
    """Set the open house filter checkbox. Fails fast if element not found."""
    if has_open_house:
        checkbox = page.get_by_role(
            "checkbox", name=re.compile(r"Ha.*upcoming open hou", re.IGNORECASE)
        )
        await checkbox.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        await checkbox.check()
        await page.wait_for_timeout(FILTER_WAIT_MS)


async def set_live_stream_filter(page: Page, has_live_stream: bool) -> None:
    """Set the live stream open house filter checkbox. Fails fast."""
    if has_live_stream:
        checkbox = page.get_by_role(
            "checkbox", name=re.compile(r"Ha.*Live Stream Open Hou", re.IGNORECASE)
        )
        await checkbox.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)
        await checkbox.check()
        await page.wait_for_timeout(FILTER_WAIT_MS)


# =============================================================================
# Result Extraction
# =============================================================================

# Pattern to extract listing number (e.g., #C12410955 or #N11980487)
LISTING_NUMBER_PATTERN = re.compile(r"#([A-Z]\d+)")


async def extract_listing_results(page: Page) -> list[ListingResult]:
    """Extract listing results from the search results page."""
    results: list[ListingResult] = []

    # Wait briefly for results panel to update
    await page.wait_for_timeout(500)

    # Get the current URL to construct listing URLs
    current_url = page.url
    parsed = urlparse(current_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    # Extract search ID from URL (format: #search/{search_id}/...)
    search_id_match = re.search(r"#search/([^/]+)", current_url)
    search_id = search_id_match.group(1) if search_id_match else ""

    # Find listing items in the results list
    # Find table with ID starting with DataTables_Table_*
    listing_table = page.locator("[id^='DataTables_Table_']").first
    await listing_table.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_MS)

    # Find rows in the table
    listing_rows = await listing_table.locator("tbody tr").all()

    print(f"found {len(listing_rows)} listings")

    for row in listing_rows:
        # Extract listing number from data-id attribute
        data_id = await row.get_attribute("data-id", timeout=ELEMENT_TIMEOUT_MS)
        if not data_id or not data_id.startswith("TREB-"):
            continue

        listing_number = data_id.replace("TREB-", "")

        # Extract address from the address column (index 2)
        address_cell = row.locator("td").nth(2)
        address = await address_cell.inner_text(timeout=ELEMENT_TIMEOUT_MS)
        address = address.strip()

        listing_url = f"{base_url}/searchlistings#search/{search_id}/listing/TREB-{listing_number}"

        results.append(
            ListingResult(
                listing_number=listing_number,
                address=address,
                url=listing_url,
            )
        )

    return results


async def get_total_count(page: Page) -> int:
    """Get the total number of listings from the page."""
    try:
        # Look for text like "12 Listings" or similar count indicator
        count_element = page.locator("text=/\\d+\\s*Li.*ting/i").first
        count_text = await count_element.inner_text()
        match = re.search(r"(\d+)", count_text)
        if match:
            return int(match.group(1))
    except (PlaywrightTimeoutError, Exception):
        pass
    return 0


# =============================================================================
# Main Tool Function
# =============================================================================


async def search_listings_async(
    params: TREBSearchParams,
    *,
    headed: bool = False,
    max_results: int = 50,
) -> SearchResults:
    """
    Search TRREB listings with the specified filters.

    Args:
        params: Search parameters including location, price range, bedrooms, etc.
        headed: If True, launch browser in headed mode for debugging.
        max_results: Maximum number of results to return.

    Returns:
        SearchResults containing matching listings.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            # Navigate to search page
            await page.goto(
                SEARCH_URL, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT_MS
            )
            await page.wait_for_timeout(500)

            # Accept terms if prompted
            await maybe_accept_terms(page)
            await page.wait_for_load_state(
                "domcontentloaded", timeout=PAGE_LOAD_TIMEOUT_MS
            )

            # Swithc to list view
            await switch_to_list_view(page)

            # Apply filters
            # 1. Set listing type (Sale/Lease)
            await set_listing_type(page, params.listing_type)

            # 2. Set location if provided
            if params.location:
                await set_location(page, params.location)

            # 3. Set property categories
            if params.property_categories:
                await set_property_categories(page, params.property_categories)

            # 4. Set price range
            if params.price_min is not None or params.price_max is not None:
                await set_price_range(page, params.price_min, params.price_max)

            # 5. Set bedrooms
            if params.bedrooms or params.bedrooms_plus:
                await set_bedrooms(page, params.bedrooms, params.bedrooms_plus)

            # 6. Set bathrooms
            if params.bathrooms:
                await set_bathrooms(page, params.bathrooms)

            # 7. Set freehold property types
            if params.freehold_types:
                await set_property_types(page, params.freehold_types, "Freehold")

            # 8. Set condo types
            if params.condo_types:
                await set_property_types(page, params.condo_types, "Condo")

            # 9. Set styles
            if params.styles:
                await set_styles(page, params.styles)

            # 10. Set square footage
            if params.sqft_ranges:
                await set_sqft_ranges(page, params.sqft_ranges)

            # 11. Set open house filter
            if params.has_open_house:
                await set_open_house_filter(page, params.has_open_house)

            # 12. Set live stream filter
            if params.has_live_stream:
                await set_live_stream_filter(page, params.has_live_stream)

            # Wait for results to update
            await page.wait_for_timeout(1000)
            await page.wait_for_load_state("networkidle", timeout=PAGE_LOAD_TIMEOUT_MS)

            # Extract results
            listings = await extract_listing_results(page)
            total_count = await get_total_count(page)
            search_url = page.url

            # Limit results
            if len(listings) > max_results:
                listings = listings[:max_results]

            return SearchResults(
                listings=listings,
                total_count=total_count or len(listings),
                search_url=search_url,
            )

        finally:
            await context.close()
            await browser.close()


def search_listings(
    params: TREBSearchParams,
    *,
    headed: bool = False,
    max_results: int = 50,
) -> SearchResults:
    """
    Synchronous wrapper for search_listings_async.

    Search TRREB listings with the specified filters.

    Args:
        params: Search parameters including location, price range, bedrooms, etc.
        headed: If True, launch browser in headed mode for debugging.
        max_results: Maximum number of results to return.

    Returns:
        SearchResults containing matching listings.
    """
    return asyncio.run(
        search_listings_async(params, headed=headed, max_results=max_results)
    )


# =============================================================================
# Tool Definition for AI Agents
# =============================================================================

TOOL_DEFINITION = {
    "name": "search_trreb_listings",
    "description": (
        "Search for real estate listings on the Toronto Regional Real Estate Board (TRREB) "
        "website. Set various filters like location, price range, bedrooms, bathrooms, "
        "property type, and more. Returns a list of matching listings with their MLS numbers, "
        "addresses, and URLs."
    ),
    "input_schema": TREBSearchParams.model_json_schema(),
    "output_schema": SearchResults.model_json_schema(),
}


# =============================================================================
# CLI for Testing
# =============================================================================


def main() -> None:
    """Simple CLI for testing the search tool."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Search TRREB listings")
    parser.add_argument("--location", help="Location to search")
    parser.add_argument(
        "--listing-type",
        choices=["sale", "lease"],
        default="sale",
        help="Listing type",
    )
    parser.add_argument("--price-min", type=int, help="Minimum price")
    parser.add_argument("--price-max", type=int, help="Maximum price")
    parser.add_argument(
        "--bedrooms", nargs="+", help="Number of bedrooms (e.g., 2 3 4)"
    )
    parser.add_argument(
        "--bathrooms", nargs="+", help="Number of bathrooms (e.g., 1 2)"
    )
    parser.add_argument(
        "--headed", action="store_true", help="Run browser in headed mode"
    )
    parser.add_argument(
        "--max-results", type=int, default=20, help="Maximum results to return"
    )

    args = parser.parse_args()

    params = TREBSearchParams(
        location=args.location,
        listing_type=args.listing_type,
        price_min=args.price_min,
        price_max=args.price_max,
        bedrooms=args.bedrooms,
        bathrooms=args.bathrooms,
    )

    print(f"Searching with params: {params.model_dump_json(indent=2)}")
    print("-" * 50)

    results = search_listings(params, headed=args.headed, max_results=args.max_results)

    print(f"\nFound {results.total_count} listings")
    print(f"Search URL: {results.search_url}")
    print(f"\nReturning {len(results.listings)} results:\n")

    for listing in results.listings:
        print(f"  #{listing.listing_number}: {listing.address}")
        print(f"    URL: {listing.url}")
        print()

    # Also output as JSON
    print("\n" + "=" * 50)
    print("JSON Output:")
    print(results.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
