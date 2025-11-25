from pathlib import Path
import sys

from bs4 import BeautifulSoup, NavigableString


def _normalize_whitespace(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _extract_details(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    price_node = soup.select_one("div.container.listing-full div.price h1 span")
    if not price_node:
        raise ValueError("Price element not found in HTML document")
    price_text = _normalize_whitespace(
        price_node.get_text(strip=True).replace("\xa0", " ")
    )

    taxes_label = soup.find(
        "label", string=lambda text: text and text.strip().lower() == "taxes"
    )
    if not taxes_label:
        raise ValueError("Taxes element not found in HTML document")
    taxes_amount = None
    taxes_year = None

    for sibling in taxes_label.next_siblings:
        if isinstance(sibling, NavigableString):
            text = sibling.strip()
        else:
            text = (
                sibling.get_text(" ", strip=True)
                if hasattr(sibling, "get_text")
                else ""
            )
        text = _normalize_whitespace(text)
        if not text:
            continue
        if text.startswith("(") and text.endswith(")") and not taxes_year:
            taxes_year = text.strip("()").strip()
            continue
        if taxes_amount:
            taxes_amount = f"{taxes_amount} {text}"
        else:
            taxes_amount = text

    if not taxes_amount:
        raise ValueError("Taxes amount not found in HTML document")
    taxes_amount = _normalize_whitespace(taxes_amount)

    address_container = (
        soup.select_one("div.container.listing-full div.addr")
        or soup.select_one(".listing-full .addr")
        or soup.find(class_=lambda cls: cls and "addr" in cls)
    )
    address_text = ""
    house_type = ""
    if address_container:
        address_parts: list[str] = []
        h1 = address_container.find("h1")
        if h1:
            address_parts.append(_normalize_whitespace(h1.get_text(" ", strip=True)))
        h3 = address_container.find("h3")
        if h3:
            address_parts.append(_normalize_whitespace(h3.get_text(" ", strip=True)))
        if not address_parts:
            address_parts.append(
                _normalize_whitespace(address_container.get_text(" ", strip=True))
            )
        address_text = _normalize_whitespace(
            " ".join(part for part in address_parts if part)
        )
        h2 = address_container.find("h2")
        if h2:
            house_type = _normalize_whitespace(h2.get_text(" ", strip=True))

    description_node = (
        soup.select_one(".description.readmore")
        or soup.select_one(".description")
        or _find_description_near_title(soup)
    )
    description_text = (
        _normalize_whitespace(description_node.get_text(" ", strip=True))
        if description_node
        else ""
    )

    return {
        "price": price_text,
        "taxes": taxes_amount,
        "tax_year": taxes_year,
        "address": address_text,
        "house_type": house_type,
        "description": description_text,
    }


def _find_description_near_title(soup: BeautifulSoup):
    title_node = soup.find(id="description") or soup.find(
        string=lambda text: text and text.strip().lower() == "description"
    )
    if not title_node:
        return None
    node = title_node if hasattr(title_node, "next_siblings") else title_node.parent
    if not node:
        return None
    for sibling in node.next_siblings:
        if not hasattr(sibling, "get_text"):
            continue
        classes = [cls.lower() for cls in sibling.get("class", [])]
        if any("description" in cls for cls in classes):
            return sibling
        text = _normalize_whitespace(sibling.get_text(" ", strip=True))
        if text:
            return sibling
    return None


def extract_price(html: str) -> str:
    """Return the listing price text from a TRREB listing page."""
    return _extract_details(html)["price"]


def extract_taxes(html: str) -> str:
    """Return the taxes text from a TRREB listing page."""
    return _extract_details(html)["taxes"]


def extract_tax_year(html: str) -> str | None:
    """Return the taxes year from a TRREB listing page."""
    return _extract_details(html)["tax_year"]


def extract_address(html: str) -> str:
    """Return the full address string from a TRREB listing page."""
    return _extract_details(html)["address"]


def extract_description(html: str) -> str:
    """Return the public description from a TRREB listing page."""
    return _extract_details(html)["description"]


def extract_house_type(html: str) -> str:
    """Return the property house type from a TRREB listing page."""
    return _extract_details(html)["house_type"]


def extract_price_from_file(path: Path) -> str:
    """Load HTML from disk and extract the listing price."""
    html = path.read_text(encoding="utf-8")
    return extract_price(html)


def extract_taxes_from_file(path: Path) -> str:
    """Load HTML from disk and extract the listing taxes."""
    html = path.read_text(encoding="utf-8")
    return extract_taxes(html)


def extract_tax_year_from_file(path: Path) -> str | None:
    """Load HTML from disk and extract the taxes year."""
    html = path.read_text(encoding="utf-8")
    return extract_tax_year(html)


def extract_address_from_file(path: Path) -> str:
    """Load HTML from disk and extract the listing address."""
    html = path.read_text(encoding="utf-8")
    return extract_address(html)


def extract_description_from_file(path: Path) -> str:
    """Load HTML from disk and extract the listing description."""
    html = path.read_text(encoding="utf-8")
    return extract_description(html)


def extract_house_type_from_file(path: Path) -> str:
    """Load HTML from disk and extract the house type."""
    html = path.read_text(encoding="utf-8")
    return extract_house_type(html)


if __name__ == "__main__":
    target_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).with_name("page_source.html")
    )
    html_text = target_path.read_text(encoding="utf-8")
    details = _extract_details(html_text)
    print("Price: ", details["price"])
    taxes_line = details["taxes"]
    if details.get("tax_year"):
        taxes_line = f"{taxes_line} ({details['tax_year']})"
    print("Taxes: ", taxes_line)
    if details.get("address"):
        print("Address: ", details["address"])
    if details.get("house_type"):
        print("House Type: ", details["house_type"])
    if details.get("description"):
        print("Description: ", details["description"])
