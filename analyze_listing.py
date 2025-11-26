import os
import json
import requests
from typing import Optional, List, Union, Any
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from io import BytesIO
import argparse
from google.genai import types
from google import genai
from pydantic import BaseModel, Field

# =============================================================================
# Schemas
# =============================================================================


class AnalyzeListingParams(BaseModel):
    """Input parameters for the listing analysis tool."""

    listing_url: str = Field(description="Listing URL to which the images belong to")
    image_urls: List[str] = Field(description="List of URLs of images to analyze")
    prompt: Optional[str] = Field(
        default=None,
        description="Optional custom prompt for analysis. If not provided, a default real estate analysis prompt is used.",
    )


class AnalyzeListingResult(BaseModel):
    """Result of the listing analysis."""

    analysis: str = Field(description="The detailed analysis of the listing images")


# =============================================================================
# Core Logic
# =============================================================================

DEFAULT_PROMPT = """
Analyze these images of a house listing and answer the following questions:

- Is the house under construction?
- Does the house have high ceilings? Estimate the ceiling by looking at the gap between door frames and ceilings.
- Comment on the overall layout of the house. Is there enough separation between formal and family areas?
- Does the house have a lot of natural light?
- Try and estimate the number of skylights on the top level.
- How big and spacious are the rooms?
- Anything that is weird and concerning.

Provide a detailed summary answering these points.
"""


def _analyze_images(
    image_urls: List[str],
    api_key: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Core function to analyze images using Gemini."""
    if not image_urls:
        return "No images provided for analysis."

    # Prepare images for Gemini
    print(f"Downloading {len(image_urls)} images...")
    images = []
    for url in image_urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                images.append(
                    types.Part.from_bytes(data=resp.content, mime_type="image/jpeg")
                )
        except Exception as e:
            print(f"Failed to download image {url}: {e}")
            continue

    if not images:
        return "No images could be downloaded."

    print(f"Successfully prepared {len(images)} images for analysis.")

    # Configure Gemini
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please set it in environment or pass via --api-key."
        )

    client = genai.Client(api_key=api_key)

    analysis_prompt = prompt if prompt else DEFAULT_PROMPT

    generate_content_config = types.GenerateContentConfig(
        thinkingConfig={
            "thinkingBudget": 0,
        },
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[analysis_prompt, *images],
            config=generate_content_config,
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return f"Error generating analysis: {e}"


def analyze_listing(
    json_path: Union[str, Path, dict], api_key: Optional[str] = None
) -> Optional[str]:
    """Legacy wrapper for backward compatibility."""
    # Load listing details
    print(f"Loading data from {json_path}...")
    if isinstance(json_path, (Path, str)):
        try:
            data = json.loads(Path(json_path).read_text())
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return None
    elif isinstance(json_path, dict):
        data = json_path
    else:
        raise ValueError(f"Invalid JSON path: {json_path}")

    image_urls = data.get("images", [])
    if not image_urls:
        print("No images found in the listing details.")
        return None

    return _analyze_images(image_urls, api_key)


class Cache:
    def __init__(self, cache_dir: Path = "image_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, url: str) -> Dict[str, Any]:
        cache_file = self.cache_dir / f"{hash(url)}.json"
        if cache_file.exists():
            print("cache hit")
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return None

    def set(self, url: str, details: Dict[str, Any]):
        cache_file = self.cache_dir / f"{hash(url)}.json"
        cache_file.write_text(json.dumps(details, indent=2), encoding="utf-8")


GLOBAL_CACHE = Cache()


def analyze_listing_tool(params: AnalyzeListingParams) -> AnalyzeListingResult:
    """Tool function to be called by the LLM agent."""
    key = params.listing_url.split("/")[-1].strip() + (
        params.prompt if params.prompt else ""
    )
    result = GLOBAL_CACHE.get(key)
    if result:
        return AnalyzeListingResult(analysis=result["analysis"])
    analysis = _analyze_images(
        params.image_urls,
        prompt=params.prompt,
    )
    result = AnalyzeListingResult(analysis=analysis)
    GLOBAL_CACHE.set(key, result.model_dump())
    return result


# =============================================================================
# Tool Definition
# =============================================================================

TOOL_DEFINITION = {
    "name": "analyze_listing_images",
    "description": (
        "Analyze images of a real estate listing to evaluate condition, layout, "
        "lighting, and identify any concerns. Accepts a list of image URLs."
    ),
    "input_schema": AnalyzeListingParams.model_json_schema(),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze listing images with Gemini.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="data/listing_details.json",
        help="Path to listing details JSON",
    )
    parser.add_argument("--api-key", help="Google API Key")
    args = parser.parse_args()

    analyze_listing(args.json_path, args.api_key)
