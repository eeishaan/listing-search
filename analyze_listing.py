import os
import json
import requests

# import google.generativeai as genai
from PIL import Image
from io import BytesIO
import argparse
from google.genai import types
from google import genai
from pathlib import Path


def analyze_listing(json_path, api_key=None):
    # Load listing details
    print(f"Loading data from {json_path}...")
    if isinstance(json_path, (Path, str)):
        data = json.loads(Path(json_path).read_text())
    elif isinstance(json_path, dict):
        data = json_path
    else:
        raise ValueError(f"Invalid JSON path: {json_path}")

    image_urls = data.get("images", [])
    if not image_urls:
        print("No images found in the listing details.")
        return

    # Prepare images for Gemini
    images = [
        types.Part.from_bytes(data=requests.get(url).content, mime_type="image/jpeg")
        for url in image_urls
    ]
    print(f"Found {len(image_urls)} images.")

    if not images:
        print("No images could be downloaded.")
        return

    # Configure Gemini
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print(
            "GOOGLE_API_KEY not found. Please set it in environment or pass via --api-key."
        )
        return

    model = genai.Client(api_key=api_key)

    prompt = """
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

    try:
        response = model.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, *images],
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini: {e}")


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
