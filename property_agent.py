#!/usr/bin/env python3
"""
Property Search Agent

An AI agent that helps users find and analyze real estate properties using:
1. TRREB Search Tool (for finding listings)
2. Listing Extractor (for getting details)
3. Listing Analyzer (for analyzing images with Gemini)
"""

import os
import sys
import asyncio
import json
from typing import List, Optional, Dict, Any, Union
import time
from google import genai
from google.genai import types
from pydantic import BaseModel

# Import tools
from trreb_search_tool import (
    search_listings,
    TREBSearchParams,
)
from fetch_and_extract_listing import extract_listing_details, ListingExtractionParams
from analyze_listing import analyze_listing_tool, AnalyzeListingParams
from google.genai import errors

# =============================================================================
# Tool Wrappers
# =============================================================================


def search_properties(
    location: str | None = None,
    listing_type: str | None = None,
    price_min: int | None = None,
    price_max: int | None = None,
    bedrooms: List[str] | None = None,
    bathrooms: List[str] | None = None,
    property_categories: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Search for real estate listings on TRREB (Toronto Regional Real Estate Board).

    Args:
        location: City, postal code, address, or MLS# to search for. Pass None if not specified.
        listing_type: 'sale' or 'lease'. Pass None to default to 'sale'.
        price_min: Minimum price in dollars. Pass None if no minimum.
        price_max: Maximum price in dollars. Pass None if no maximum.
        bedrooms: Number of bedrooms to filter by (e.g. ["1", "2", "3"]). Pass None if any.
        bathrooms: Number of bathrooms to filter by (e.g. ["1", "2"]). Pass None if any.
        property_categories: Categories like 'freehold', 'condo', 'commercial'. Pass None if any.

    Returns:
        A dictionary containing the search results (count and list of listings).
    """
    print(f"\n[Tool] Searching listings in {location}...")

    # Handle defaults internally since Gemini API tool schema doesn't support defaults
    final_listing_type = listing_type if listing_type else "sale"

    try:
        bedrooms = [x if int(x) < 5 else "5+" for x in bedrooms]
        bathrooms = [x if int(x) < 4 else "4+" for x in bathrooms]
    except:
        pass

    params = TREBSearchParams(
        location=location,
        listing_type=final_listing_type,
        price_min=price_min,
        price_max=price_max,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        property_categories=property_categories,  # type: ignore
    )

    # Run the search
    results = search_listings(params)

    # Convert to dict for the model
    res = results.model_dump()

    return res


def get_listing_details(url: str, image_analysis_prompt: str | None) -> Dict[str, Any]:
    """
    Extract detailed information from a specific listing URL.
    Use this after finding a listing in the search results to get more info like image analysis and description.

    Args:
        url: The full URL of the listing.
        image_analysis_prompt: Optional specific question or focus for the analysis. Pass None for general analysis.

    Returns:
        A dictionary with listing details (price, address, description, image_urls).
    """
    print(f"\n[Tool] Extracting details from {url}...")
    # Run extraction
    details = extract_listing_details(url)

    image_analysis = analyze_listing_images(
        url, details["images"], image_analysis_prompt
    )
    details["image_analysis"] = image_analysis
    return details


def analyze_listing_images(
    listing_url: str, image_urls: List[str], prompt: str | None
) -> str:
    """
    Analyze images of a property using AI to evaluate condition, layout, and lighting.

    Args:
        listing_url: Listing URL to which the images belong to.
        image_urls: List of image URLs to analyze.
        prompt: Optional specific question or focus for the analysis. Pass None for general analysis.

    Returns:
        A text analysis of the images.
    """
    print(f"\n[Tool] Analyzing {len(image_urls)} images...")
    params = AnalyzeListingParams(
        listing_url=listing_url,
        image_urls=image_urls,
        prompt=prompt,
    )

    result = analyze_listing_tool(params)
    return result.analysis


# =============================================================================
# Agent Class
# =============================================================================

TOOL_MAP = {
    "search_properties": search_properties,
    "get_listing_details": get_listing_details,
    "analyze_listing_images": analyze_listing_images,
}


class PropertyAgent:
    def __init__(self, model_name: str = "gemini-3.0-pro"):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            print("Warning: GOOGLE_API_KEY not set in environment.")
            self.api_key = input("Please enter your Google API Key: ").strip()
            if not self.api_key:
                raise ValueError("API Key is required to run this agent.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

        # Register tools
        self.tools = [
            search_properties,
            get_listing_details,
        ]
        # self.tools = [types.Tool(function_declarations=[SEARCH_TOOL_DEFINITION])]

        # Initial system prompt
        system_instruction = """
        You are an expert real estate agent assistant for the Toronto market.
        Your goal is to help the user find the perfect property.

        Follow this process:
        1.  Understand the User's Needs:
            - Ask questions to clarify location, budget, property type (condo/freehold), bedrooms, etc. if not provided.
            - Don't assume; ask if unsure.

        2.  Search and Filter:
            - Briefly state your search plan to the user (e.g., "I will search for 2-bedroom condos in Toronto under $800k").
            - Use the `search_properties` tool to find listings matching the criteria.
            - When user asks for something than the search tool can't provide, you'll have to creatively use other tools to find that information.
            - IF YOU HAVE TO USE OTHER TOOLS, DO NOT PRESENT A SUMMARY OF YOUR RESULTS JUST YET. GO TO NEXT STEP FOR THAT.

        3.  Deep Dive & Analyze:
            - For analysing a specific listing, use `get_listing_details` to fetch full details and image analysis.
            - Combine the listing details, such as description,and image analysis to give a comprehensive recommendation.
            - Paginate your recommendation to top 5 listings.
            - Structure of the final result should be like this:
              - Listing 1:
                - Address: 1234567890
                - Price: 1234567890
                - URL: https://www.example.com
              - Listing 2:
                ...

        4.  Refine:
            - Iterate based on user feedback.

        Always be professional, helpful, and concise.
        """

        self.config = types.GenerateContentConfig(
            tools=self.tools,
            system_instruction=system_instruction,
            temperature=0.5,  # Balance between creativity and precision
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            # thinking_config=types.ThinkingConfig(thinking_level="low"),
            thinkingConfig={
                "thinkingBudget": 1024,
            },
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="VALIDATED")
            ),
        )

    def start_chat(self):
        """Starts an interactive chat session with the user."""
        print("Property Search Agent initialized.")
        print("I can help you find and analyze properties in Toronto.")
        print("What are you looking for today? (Type 'quit' to exit)")

        # Start the chat session
        # Note: tools are passed as a list of callables.

        contents = []
        is_user_turn = True
        first_query = "I'm looking to buy a property with at least 3 bedrooms and 2 baths. There should be at least 1 parking space. It can be a townhouse, semi-detached, or detached house. My area of choice is st. andrew-york mills. The layout should be simple and inviting and the light in the house should be adequate. My budget range is 700k to 900k."
        while True:
            print("len of contents at entry: ", len(contents))

            if is_user_turn:
                user_input = input("\nYou: ") if first_query is None else first_query
                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break
                first_query = None
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=user_input)])
                )

            try:
                # Send message to the model
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=self.config,
                )
            except errors.ClientError as e:
                is_user_turn = False
                continue
            except Exception as e:
                print(f"\nError: {e}")
                import traceback

                traceback.print_exc()
                is_user_turn = False
                time.sleep(1)
                continue

            if response.candidates[0].finish_reason:
                if response.candidates[0].finish_reason == "MALFORMED_FUNCTION_CALL":
                    # remove both tool call and result.
                    contents.pop(-1)
                    contents.pop(-1)
                    continue

            contents.append(
                response.candidates[0].content
            )  # Append the content from the model's response.

            is_user_turn = True
            if response.text:
                print(f"\nAgent: {response.text}")

            tool_call = response.candidates[0].content.parts[-1].function_call

            if tool_call is None:
                continue

            tool_name = tool_call.name
            tool = TOOL_MAP.get(tool_name)
            if tool:
                print(f"calling tool {tool_name} with args: {tool_call.args}")
                result = tool(**tool_call.args)
                print(f"got result from tool {tool_name}: {result}")
                function_response_part = types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )
                contents.append(
                    types.Content(role="user", parts=[function_response_part])
                )  # Append the function response
                is_user_turn = False
            else:
                print(f"Tool {tool_name} not found")
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"Tool {tool_name} not found")],
                    )
                )


def main():
    # Allow model override via env var
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
    agent = PropertyAgent(model_name=model)
    agent.start_chat()


if __name__ == "__main__":
    main()
