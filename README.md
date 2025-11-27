# Toronto Property Search Agent

An AI-powered real estate assistant that helps users find and analyze properties in Toronto. It integrates with the Toronto Regional Real Estate Board (TRREB) search, extracts listing details using Playwright, and analyzes property images and data using Google's Gemini AI.

🎥 : [Click here for demo video](https://youtu.be/BnkGwZdtO0M)

## Features

-   **Intelligent Search**: Automates property searches on TRREB based on natural language criteria.
-   **Deep Extraction**: Scrapes detailed property information including images and descriptions.
-   **AI Analysis**: Uses Google's **Gemini 2.5 Flash** to analyze property images, assess value, and answer specific questions about the listing.
-   **Chat Interface**: A clean, Flask-based web interface to interact with the agent conversationally.
-   **Termina Interface** : For easy terminal testing.

## Prerequisites

-   Python 3.11+
-   [Playwright](https://playwright.dev/) browsers
-   A Gemini API key.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/listing-search.git
    cd listing-search
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv listing_env
    source listing_env/bin/activate 
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright browsers:**

    ```bash
    playwright install
    ```

## Configuration

1.  Create a `.env` file in the root directory:

    ```bash
    cp example.env .env
    ```

2.  Add your API keys and configuration:

    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

## Usage

1.  **Start the web application:**

    ```bash
    python app.py
    ```

2.  **Access the interface:**
    Open your browser and navigate to `http://localhost:2000` (or the port displayed in the terminal).

3.  **Interact with the agent:**
    -   Ask to find properties in a specific area (e.g., "Find 2-bedroom condos in Toronto under $800k").
    -   The agent will search TRREB, show you results, and you can ask it to analyze specific listings.

4. **Terminal Usage**
    - Instead of a web interface, use `python property_agent.py` to start a terminal chat.
    - This is helpful during debugging.

## Limitations
1. This project is mostly for fun and personal use.
1. Property search is limited to Toronto area.
1. Due to a bug in TRREB's search interface, searching for "leases" is not supported by the current code.
1. The top 5 listings selected are not semantically "top 5". They follow the order selected by TRREB search; which as of writing is ascending order of price.
1. Not all filters present in TRREB search are supported in the search listing tool. It's trivial to add this support. Simply modify the tool function signature.
1. I've extensively used AI assistance in building this tool. It should be quite apparent when going through the code :) 


## Project Structure

-   `app.py`: Flask web application entry point.
-   `property_agent.py`: Core agent logic using Google GenAI.
-   `trreb_search_tool.py`: Playwright script for searching TRREB.
-   `analyze_listing.py`: Tool for analyzing listing images and details with Gemini.
-   `fetch_and_extract_listing.py`: Scraper for extracting listing details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

