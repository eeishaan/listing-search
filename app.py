import os
import logging
import json
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    Response,
    stream_with_context,
)
from property_agent import PropertyAgent, TOOL_MAP
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebApp")

app = Flask(__name__)
# In a real app, use a secret key from env
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# Initialize Agent
# We'll use a single global agent instance for simplicity in this demo.
# In a multi-user production app, we might want to instantiate per request or manage sessions differently.
# However, the Agent class in property_agent.py is mostly a configuration holder + client.
# The state (history) is what matters.
try:
    agent = PropertyAgent()
    logger.info("Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    agent = None

# Store conversation history in memory for simplicity (Session ID -> History List)
# In production, use a database or Redis.
conversations = {}


@app.route("/")
def home():
    # Initialize session if needed
    if "session_id" not in session:
        session["session_id"] = os.urandom(8).hex()

    sid = session["session_id"]
    if sid not in conversations:
        conversations[sid] = []

    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not agent:
        return jsonify({"error": "Agent not initialized"}), 500

    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    sid = session.get("session_id")
    if not sid or sid not in conversations:
        if not sid:
            sid = os.urandom(8).hex()
            session["session_id"] = sid
        conversations[sid] = []

    history = conversations[sid]
    history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    def generate():
        try:
            while True:
                logger.info(f"Calling model with {len(history)} messages (stream=True)")
                stream = agent.client.models.generate_content_stream(
                    model=agent.model_name,
                    contents=history,
                    config=agent.config,
                )

                full_text = ""
                tool_call = None

                for chunk in stream:
                    # Handle text chunks
                    try:
                        if chunk.text:
                            text_part = chunk.text
                            full_text += text_part
                            yield json.dumps(
                                {"type": "text", "content": text_part}
                            ) + "\n"
                    except Exception:
                        pass

                    # Check for tool calls in the chunk
                    if chunk.candidates:
                        for part in chunk.candidates[0].content.parts:
                            if part.function_call:
                                tool_call = part.function_call
                                break
                    if tool_call:
                        break

                # Update history with the assistant's turn
                parts = []
                if full_text:
                    parts.append(types.Part(text=full_text))
                if tool_call:
                    parts.append(types.Part(function_call=tool_call))

                if parts:
                    history.append(types.Content(role="model", parts=parts))

                if not tool_call:
                    yield json.dumps({"type": "done"}) + "\n"
                    break

                # Handle Tool Call
                tool_name = tool_call.name
                logger.info(f"Tool call detected: {tool_name}")

                # User friendly notification
                friendly_msg = "Processing..."
                if "search" in tool_name.lower():
                    friendly_msg = "Searching for properties..."
                elif "detail" in tool_name.lower():
                    friendly_msg = "Fetching listing details..."
                elif "analyze" in tool_name.lower():
                    friendly_msg = "Analyzing property images..."

                yield json.dumps({"type": "info", "content": friendly_msg}) + "\n"

                tool = TOOL_MAP.get(tool_name)
                if tool:
                    try:
                        args = dict(tool_call.args)
                        result = tool(**args)

                        # Add result to history
                        function_response_part = types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result},
                        )
                        history.append(
                            types.Content(role="user", parts=[function_response_part])
                        )

                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        history.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_function_response(
                                        name=tool_name, response={"error": str(e)}
                                    )
                                ],
                            )
                        )
                else:
                    logger.error(f"Tool {tool_name} not found")
                    history.append(
                        types.Content(
                            role="user",
                            parts=[types.Part(text=f"Tool {tool_name} not found")],
                        )
                    )

        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return Response(
        stream_with_context(generate()), content_type="application/x-ndjson"
    )


@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get("session_id")
    if sid and sid in conversations:
        conversations[sid] = []
    return jsonify({"status": "reset"})


import requests
from bs4 import BeautifulSoup


@app.route("/preview", methods=["GET"])
def preview():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "Missing URL"}), 400

    try:
        # User-Agent header is important for some sites
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")

        # Helper to get meta tag content
        def get_meta(property_name):
            tag = soup.find("meta", property=property_name) or soup.find(
                "meta", attrs={"name": property_name}
            )
            return tag["content"] if tag else None

        title = get_meta("og:title") or soup.title.string if soup.title else url
        description = get_meta("og:description") or get_meta("description")
        image = get_meta("og:image")

        # If no OG image, try to find the first relevant image
        if not image:
            # Basic heuristic: find first large image or logo
            pass

        return jsonify(
            {"title": title, "description": description, "image": image, "url": url}
        )
    except Exception as e:
        logger.error(f"Error fetching preview for {url}: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 2000))
    app.run(host="0.0.0.0", port=port, debug=True)
