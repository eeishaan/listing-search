import os
import logging
from flask import Flask, render_template, request, jsonify, session
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
        # Restore session if lost or new
        if not sid:
            sid = os.urandom(8).hex()
            session["session_id"] = sid
        conversations[sid] = []

    history = conversations[sid]

    # Add user message to history
    history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    # Process through the agent loop
    final_response_text = "I'm sorry, I couldn't generate a response."

    try:
        # Loop to handle tool calls
        while True:
            logger.info(f"Calling model with {len(history)} messages")
            response = agent.client.models.generate_content(
                model=agent.model_name,
                contents=history,
                config=agent.config,
            )

            # Check for malformed calls
            if response.candidates[0].finish_reason == "MALFORMED_FUNCTION_CALL":
                logger.error("Malformed function call")
                final_response_text = "I encountered an error with the function call."
                break

            # Append response to history
            history.append(response.candidates[0].content)

            # Check for tool call
            parts = response.candidates[0].content.parts
            tool_call = None
            if parts:
                for part in parts:
                    if part.function_call:
                        tool_call = part.function_call
                        break

            if tool_call:
                tool_name = tool_call.name
                logger.info(f"Tool call: {tool_name}")

                tool = TOOL_MAP.get(tool_name)
                if tool:
                    try:
                        args = dict(tool_call.args)
                        # Execute tool
                        result = tool(**args)

                        # Create function response
                        function_response_part = types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result},
                        )
                        history.append(
                            types.Content(role="user", parts=[function_response_part])
                        )
                        # Loop continues to send tool output back to model
                        continue

                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        # Feed error back to model
                        history.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part(
                                        text=f"Error executing tool {tool_name}: {e}"
                                    )
                                ],
                            )
                        )
                        continue
                else:
                    logger.error(f"Tool {tool_name} not found")
                    history.append(
                        types.Content(
                            role="user",
                            parts=[types.Part(text=f"Tool {tool_name} not found")],
                        )
                    )
                    continue

            # If no tool call, it's a text response (or empty)
            if response.text:
                final_response_text = response.text

            break

    except Exception as e:
        logger.error(f"Error in chat loop: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    return jsonify({"response": final_response_text})


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
