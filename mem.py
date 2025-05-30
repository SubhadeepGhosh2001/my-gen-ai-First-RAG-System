from mem0 import Memory
import google.generativeai as genai
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Config
GEMINI_API_KEY = "AIzaSyDE-eVV3V-YRskGgYCO55aYTMax-ygOY6k"
QUADRANT_HOST = "localhost"

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "aFMyglU-YvGNg4mnK1zJrEMJGpV1-8bbEKdnJXs5d_c"

# Setup memory config
config = {
    "version": "v1.1",
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    },
    "llm": {
        "provider": "gemini",
        "config": {
        "api_key": GEMINI_API_KEY,
        "model": "gemini-2.0-flash",
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QUADRANT_HOST,
            "port": 6333,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD
        },
    },
}

# Memory client and Gemini setup
mem_client = Memory.from_config(config)
genai.configure(api_key=GEMINI_API_KEY)

# Gemini via OpenAI-compatible SDK
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Chat function
def chat(message):
    mem_result = mem_client.search(query=message, user_id="p123")

    memories = "\n".join([m["memory"] for m in mem_result.get("results")])

    print(f"\n🔍 MEMORY:\n{memories}\n")

    SYSTEM_PROMPT = f"""
    You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
    systematically analyze input content, extract structured knowledge, and maintain an
    optimized memory store. Your primary function is information distillation
    and knowledge preservation with contextual awareness.

    Memory and Score:
    {memories}
    """

    # Building the messages list incrementally
    messages = [
         { "role": "system", "content": SYSTEM_PROMPT },
         { "role": "user", "content": message }
    ]
    # messages.append({"role": "system", "content": SYSTEM_PROMPT})
    # messages.append({"role": "user", "content": message})

    # Generate assistant response
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages
    )

    assistant_reply = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": assistant_reply})

    # Store messages to memory
    mem_client.add(messages, user_id="p123")

    return assistant_reply

# Run the agent
if __name__ == "__main__":
    while True:
        message = input(">> ")
        print("\n🧠 BOT:", chat(message))
