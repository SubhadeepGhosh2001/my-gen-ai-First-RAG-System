import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mem0 import Memory
import google.generativeai as genai
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Config (same as your original setup)
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

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to split text into chunks using LangChain
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to process and store PDF chunks in mem0
def process_pdf(file_path):
    # Extract text
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(file_path)

    # Split into chunks
    print("Splitting text into chunks...")
    chunks = split_text(text)

    # Store chunks in mem0
    print("Storing chunks in memory...")
    for i, chunk in enumerate(chunks):
        mem_client.add(chunk, user_id="pdf_chunks")
        print(f"Stored chunk {i+1}/{len(chunks)}")
    print("PDF processing complete!")

# Function to retrieve and query chunks
def query_pdf(message):
    # Search for relevant chunks in mem0
    mem_result = mem_client.search(query=message, user_id="pdf_chunks")

    memories = "\n".join([m["memory"] for m in mem_result.get("results")])

    print(f"\n🔍 RETRIEVED CHUNKS:\n{memories}\n")

    SYSTEM_PROMPT = f"""
    You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
    systematically analyze input content, extract structured knowledge, and maintain an
    optimized memory store. Your primary function is information distillation
    and knowledge preservation with contextual awareness.

    Retrieved Chunks:
    {memories}
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]

    # Generate assistant response
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages
    )

    assistant_reply = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": assistant_reply})

    # Store the conversation in mem0
    mem_client.add(messages, user_id="p123")

    return assistant_reply

# Main function to process PDF and query
if __name__ == "__main__":
    # Process the PDF
    file_path = input("Enter the path to your PDF file: ")
    process_pdf(file_path)

    # Query loop
    while True:
        message = input(">> Enter a query to search the PDF content (or 'exit' to quit): ")
        if message.lower() == "exit":
            break
        print("\n🧠 BOT:", query_pdf(message))