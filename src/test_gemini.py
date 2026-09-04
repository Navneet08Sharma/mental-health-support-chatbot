import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

print("=" * 60)
print("GEMINI API KEY TEST")
print("=" * 60)

if api_key:
    print("Gemini API key loaded successfully.")
    print(f"Key starts with: {api_key[:6]}...")
else:
    print("ERROR: GEMINI_API_KEY was not found.")

print("=" * 60)