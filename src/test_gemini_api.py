import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# ============================================================
# CHECK API KEY
# ============================================================

print("=" * 60)
print("GEMINI API CONNECTION TEST")
print("=" * 60)

if not api_key:
    print("ERROR: GEMINI_API_KEY was not found.")
    raise SystemExit(1)

print("API key loaded successfully.")


# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# SEND TEST REQUEST
# ============================================================

print("\nSending test request to Gemini...")

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Say hello and briefly confirm that you are working."
)


# ============================================================
# DISPLAY RESPONSE
# ============================================================

print("\nGemini response:")
print(interaction.output_text)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("GEMINI API TEST COMPLETE")
print("=" * 60)