import os
import sys
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

import google.generativeai as genai
genai.configure(api_key=api_key)

models = ["gemini-3.6-flash", "gemini-flash-latest", "gemini-3.7-flash", "gemini-2.5-pro"]
for m in models:
    try:
        print(f"\nTesting model: {m}")
        model_inst = genai.GenerativeModel(model_name=m)
        resp = model_inst.generate_content("Synthesize in one sentence what RAG is.")
        print(f"SUCCESS with {m}!")
        print(f"Response: {resp.text.strip()}")
        break
    except Exception as e:
        print(f"FAILED with {m}: {type(e).__name__}: {e}")
