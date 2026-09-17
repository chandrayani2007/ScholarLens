"""
Test script to send the exact question from user screenshot to the live backend API.
Question: "what are the benefits of the quantum computing?"
"""

import urllib.request
import json

url = "http://127.0.0.1:8000/research/query"
payload = {
    "question": "what are the benefits of the quantum computing?",
    "top_k": 5
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("\n" + "=" * 80)
        print("LIVE API RESPONSE FOR USER QUESTION:")
        print("Question:", data.get("question"))
        print("Confidence:", data.get("confidence"))
        print("Citations:", list(data.get("citations", {}).keys()))
        print("\n[ANSWER]:\n", data.get("answer"))
        print("=" * 80)
except Exception as e:
    print("Error querying live API:", e)
