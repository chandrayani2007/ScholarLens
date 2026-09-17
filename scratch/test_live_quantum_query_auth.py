"""
Test script with OAuth2 form-data login for /research/query.
"""

import urllib.request
import urllib.parse
import json

base_url = "http://127.0.0.1:8000"

login_url = f"{base_url}/auth/login"
reg_url = f"{base_url}/auth/register"

user_data = {
    "username": "testuser_debug",
    "email": "testuser_debug@example.com",
    "password": "Password123!",
    "full_name": "Debug User"
}

# Try register
try:
    req_reg = urllib.request.Request(reg_url, data=json.dumps(user_data).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req_reg)
except Exception as e:
    pass

# Login with form data
login_data = urllib.parse.urlencode({
    "username": "testuser_debug@example.com",
    "password": "Password123!"
}).encode("utf-8")

req_login = urllib.request.Request(
    login_url,
    data=login_data,
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    method="POST"
)

with urllib.request.urlopen(req_login) as resp:
    token_data = json.loads(resp.read().decode("utf-8"))
    token = token_data.get("access_token")

print(f"Obtained JWT token successfully: {token[:20]}...")

# Execute Query
query_url = f"{base_url}/research/query"
payload = {
    "question": "what are the benefits of quantum computing?",
    "top_k": 5
}

req_q = urllib.request.Request(
    query_url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    },
    method="POST"
)

with urllib.request.urlopen(req_q) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print("\n" + "=" * 80)
    print("LIVE API RESPONSE FOR USER QUESTION:")
    print("Question:", res.get("question"))
    print("Confidence:", res.get("confidence"))
    print("Citations:", list(res.get("citations", {}).keys()))
    print("\n[ANSWER]:\n", res.get("answer"))
    print("=" * 80)
