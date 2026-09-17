import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

# Log in to get token
login_res = requests.post("http://127.0.0.1:8000/auth/login", data={"username": "test@scholar.lens", "password": "pw"})
print("Login status:", login_res.status_code)
token = None
if login_res.status_code == 200:
    token = login_res.json().get("access_token")
else:
    # Try registering or get any user
    reg_res = requests.post("http://127.0.0.1:8000/auth/register", json={"email": "live_user@scholar.lens", "password": "SecurePassword123!", "full_name": "Live User"})
    print("Register status:", reg_res.status_code)
    login_res2 = requests.post("http://127.0.0.1:8000/auth/login", data={"username": "live_user@scholar.lens", "password": "SecurePassword123!"})
    token = login_res2.json().get("access_token")

print("Token obtained:", bool(token))

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
payload = {
    "question": "What research problem does this paper address?",
    "paper_id": "AI038",
    "domain": "artificial_intelligence",
    "top_k": 10
}

print("Sending query to http://127.0.0.1:8000/research/query ...")
try:
    res = requests.post("http://127.0.0.1:8000/research/query", json=payload, headers=headers, timeout=60)
    print("Query status:", res.status_code)
    print("Query response:\n", res.json().get("answer"))
except Exception as e:
    print("Query failed with exception:", type(e).__name__, e)
