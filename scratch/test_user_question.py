import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def get_token():
    res = requests.post(f"{BASE_URL}/auth/login", data={"username": "testuser", "password": "password123"})
    if res.status_code == 200:
        return res.json()["access_token"]
    res = requests.post(f"{BASE_URL}/auth/register", json={
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "password123",
        "full_name": "Test User"
    })
    res = requests.post(f"{BASE_URL}/auth/login", data={"username": "testuser", "password": "password123"})
    return res.json()["access_token"]

token = get_token()
headers = {"Authorization": f"Bearer {token}"}

questions = [
    "what problem does it address?",
    "what methodology does this paper propose?",
    "what are the quantitative results?",
    "what is the main objective of this paper?"
]

for q in questions:
    print(f"\n==========================================")
    print(f"QUESTION: '{q}'")
    print(f"==========================================")
    payload = {
        "question": q,
        "paper_id": "AI003",
        "domain": "artificial_intelligence"
    }
    res = requests.post(f"{BASE_URL}/research/query", json=payload, headers=headers, timeout=20)
    if res.status_code == 200:
        data = res.json()
        print("ANSWER:\n", data.get("answer"))
        print("CITATIONS:", list(data.get("citations", {}).keys()))
    else:
        print("ERROR:", res.status_code, res.text)
