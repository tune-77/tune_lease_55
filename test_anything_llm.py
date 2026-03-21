import os
import requests

url = "http://localhost:3001/api/v1/workspace/71b18af2-ab59-4bcb-874c-4cb329bd1b41/chat"
headers = {
    "Authorization": "Bearer RGK6FNE-HHK4VTT-NJVQS1S-CK2847K",
    "Content-Type": "application/json"
}
data = {
    "message": "こんにちは",
    "mode": "chat"
}
try:
    response = requests.post(url, headers=headers, json=data, timeout=30)
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        print("Response:", response.json().get('textResponse', '')[:100])
    else:
        print("Error:", response.text[:100])
except Exception as e:
    print("Exception:", e)
