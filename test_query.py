import requests

url = "http://127.0.0.1:5000/query"
headers = {"Content-Type": "application/json"}
data = {"user_input": "What is the role of the state in a mixed economy?"}

response = requests.post(url, headers=headers, json=data)
print("Response:", response.json())