import requests

# URL of your running FastAPI server
url = "http://127.0.0.1:8000/predict"

# Path to your test image
file_path = "0b37761a-de32-47ee-a3a4-e138b97ef542___JR_FrgE.S 2908_new30degFlipLR.JPG"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())
