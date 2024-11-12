import base64
import requests

# Read the photo and encode it as base64
with open("example_inputs/rihanna.webp", "rb") as photo_file:
    photo_base64 = base64.b64encode(photo_file.read()).decode("utf-8")

# Define the request payload
payload = {
    "email": "konenkovma@gmail.com",
    "firstName": "Mikhail",
    "photoStyle": "Street",
    "role": "Engineer",
    "purpose": "LinkedIn Profile",
    "ageRange": "30-40",
    "photo": photo_base64
}

# Send the request
response = requests.post("https://6be8-185-242-87-183.ngrok-free.app/api/photo-transform", json=payload)

# Print response
print(response.json())