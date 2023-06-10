import requests
import requests

API_URL = "http://localhost:3000/api/v1/prediction/7df3e69c-9d10-4e58-9302-fd64ed28eb96"


def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.text


while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    output = query({
        "question": user_input,
    })

    print("Chatbot: ", output)
