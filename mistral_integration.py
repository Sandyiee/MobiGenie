import requests
import os
from dotenv import load_dotenv

# Load the HF token from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class MistralChatbot:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        self.headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

    def ask_question(self, query):
        # Hugging Face instruct models need special formatting
        formatted_query = f"<s>[INST] {query} [/INST]"

        payload = {
            "inputs": formatted_query,
            "parameters": {"temperature": 0.7, "max_new_tokens": 300}
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            output = response.json()
            # Extract and clean response
            generated_text = output[0]["generated_text"].replace(formatted_query, "").strip()
            return generated_text
        else:
            return f"Error {response.status_code}: {response.text}"
