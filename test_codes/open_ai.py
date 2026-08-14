import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
base_url="https://api.groq.com/openai/v1",
api_key=os.getenv("GITHUB_TOKEN"),
)
r = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful memory-based assistant."},
        {"role": "user", "content": question},
    ],
)

def question(question):
    return r.choices[0].message.content