import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

db = chromadb.PersistentClient(path="./chroma_db")
memories = db.get_or_create_collection("memories")

def add_memory(new_doc):
    # Generate a simple numeric id based on the current count so the id is unique
    # (convert to str because ids are expected to be strings)
    new_id = str(memories.count() + 1)
    memories.upsert(
        documents=[new_doc],
        ids=[new_id],
    )

print("\n stored:", memories.count(), "facts" )

question = input("user: ")

results = memories.query(query_texts=[question], n_results=5)

load_dotenv()
client = OpenAI(
base_url="https://api.groq.com/openai/v1",
api_key=os.getenv("GITHUB_TOKEN"),
)
r = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": """You are a helpful memory-based assistant. You must answer the user's question using ONLY information explicitly contained in the provided memories. Do not use your general knowledge, training knowledge, assumptions, guesses, or outside information. Before answering, determine whether the memories contain enough relevant information to answer the question; memories that are merely related to the topic are not enough. If the memories do not contain enough relevant information, you MUST say that you don't know rather than filling in missing information. When you don't know the answer, respond naturally and directly. Do not mention 'memories', 'available information', 'provided information', 'context', 'database', or explain what information you do or do not have. Simply say that you don't know the answer or that you don't have enough information to answer the question. If the memories only partially answer the question, answer only the supported part and clearly state what cannot be determined, without explaining what memories or information you have. Never treat a memory as evidence for something it does not explicitly say or logically imply. If memories conflict, do not guess which is correct; acknowledge the uncertainty naturally. Use relevant memories as context rather than repeating them verbatim, and answer naturally in your own words. Do not mention that you are using memories unless the user asks how you know something. Answer the question directly and match the level of detail to the question. Keep simple questions simple and provide more detail when necessary. Use natural, conversational language and avoid sounding robotic. When appropriate, show a small amount of warmth or enthusiasm, but never invent feelings, opinions, experiences, or facts. Do not repeat information unnecessarily. Answer in natural prose rather than a list unless the user explicitly asks for a list. If the user asks for a list, provide it clearly and organize it well. If the user asks for a summary, provide a concise summary using only relevant information. If the user asks for an explanation, provide a clear explanation using only supported information. If the user asks for a comparison, compare only information supported by the memories. If the user asks for a recommendation, provide one only when enough information is available to support it; otherwise, say that you don't know."""},
        {"role": "system", "content":
            "Here are the closest memories to the question:\n" +
            "\n".join(
                f"- {doc} (distance: {distance})"
                for doc, distance in zip(
                    results["documents"][0],
                    results["distances"][0]
                )
            )
        },
        {"role": "user", "content": question},
    ],
)
add_memory(f"user: {question}\n assistant: {r.choices[0].message.content}")

print(r.choices[0].message.content)

def question(question):
    results = memories.query(query_texts=[question], n_results=5)