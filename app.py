import streamlit as st
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
base_url="https://api.groq.com/openai/v1",
api_key=os.getenv("GITHUB_TOKEN"),
)


def question(question):
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=creativity,
        messages=[
            {"role": "system", "content": "You are a helpful memory-based assistant."},
            {"role": "system", "content": f"use this for response lenght: {response_length}"},
            {"role": "system", "content": f"this is your response style: {style}"},
            {"role": "system", "content": f"this is your response units: {units}"},
            {"role": "system", "content": f"this is your response engineering explanation level: {explanation_level}"},
            {"role": "system", "content": f"this is your cost priority out of 1.0: {cost_priority}"},
            {"role": "system", "content": f"this is your performance priority out of 1.0: {performance_priority}"},
            {"role": "system", "content": f"this is your reliability priority out of 1.0: {reliability_priority}"},
            {"role": "system", "content": f"this is your safety priority out of 1.0: {safety_priority}"},
            {"role": "user", "content": question},
        ],
    )
    return r.choices[0].message.content


def submit():
    st.session_state.x += 1


st.title("RAG-Based AI")
with st.sidebar:
    st.subheader("Settings")
    response_length = st.radio("How are you want the ai to respond?", ("Talkative", "Balanced", "Concise"))
    style = st.radio("What style do you want the ai to respond in?", ("Professional", "Casual", "Friendly"))
    creativity = st.slider("How creative do you want the ai to be?", 0.0, 1.0, 0.5, 0.01)
    units = st.radio("What units do you want the ai to use?", ("Metric", "Imperial"))
    explanation_level = st.radio("How detailed do you want the explanations to be?", ("Basic", "Intermediate", "Advanced"))
    cost_priority = st.slider("How important is cost to you?", 0.0, 1.0, 0.5, 0.01)
    performance_priority = st.slider("How important is performance to you?", 0.0, 1.0, 0.5, 0.01)
    reliability_priority = st.slider("How important is reliability to you?", 0.0, 1.0, 0.5, 0.01)
    safety_priority = st.slider("How important is safety to you?", 0.0, 1.0, 0.5, 0.01)

prompt = st.chat_input("Enter your question here: ")
response = question(prompt) if prompt else ("you haven't asked a question yet.")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("AI"):
        st.write(response)