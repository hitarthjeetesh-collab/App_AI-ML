import streamlit as st
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
base_url="https://api.groq.com/openai/v1",
api_key=os.getenv("GITHUB_TOKEN"),
)


st.title("Engineering Assistant")
with st.sidebar:
    st.subheader("Settings")
    model = st.selectbox("Select the model to use:", ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"))
    stream_it = st.toggle("Stream response", True)
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

if prompt:
    system_prompt = f"You are an engineering assistant that helps with engineering tasks such as brainstorming, design, analysis, calculations, troubleshooting, and optimization. Use these settings when responding: response length: {response_length}; response style: {style}; creativity: {creativity}/1.0; cost priority: {cost_priority}/1.0; performance priority: {performance_priority}/1.0; reliability priority: {reliability_priority}/1.0; safety priority: {safety_priority}/1.0; units: {units}; engineering explanation level: {explanation_level}. Prioritize correctness, practicality, and safety. State important assumptions, show relevant calculations, identify tradeoffs and potential problems, and do not present uncertain estimates as exact facts."
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("AI"):
        if stream_it:
            stream = client.chat.completions.create(
                model=model,
                temperature=creativity,
                messages=[
                    {"role": "system",
                     "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )
            st.write_stream(stream)
        else:
            with st.spinner("Thinking..."):
                r = client.chat.completions.create(
                    model=model,
                    temperature=creativity,
                    messages=[
                        {"role": "system",
                         "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                st.write(r.choices[0].message.content)