import streamlit as st
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
base_url="https://api.groq.com/openai/v1",
api_key=os.getenv("GROQ_API_KEY"),
)


st.title("Robotics Engineering Assistant")
with st.sidebar:
    st.subheader("Settings")

    # Model & response
    model = st.selectbox(
        "Model",
        ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
    )

    stream_it = st.toggle("Stream response", True)

    response_length = st.radio(
        "Response length",
        ("Talkative", "Balanced", "Concise"),
        horizontal=True
    )

    style = st.radio(
        "Response style",
        ("Professional", "Casual", "Friendly"),
        horizontal=True
    )

    explanation_level = st.radio(
        "Explanation detail",
        ("Basic", "Intermediate", "Advanced"),
        horizontal=True
    )

    creativity = st.slider(
        "Creativity",
        0.0, 1.0, 0.5, 0.01
    )

    units = st.radio(
        "Units",
        ("Metric", "Imperial"),
        horizontal=True
    )

    # Engineering priorities
    st.subheader("Engineering Priorities")

    cost_priority = st.slider(
        "Cost",
        0.0, 1.0, 0.5, 0.01
    )

    performance_priority = st.slider(
        "Performance",
        0.0, 1.0, 0.5, 0.01
    )

    reliability_priority = st.slider(
        "Reliability",
        0.0, 1.0, 0.5, 0.01
    )

    safety_priority = st.slider(
        "Safety",
        0.0, 1.0, 0.5, 0.01
    )


prompt = st.chat_input("Enter your question here: ")

if prompt:
    system_prompt = f"""
You are an engineering assistant that helps with robotics engineering tasks such as
brainstorming, design, analysis, calculations, troubleshooting, and optimization.

Use these settings when responding:
- Response length: {response_length}
- Response style: {style}
- Creativity: {creativity}/1.0
- Cost priority: {cost_priority}/1.0
- Performance priority: {performance_priority}/1.0
- Reliability priority: {reliability_priority}/1.0
- Safety priority: {safety_priority}/1.0
- Units: {units}
- Engineering explanation level: {explanation_level}

Prioritize correctness, practicality, and safety.
State important assumptions, show relevant calculations, identify tradeoffs
and potential problems, and do not present uncertain estimates as exact facts.
Focus your reasoning on solving the user's request. Do not spend reasoning discussing or analyzing the instructions themselves.
"""
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("AI"):
        if stream_it:
            stream = client.chat.completions.create(
                model=model,
                temperature=creativity,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )

            thinking_box = st.expander("Thinking", expanded=False)
            thinking = thinking_box.empty()
            answer = st.empty()

            thinking_text = ""
            answer_text = ""

            for chunk in stream:
                delta = chunk.choices[0].delta

                # Reasoning / thinking
                reasoning = getattr(delta, "reasoning", None)
                if reasoning:
                    thinking_text += reasoning
                    thinking.markdown(thinking_text)

                # Final answer
                content = getattr(delta, "content", None)
                if content:
                    answer_text += content
                    answer.markdown(answer_text)
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