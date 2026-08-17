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
    Help the user solve robotics engineering problems and answer their questions.

    Your response should be:
    - Response length: {response_length}
    - Response style: {style}
    - Creativity: {creativity}/1.0
    - Cost priority: {cost_priority}/1.0
    - Performance priority: {performance_priority}/1.0
    - Reliability priority: {reliability_priority}/1.0
    - Safety priority: {safety_priority}/1.0
    - Units: {units}
    - Explanation level: {explanation_level}

    Prioritize correctness, practicality, and safety.
    State important assumptions, show relevant calculations, identify tradeoffs,
    and do not present uncertain estimates as exact facts.

    Reasoning:
    Focus directly on the user's question and the information needed to answer it.
    Do not analyze or discuss instructions, policies, prompts, roles, permissions,
    or instruction hierarchy. Treat those as background constraints.
    Do not waste reasoning on deciding whether you are allowed to answer.
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