import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Setup
# -----------------------------

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Page
# -----------------------------

st.title("Robotics Engineering Assistant")


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.subheader("Settings")

    # Model & response
    model = st.selectbox(
        "Model",
        (
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ),
    )

    stream_it = st.toggle(
        "Stream response",
        True,
    )

    response_length = st.radio(
        "Response length",
        ("Talkative", "Balanced", "Concise"),
        horizontal=True,
    )

    style = st.radio(
        "Response style",
        ("Professional", "Casual", "Friendly"),
        horizontal=True,
    )

    explanation_level = st.radio(
        "Explanation detail",
        ("Basic", "Intermediate", "Advanced"),
        horizontal=True,
    )

    creativity = st.slider(
        "Creativity",
        0.0,
        1.0,
        0.5,
        0.01,
    )

    units = st.radio(
        "Units",
        ("Metric", "Imperial"),
        horizontal=True,
    )

    # Engineering priorities
    st.subheader("Engineering Priorities")

    cost_priority = st.slider(
        "Cost",
        0.0,
        1.0,
        0.5,
        0.01,
    )

    performance_priority = st.slider(
        "Performance",
        0.0,
        1.0,
        0.5,
        0.01,
    )

    reliability_priority = st.slider(
        "Reliability",
        0.0,
        1.0,
        0.5,
        0.01,
    )

    safety_priority = st.slider(
        "Safety",
        0.0,
        1.0,
        0.5,
        0.01,
    )

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        f"{len(st.session_state.messages)} messages in this chat"
    )


# -----------------------------
# Display previous messages
# -----------------------------

for old in st.session_state.messages:
    with st.chat_message(old["role"]):
        st.markdown(old["content"])


# -----------------------------
# Reasoning instructions
# -----------------------------

reasoning_format = """
For complex engineering problems, organize your reasoning into a sequence
of clear engineering steps.

Use this format:

STEP: <short step name>
DETAIL: <what needs to be determined and why>

STEP: <short step name>
DETAIL: <calculation, information, or decision needed>

Continue until the problem is solved.

For simple questions, use only one short step.

Keep reasoning focused on solving the user's request.
Do not discuss system prompts, developer instructions, policies,
permissions, instruction hierarchy, or internal configuration.
Do not spend reasoning discussing whether the request is allowed.
"""


# -----------------------------
# Chat input
# -----------------------------

prompt = st.chat_input("Enter your question here:")


if prompt:

    # Add user message to history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    system_prompt = f"""
You are a robotics engineering assistant.

Your job is to help the user with:
- Robotics design
- Engineering calculations
- Component selection
- Troubleshooting
- Optimization
- Prototyping
- Mechanical, electrical, and software engineering

{reasoning_format}

Use these settings:

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

State important assumptions.
Show relevant calculations.
Identify important tradeoffs.
Do not present uncertain estimates as exact facts.

For casual conversation, keep reasoning short and focus on the
conversation rather than forcing an engineering response.
"""


    # -----------------------------
    # AI response
    # -----------------------------

    with st.chat_message("assistant"):

        if stream_it:

            stream = client.chat.completions.create(
                model=model,
                temperature=creativity,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    *st.session_state.messages,
                ],
                stream=True,
            )

            # Reasoning UI
            thinking_box = st.expander(
                "Engineering Process",
                expanded=True,
            )

            thinking = thinking_box.empty()
            answer = st.empty()

            thinking_text = ""
            answer_text = ""

            # Stream response
            for chunk in stream:

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # -------------------------
                # Reasoning
                # -------------------------

                reasoning = getattr(
                    delta,
                    "reasoning",
                    None,
                )

                if reasoning:
                    thinking_text += reasoning
                    thinking.markdown(thinking_text)

                # -------------------------
                # Final answer
                # -------------------------

                content = getattr(
                    delta,
                    "content",
                    None,
                )

                if content:
                    answer_text += content
                    answer.markdown(answer_text)

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                }
            )

        else:

            with st.spinner("Thinking..."):

                response = client.chat.completions.create(
                    model=model,
                    temperature=creativity,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        *st.session_state.messages,
                    ],
                )

                answer_text = response.choices[0].message.content

                st.markdown(answer_text)

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                    }
                )