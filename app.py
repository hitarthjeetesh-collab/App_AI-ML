import streamlit as st
import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# PAGE
# ============================================================

st.title("Robotics Engineering Assistant")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("Settings")

    model = st.selectbox(
        "Model",
        (
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        )
    )

    stream_it = st.toggle(
        "Stream response",
        value=True
    )

    response_length = st.radio(
        "Response length",
        (
            "Talkative",
            "Balanced",
            "Concise",
        ),
        horizontal=True
    )

    style = st.radio(
        "Response style",
        (
            "Professional",
            "Casual",
            "Friendly",
        ),
        horizontal=True
    )

    explanation_level = st.radio(
        "Explanation detail",
        (
            "Basic",
            "Intermediate",
            "Advanced",
        ),
        horizontal=True
    )

    creativity = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    units = st.radio(
        "Units",
        (
            "Metric",
            "Imperial",
        ),
        horizontal=True
    )

    st.subheader("Engineering Priorities")

    cost_priority = st.slider(
        "Cost",
        0.0,
        1.0,
        0.5,
        0.01
    )

    performance_priority = st.slider(
        "Performance",
        0.0,
        1.0,
        0.5,
        0.01
    )

    reliability_priority = st.slider(
        "Reliability",
        0.0,
        1.0,
        0.5,
        0.01
    )

    safety_priority = st.slider(
        "Safety",
        0.0,
        1.0,
        0.5,
        0.01
    )

    st.subheader("Reasoning")

    reasoning_effort = st.selectbox(
        "Reasoning effort",
        (
            "low",
            "medium",
            "high",
        ),
        index=1
    )

    if st.button(
        "Clear chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        f"{len(st.session_state.messages)} messages in this chat"
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = f"""
You are a Robotics Engineering Assistant.

Your job is to help the user with robotics engineering, including:

- Mechanical design
- Electrical design
- Motors and actuators
- Batteries and power systems
- Sensors
- Control systems
- Embedded systems
- Calculations
- Component selection
- Troubleshooting
- Optimization
- Prototyping
- Engineering tradeoffs

RESPONSE SETTINGS:

Response length: {response_length}
Response style: {style}
Explanation level: {explanation_level}
Creativity: {creativity}
Units: {units}

ENGINEERING PRIORITIES:

Cost priority: {cost_priority}
Performance priority: {performance_priority}
Reliability priority: {reliability_priority}
Safety priority: {safety_priority}

ENGINEERING BEHAVIOR:

For engineering problems:

1. Identify the important requirements.
2. Identify missing information.
3. State reasonable assumptions.
4. Determine the governing equations or engineering principles.
5. Perform the important calculations.
6. Identify limiting conditions.
7. Consider practical component constraints.
8. Explain important tradeoffs.
9. Give clear design recommendations.
10. Distinguish estimates from known specifications.

The Engineering Process section should contain a concise,
useful engineering analysis of the problem.

The final answer should contain the actual answer the user needs.

Do not put the final answer inside the Engineering Process.

Do not say "Analyzing problem..." as the only content
of the Engineering Process.

Show important calculations in the final answer when useful.

For simple questions, answer directly without unnecessary detail.

For casual conversation, respond naturally.

Do not mention system messages, developer instructions,
hidden configuration, instruction hierarchy, or internal
implementation.
"""


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Enter your question here:"
)


# ============================================================
# NEW MESSAGE
# ============================================================

if prompt:

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(prompt)

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        # ====================================================
        # STREAMING
        # ====================================================

        if stream_it:

            # ------------------------------------------------
            # CREATE ENGINEERING PROCESS CONTAINER FIRST
            # ------------------------------------------------

            with st.expander(
                "Engineering Process",
                expanded=True
            ):

                thinking_placeholder = st.empty()

            # ------------------------------------------------
            # CREATE FINAL ANSWER CONTAINER
            # ------------------------------------------------

            answer_placeholder = st.empty()

            # ------------------------------------------------
            # REQUEST
            # ------------------------------------------------

            stream = client.chat.completions.create(
                model=model,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    *st.session_state.messages
                ],

                temperature=creativity,

                reasoning_effort=reasoning_effort,

                stream=True,
            )

            # ------------------------------------------------
            # STORAGE
            # ------------------------------------------------

            reasoning_text = ""
            answer_text = ""

            # ------------------------------------------------
            # READ STREAM
            # ------------------------------------------------

            for chunk in stream:

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # ============================================
                # REASONING
                # ============================================

                reasoning = getattr(
                    delta,
                    "reasoning",
                    None
                )

                if reasoning:

                    reasoning_text += reasoning

                    thinking_placeholder.markdown(
                        reasoning_text
                    )

                # ============================================
                # FINAL CONTENT
                # ============================================

                content = getattr(
                    delta,
                    "content",
                    None
                )

                if content:

                    answer_text += content

                    answer_placeholder.markdown(
                        answer_text
                    )

            # ------------------------------------------------
            # IF NO REASONING WAS RETURNED
            # ------------------------------------------------

            if not reasoning_text:

                thinking_placeholder.markdown(
                    "*No reasoning content was returned by the model.*"
                )

            # ------------------------------------------------
            # SAVE FINAL ANSWER ONLY
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text
                }
            )


        # ====================================================
        # NON-STREAMING
        # ====================================================

        else:

            response = client.chat.completions.create(
                model=model,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    *st.session_state.messages
                ],

                temperature=creativity,

                reasoning_effort=reasoning_effort,
            )

            message = response.choices[0].message

            # ------------------------------------------------
            # FINAL ANSWER
            # ------------------------------------------------

            answer_text = message.content or ""

            # ------------------------------------------------
            # REASONING
            # ------------------------------------------------

            reasoning_text = getattr(
                message,
                "reasoning",
                None
            )

            # ------------------------------------------------
            # ENGINEERING PROCESS
            # ------------------------------------------------

            with st.expander(
                "Engineering Process",
                expanded=False
            ):

                if reasoning_text:

                    st.markdown(
                        reasoning_text
                    )

                else:

                    st.markdown(
                        "*No reasoning content was returned by the model.*"
                    )

            # ------------------------------------------------
            # FINAL ANSWER
            # ------------------------------------------------

            st.markdown(
                answer_text
            )

            # ------------------------------------------------
            # SAVE ANSWER
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text
                }
            )