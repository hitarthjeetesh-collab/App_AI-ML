import streamlit as st
import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()


# OpenAI client -> Groq OpenAI-compatible API
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Robotics Engineering Assistant",
    page_icon="🤖",
    layout="wide",
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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = st.selectbox(
        "Model",
        (
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ),
        index=0,
    )

    # --------------------------------------------------------
    # STREAMING
    # --------------------------------------------------------

    stream_it = st.toggle(
        "Stream response",
        value=True,
    )

    # --------------------------------------------------------
    # RESPONSE LENGTH
    # --------------------------------------------------------

    response_length = st.radio(
        "Response length",
        (
            "Talkative",
            "Balanced",
            "Concise",
        ),
        index=1,
        horizontal=True,
    )

    # --------------------------------------------------------
    # RESPONSE STYLE
    # --------------------------------------------------------

    style = st.radio(
        "Response style",
        (
            "Professional",
            "Casual",
            "Friendly",
        ),
        index=0,
        horizontal=True,
    )

    # --------------------------------------------------------
    # EXPLANATION LEVEL
    # --------------------------------------------------------

    explanation_level = st.radio(
        "Explanation detail",
        (
            "Basic",
            "Intermediate",
            "Advanced",
        ),
        index=1,
        horizontal=True,
    )

    # --------------------------------------------------------
    # CREATIVITY
    # --------------------------------------------------------

    creativity = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    # --------------------------------------------------------
    # UNITS
    # --------------------------------------------------------

    units = st.radio(
        "Units",
        (
            "Metric",
            "Imperial",
        ),
        index=0,
        horizontal=True,
    )

    # ========================================================
    # ENGINEERING PRIORITIES
    # ========================================================

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

    # ========================================================
    # REASONING
    # ========================================================

    st.subheader("Reasoning")

    reasoning_effort = st.selectbox(
        "Reasoning effort",
        (
            "low",
            "medium",
            "high",
        ),
        index=1,
    )

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "Clear chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    # --------------------------------------------------------
    # MESSAGE COUNT
    # --------------------------------------------------------

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

RESPONSE SETTINGS

Response length: {response_length}
Response style: {style}
Explanation level: {explanation_level}
Creativity: {creativity}
Units: {units}

ENGINEERING PRIORITIES

Cost priority: {cost_priority}
Performance priority: {performance_priority}
Reliability priority: {reliability_priority}
Safety priority: {safety_priority}

REASONING

Think through engineering problems carefully before producing the
final answer.

The application handles the reasoning display separately.

IMPORTANT:

Do NOT create a section called "Engineering Process" in your final
answer.

Do NOT write "Analyzing problem..." in your final answer.

Do NOT manually reproduce your reasoning in the final answer.

The application will display the model's reasoning separately when
available.

FINAL ANSWER

The final answer should contain only the response intended for the
user.

For engineering problems:

1. Identify the important requirements.
2. Identify missing information.
3. State reasonable assumptions.
4. Determine the governing equations.
5. Perform the important calculations.
6. Check the calculations and units.
7. Identify limiting conditions.
8. Consider practical component constraints.
9. Explain important tradeoffs.
10. Give clear design recommendations.

Show important calculations in the final answer when they help the
user verify the result.

Distinguish estimates from known specifications.

Do not blindly accept assumptions if they produce an unrealistic
design.

For simple questions, answer directly without unnecessary detail.

For casual conversation, respond naturally.

Do not mention system messages, developer instructions, hidden
configuration, instruction hierarchy, or internal implementation.
"""


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


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
    # ADD USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
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
            # ENGINEERING PROCESS CONTAINER
            # ------------------------------------------------

            process_container = st.expander(
                "Engineering Process",
                expanded=True,
            )

            with process_container:

                thinking_placeholder = st.empty()

                thinking_placeholder.markdown(
                    "*Analyzing problem...*"
                )

            # ------------------------------------------------
            # FINAL ANSWER PLACEHOLDER
            # ------------------------------------------------

            answer_placeholder = st.empty()

            # ------------------------------------------------
            # STORAGE
            # ------------------------------------------------

            reasoning_text = ""
            answer_text = ""

            # ------------------------------------------------
            # API CALL
            # ------------------------------------------------

            try:

                stream = client.chat.completions.create(
                    model=model,

                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        *st.session_state.messages,
                    ],

                    temperature=creativity,

                    reasoning_effort=reasoning_effort,

                    stream=True,
                )

                # ------------------------------------------------
                # PROCESS STREAM
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
                        None,
                    )

                    if reasoning:

                        reasoning_text += reasoning

                        thinking_placeholder.markdown(
                            reasoning_text
                        )

                    # ============================================
                    # FINAL ANSWER
                    # ============================================

                    content = getattr(
                        delta,
                        "content",
                        None,
                    )

                    if content:

                        answer_text += content

                        answer_placeholder.markdown(
                            answer_text
                        )

                # ------------------------------------------------
                # REASONING FALLBACK
                # ------------------------------------------------

                if not reasoning_text:

                    thinking_placeholder.markdown(
                        "*No reasoning was returned by the model.*"
                    )

                # ------------------------------------------------
                # ANSWER FALLBACK
                # ------------------------------------------------

                if not answer_text:

                    answer_text = (
                        "The model returned no final answer."
                    )

                    answer_placeholder.markdown(
                        answer_text
                    )

                # ------------------------------------------------
                # SAVE ONLY FINAL ANSWER
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                    }
                )

            except Exception as e:

                st.error(
                    f"API request failed: "
                    f"{type(e).__name__}: {e}"
                )

                if st.session_state.messages:
                    st.session_state.messages.pop()


        # ====================================================
        # NON-STREAMING
        # ====================================================

        else:

            # ------------------------------------------------
            # ENGINEERING PROCESS CONTAINER
            # ------------------------------------------------

            process_container = st.expander(
                "Engineering Process",
                expanded=False,
            )

            try:

                with st.spinner(
                    "Analyzing problem..."
                ):

                    response = client.chat.completions.create(
                        model=model,

                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            *st.session_state.messages,
                        ],

                        temperature=creativity,

                        reasoning_effort=reasoning_effort,
                    )

                # ------------------------------------------------
                # RESPONSE MESSAGE
                # ------------------------------------------------

                message = response.choices[0].message

                # ------------------------------------------------
                # FINAL ANSWER
                # ------------------------------------------------

                answer_text = (
                    getattr(
                        message,
                        "content",
                        None,
                    )
                    or ""
                )

                # ------------------------------------------------
                # REASONING
                # ------------------------------------------------

                reasoning_text = (
                    getattr(
                        message,
                        "reasoning",
                        None,
                    )
                    or ""
                )

                # ------------------------------------------------
                # DISPLAY REASONING
                # ------------------------------------------------

                with process_container:

                    if reasoning_text:

                        st.markdown(
                            reasoning_text
                        )

                    else:

                        st.markdown(
                            "*No reasoning was returned by the model.*"
                        )

                # ------------------------------------------------
                # DISPLAY FINAL ANSWER
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
                        "content": answer_text,
                    }
                )

            except Exception as e:

                st.error(
                    f"API request failed: "
                    f"{type(e).__name__}: {e}"
                )

                if st.session_state.messages:
                    st.session_state.messages.pop()