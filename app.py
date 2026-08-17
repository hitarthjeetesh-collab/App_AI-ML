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


# IMPORTANT:
# We are using the OpenAI Python client,
# but connecting it to Groq's OpenAI-compatible API.
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
# PAGE TITLE
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

REASONING BEHAVIOR

You are a reasoning model.

Before producing the final answer, reason through the engineering
problem carefully.

Your reasoning may include:

- Identifying requirements
- Finding missing information
- Stating assumptions
- Selecting equations
- Performing calculations
- Checking units
- Checking whether results are physically reasonable
- Considering limiting conditions
- Considering safety margins
- Comparing engineering tradeoffs

IMPORTANT:

The application automatically displays your reasoning separately
from your final answer.

Therefore:

DO NOT write a section called:
"Engineering Process"

DO NOT write:
"Analyzing problem..."

DO NOT manually expose or reproduce your reasoning in the final
answer.

DO NOT put your reasoning inside the final response.

Your final response should contain ONLY the answer intended for the
user.

You may include:

- Results
- Important equations
- Important calculations
- Assumptions
- Design recommendations
- Component specifications
- Tradeoffs

But do not reproduce the hidden reasoning process.

ENGINEERING BEHAVIOR

For engineering problems:

1. Identify important requirements.
2. Determine missing information.
3. Make reasonable assumptions.
4. Select appropriate equations.
5. Perform calculations.
6. Check the result.
7. Identify limiting conditions.
8. Consider practical component constraints.
9. Explain important tradeoffs.
10. Give clear design recommendations.

Distinguish estimates from known specifications.

Use the user's selected units.

For simple questions, answer directly.

For complex engineering questions, provide enough calculation
detail that the user can verify the result.

Do not blindly accept assumptions if they produce an unrealistic
design.

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
    # SAVE USER MESSAGE
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
    # ASSISTANT MESSAGE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        # ====================================================
        # STREAMING MODE
        # ====================================================

        if stream_it:

            # ------------------------------------------------
            # CREATE REASONING CONTAINER FIRST
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
            # CREATE FINAL ANSWER PLACEHOLDER
            # ------------------------------------------------

            answer_placeholder = st.empty()

            # ------------------------------------------------
            # TEXT STORAGE
            # ------------------------------------------------

            reasoning_text = ""
            answer_text = ""

            # ------------------------------------------------
            # API REQUEST
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

                    include_reasoning=True,

                    stream=True,
                )

                # ------------------------------------------------
                # RECEIVE STREAM
                # ------------------------------------------------

                for chunk in stream:

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # ========================================
                    # REASONING
                    # ========================================

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

                    # ========================================
                    # FINAL ANSWER
                    # ========================================

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
                # NO ANSWER FALLBACK
                # ------------------------------------------------

                if not answer_text:

                    answer_text = (
                        "The model returned no final answer."
                    )

                    answer_placeholder.markdown(
                        answer_text
                    )

                # ------------------------------------------------
                # NO REASONING FALLBACK
                # ------------------------------------------------

                if not reasoning_text:

                    thinking_placeholder.markdown(
                        "*Reasoning was not returned by the model.*"
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
                    f"API request failed: {type(e).__name__}: {e}"
                )

                # Remove the user message if the request failed
                if st.session_state.messages:
                    st.session_state.messages.pop()


        # ====================================================
        # NON-STREAMING MODE
        # ====================================================

        else:

            # ------------------------------------------------
            # CREATE REASONING CONTAINER
            # ------------------------------------------------

            process_container = st.expander(
                "Engineering Process",
                expanded=False,
            )

            try:

                with st.spinner("Analyzing problem..."):

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

                        include_reasoning=True,
                    )

                # ------------------------------------------------
                # GET MESSAGE
                # ------------------------------------------------

                message = response.choices[0].message

                # ------------------------------------------------
                # GET FINAL ANSWER
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
                # GET REASONING
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
                            "*Reasoning was not returned by the model.*"
                        )

                # ------------------------------------------------
                # DISPLAY FINAL ANSWER
                # ------------------------------------------------

                st.markdown(
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
                    f"API request failed: {type(e).__name__}: {e}"
                )

                # Remove failed user message
                if st.session_state.messages:
                    st.session_state.messages.pop()