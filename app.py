import os
import re

import streamlit as st
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

# Maximum output tokens.
#
# The selected Groq models support up to approximately 8K
# completion tokens. 8000 gives the assistant enough room
# for detailed engineering calculations without unnecessarily
# requesting more than the model supports.
MAX_COMPLETION_TOKENS = 8000

# Number of previous messages sent to the model.
#
# 4 messages =
# User
# Assistant
# User
# Assistant
#
# The complete conversation remains stored locally in
# st.session_state.
MAX_HISTORY_MESSAGES = 10


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
# LATEX CLEANING
# ============================================================

def clean_latex(text):
    """
    Clean common malformed LaTeX without modifying
    normal Markdown headings, tables, or prose.
    """

    if not text:
        return text

    # --------------------------------------------------------
    # Normalize escaped dollar signs
    # --------------------------------------------------------

    text = text.replace(r"\$", "$")

    # --------------------------------------------------------
    # Fix corrupted alignment spacing such as:
    #
    # \$$4pt]
    # \\$4pt]
    # \$4pt]
    # \\[4pt]
    #
    # Convert these to a normal LaTeX line break.
    # --------------------------------------------------------

    text = re.sub(
        r"(?:\\+|\$+)?\s*\$?\s*\[?\s*4pt\s*\]",
        r"\\\\",
        text,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Fix double-escaped common LaTeX commands
    # --------------------------------------------------------

    commands = [
        "frac",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "log",
        "ln",
        "exp",
        "times",
        "approx",
        "pm",
        "omega",
        "theta",
        "eta",
        "mu",
        "pi",
        "sum",
        "int",
        "cdot",
        "text",
        "mathrm",
        "left",
        "right",
        "begin",
        "end",
    ]

    for command in commands:
        text = text.replace(
            f"\\\\{command}",
            f"\\{command}",
        )

    # --------------------------------------------------------
    # Convert \[ ... \] to $$ ... $$
    # --------------------------------------------------------

    text = re.sub(
        r"\\\[\s*([\s\S]*?)\s*\\\]",
        lambda match: (
            "\n\n$$\n"
            + match.group(1).strip()
            + "\n$$\n\n"
        ),
        text,
    )

    # --------------------------------------------------------
    # Convert \( ... \) to $ ... $
    # --------------------------------------------------------

    text = re.sub(
        r"\\\(\s*([\s\S]*?)\s*\\\)",
        lambda match: (
            "$"
            + match.group(1).strip()
            + "$"
        ),
        text,
    )

    # --------------------------------------------------------
    # Fix aligned environments missing $$ delimiters
    # --------------------------------------------------------

    aligned_pattern = re.compile(
        r"(?<!\$)"
        r"(\\begin\{aligned\}[\s\S]*?\\end\{aligned\})"
        r"(?!\$)",
        flags=re.MULTILINE,
    )

    def wrap_aligned(match):
        equation = match.group(1).strip()

        return (
            "\n\n"
            "$$\n"
            + equation
            + "\n$$"
            "\n\n"
        )

    text = aligned_pattern.sub(
        wrap_aligned,
        text,
    )

    # --------------------------------------------------------
    # Fix malformed aligned spacing
    # --------------------------------------------------------

    text = re.sub(
        r"\\{2,}\s*\[4pt\]",
        r"\\\\",
        text,
    )

    # --------------------------------------------------------
    # Remove excessive blank lines
    # --------------------------------------------------------

    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text,
    )

    return text


def render_markdown(text):
    """
    Render cleaned model output through Streamlit Markdown.
    """

    if not text:
        return

    cleaned = clean_latex(text)

    st.markdown(cleaned)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("Settings")

    # ========================================================
    # MODEL
    # ========================================================

    model = st.selectbox(
        "Model",
        (
            "openai/gpt-oss-120b",
            "qwen/qwen3.6-27b",
            "openai/gpt-oss-20b",
        ),
        index=0,
    )

    # ========================================================
    # STREAMING
    # ========================================================

    stream_it = st.toggle(
        "Stream response",
        value=True,
    )

    # ========================================================
    # RESPONSE LENGTH
    # ========================================================

    response_length = st.radio(
        "Response length",
        (
            "Talkative",
            "Balanced",
            "Concise",
        ),
        horizontal=True,
        index=1,
    )

    # ========================================================
    # RESPONSE STYLE
    # ========================================================

    style = st.radio(
        "Response style",
        (
            "Professional",
            "Casual",
            "Friendly",
        ),
        horizontal=True,
        index=0,
    )

    # ========================================================
    # EXPLANATION LEVEL
    # ========================================================

    explanation_level = st.radio(
        "Explanation detail",
        (
            "Basic",
            "Intermediate",
            "Advanced",
        ),
        horizontal=True,
        index=2,
    )

    # ========================================================
    # CREATIVITY
    # ========================================================

    creativity = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    # ========================================================
    # UNITS
    # ========================================================

    units = st.radio(
        "Units",
        (
            "Metric",
            "Imperial",
        ),
        horizontal=True,
        index=0,
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

    # ========================================================
    # CHAT COUNT
    # ========================================================

    st.caption(
        f"{len(st.session_state.messages)} messages in this chat"
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = r"""
You are a Robotics Engineering Assistant.

For engineering problems:
- identify requirements and missing information
- state assumptions
- derive and calculate relevant equations
- account for real-world losses and safety margins
- check results for physical plausibility
- discuss practical constraints and tradeoffs
- distinguish assumptions, calculations, estimates, and specifications

Use Markdown and valid LaTeX.
Never use square brackets as math delimiters.
Never put raw LaTeX outside math delimiters.

For complex problems:
Requirements → Assumptions → Equations → Calculations →
Results → Limitations → Practical considerations → Recommendations.

Follow the user's settings below.
"""


# ============================================================
# CURRENT USER SETTINGS
# ============================================================

system_prompt += f"""
SETTINGS
Response length: {response_length}
Style: {style}
Explanation: {explanation_level}
Creativity: {creativity}
Units: {units}
Priorities: cost={cost_priority}, performance={performance_priority}, reliability={reliability_priority}, safety={safety_priority}
Reasoning: {reasoning_effort}
"""

# ============================================================
# PAGE TITLE
# ============================================================

st.title("Robotics Engineering Assistant")


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        render_markdown(
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

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message("user"):

        render_markdown(prompt)

    # ========================================================
    # LIMITED HISTORY
    # ========================================================

    recent_messages = (
        st.session_state.messages[
            -MAX_HISTORY_MESSAGES:
        ]
    )

    # ========================================================
    # ASSISTANT MESSAGE
    # ========================================================

    with st.chat_message("assistant"):

        # ====================================================
        # STREAMING
        # ====================================================

        if stream_it:

            try:

                stream = client.chat.completions.create(
                    model=model,

                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        *recent_messages,
                    ],

                    temperature=creativity,

                    reasoning_effort=reasoning_effort,

                    max_completion_tokens=MAX_COMPLETION_TOKENS,

                    stream=True,
                )

            except Exception as e:

                st.error(
                    f"API request failed: "
                    f"{type(e).__name__}: {e}"
                )

                st.session_state.messages.pop()

                st.stop()

            # =================================================
            # ENGINEERING PROCESS
            # =================================================

            with st.expander(
                "Engineering Process",
                expanded=True,
            ):

                thinking_placeholder = st.empty()

                thinking_placeholder.markdown(
                    "*Analyzing problem...*"
                )

            # =================================================
            # FINAL ANSWER
            # =================================================

            answer_placeholder = st.empty()

            reasoning_text = ""
            answer_text = ""

            # =================================================
            # RECEIVE STREAM
            # =================================================

            for chunk in stream:

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # =============================================
                # REASONING
                # =============================================

                reasoning = getattr(
                    delta,
                    "reasoning",
                    None,
                )

                if reasoning:

                    reasoning_text += reasoning

                    thinking_placeholder.markdown(
                        clean_latex(reasoning_text)
                    )

                # =============================================
                # FINAL ANSWER
                # =============================================

                content = getattr(
                    delta,
                    "content",
                    None,
                )

                if content:

                    answer_text += content

                    answer_placeholder.markdown(
                        clean_latex(answer_text)
                    )

            # =================================================
            # REASONING FALLBACK
            # =================================================

            if not reasoning_text:

                thinking_placeholder.markdown(
                    "*No separate reasoning was returned by the model.*"
                )

            # =================================================
            # SAVE ANSWER
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                }
            )

        # ====================================================
        # NON-STREAMING
        # ====================================================

        else:

            with st.spinner("Thinking..."):

                try:

                    response = client.chat.completions.create(
                        model=model,

                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            *recent_messages,
                        ],

                        temperature=creativity,

                        reasoning_effort=reasoning_effort,

                        max_completion_tokens=MAX_COMPLETION_TOKENS,
                    )

                except Exception as e:

                    st.error(
                        f"API request failed: "
                        f"{type(e).__name__}: {e}"
                    )

                    st.session_state.messages.pop()

                    st.stop()

            # =================================================
            # GET ANSWER
            # =================================================

            answer_text = (
                response
                .choices[0]
                .message
                .content
                or ""
            )

            # =================================================
            # GET REASONING
            # =================================================

            reasoning_text = getattr(
                response.choices[0].message,
                "reasoning",
                None,
            )

            # =================================================
            # ENGINEERING PROCESS
            # =================================================

            with st.expander(
                "Engineering Process",
                expanded=False,
            ):

                if reasoning_text:

                    render_markdown(
                        reasoning_text
                    )

                else:

                    st.markdown(
                        "*No reasoning was returned.*"
                    )

            # =================================================
            # DISPLAY ANSWER
            # =================================================

            render_markdown(answer_text)

            # =================================================
            # SAVE ANSWER
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                }
            )