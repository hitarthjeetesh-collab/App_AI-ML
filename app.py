import os
import streamlit as st
from openai import OpenAI

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Robotics Engineering Assistant",
    page_icon="🤖",
    layout="wide",
)

# ============================================================
# CONFIG
# ============================================================

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")

SYSTEM_PROMPT = r"""
# Robotics Engineering Assistant

You are a Robotics Engineering AI assistant.

Your purpose is to help users design, analyze, troubleshoot, and improve
robotics systems using engineering principles.

You should reason through engineering problems systematically.

## Engineering Process

When solving an engineering design problem, structure the response around:

1. Important requirements
2. Missing information
3. Explicit assumptions
4. Governing equations
5. Calculations
6. Results
7. Limiting conditions
8. Practical component considerations
9. Trade-offs
10. Design recommendations

## Mathematical formatting

IMPORTANT:

Use proper LaTeX for mathematical equations.

Use:

- Inline math: \( ... \)
- Display equations: \[ ... \]

For multi-line equations, use:

\[
\begin{aligned}
F_{\text{grade}} &= mg\sin(\theta) \\
F_{\text{rr}} &= C_{\text{rr}}mg\cos(\theta)
\end{aligned}
\]

Do NOT write raw LaTeX-looking text such as:

[ F_{total}=... ]

Do NOT remove LaTeX backslashes.

Do NOT put LaTeX equations inside Markdown code blocks.

Use Markdown tables where appropriate.

## Engineering rigor

Clearly distinguish between:

- calculated values
- assumptions
- estimates
- component recommendations

Check units.

Do not invent precise component specifications when they are unknown.

If a parameter significantly affects the result, explain how.

When comparing multiple engineering options, provide the pros and cons of each.

For motors, distinguish between:

- motor torque
- gearbox output torque
- wheel torque
- mechanical power
- electrical input power

For batteries, distinguish between:

- nominal voltage
- Ah capacity
- Wh energy
- usable energy
- battery efficiency
- reserve capacity

Do not assume that a motor's advertised peak torque is its continuous torque.

## Robotics-specific considerations

Consider when relevant:

- acceleration
- grade resistance
- rolling resistance
- aerodynamic drag
- drivetrain efficiency
- motor efficiency
- gearbox ratio
- wheel radius
- traction
- thermal limits
- battery discharge rate
- center of gravity
- weight distribution
- braking
- safety margin

Do not blindly accept the user's calculations. Check them independently.

## Response style

Be technically detailed but readable.

Use headings, equations, tables, and bullet points.

Do not expose hidden chain-of-thought or private reasoning.
Instead, provide concise engineering calculations and explanations.
"""

# ============================================================
# OPENAI CLIENT
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY is not configured. "
        "Set it as an environment variable before running the app."
    )
    st.stop()

client = OpenAI(api_key=api_key)

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🤖 Robotics Engineering Assistant")

    st.markdown(
        """
        **Purpose**

        An AI engineering assistant specialized in robotics.

        **Capabilities**

        - Mechanical design
        - Motor sizing
        - Battery sizing
        - Power calculations
        - Control systems
        - Sensors
        - Embedded systems
        - Robotics software
        - Engineering trade-offs
        """
    )

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# TITLE
# ============================================================

st.title("Robotics Engineering Assistant")

st.caption(
    "Design, calculate, troubleshoot, and optimize robotics systems."
)

# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Describe your robotics engineering problem..."
)

if user_input:

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------------------------------------
    # Build API messages
    # --------------------------------------------------------

    api_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    api_messages.extend(st.session_state.messages)

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=api_messages,
            )

            answer = response.choices[0].message.content

            if not answer:
                answer = "The model returned an empty response."

            response_placeholder.markdown(answer)

            # ------------------------------------------------
            # Save assistant response
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as e:

            error_message = (
                "### API Error\n\n"
                f"`{type(e).__name__}: {e}`"
            )

            response_placeholder.markdown(error_message)

            # Remove the user message if the request failed
            if (
                st.session_state.messages
                and st.session_state.messages[-1]["role"] == "user"
            ):
                st.session_state.messages.pop()