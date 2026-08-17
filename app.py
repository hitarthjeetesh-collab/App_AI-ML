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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Robotics Engineering Assistant",
    page_icon="🤖",
    layout="wide",
)


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
    # CHAT COUNT
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


============================================================
ENGINEERING BEHAVIOR
============================================================

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

Do not blindly accept assumptions if they produce an unrealistic design.

For simple questions, answer directly without unnecessary detail.

For complex engineering questions, reason through the problem carefully
before producing the final answer.


============================================================
MATHEMATICAL FORMATTING
============================================================

You MAY and SHOULD use LaTeX for mathematical equations.

The application renders Markdown with Streamlit.

Use valid Markdown-compatible LaTeX.

For inline equations, use:

$F = ma$

For important equations on their own line, use:

$$
F = ma
$$

Use LaTeX for engineering calculations when it improves readability.

Examples include:

$$
F_{{grade}} = mg\\sin(\\theta)
$$

$$
F_{{rr}} = C_{{rr}}mg\\cos(\\theta)
$$

$$
F_{{total}} = F_{{grade}} + F_{{rr}}
$$

$$
T_{{wheel}} = F_{{wheel}}r
$$

$$
P = Fv
$$

$$
E = Pt
$$

For fractions, use:

$$
C = \\frac{{E}}{{V}}
$$

For multi-step calculations, format them clearly:

$$
F_{{grade}}
=
25 \\times 9.81 \\times \\sin(10^\\circ)
$$

$$
F_{{grade}} \\approx 42.6\\ \\text{{N}}
$$

IMPORTANT:

Never write a LaTeX equation inside square brackets.

BAD:

[ F_{{total}} = F_{{grade}} + F_{{rr}} ]

GOOD:

$$
F_{{total}} = F_{{grade}} + F_{{rr}}
$$

Do not output malformed LaTeX.

Use Markdown headings, bullet lists, and tables normally.

Do not unnecessarily use equations for simple numerical answers.


============================================================
ENGINEERING PROCESS
============================================================

When solving a substantial engineering problem, carefully reason through
the engineering process.

The application displays the model's reasoning separately inside an
"Engineering Process" box.

Do NOT write the heading "Engineering Process" yourself.

Begin the reasoning directly with the analysis.

For example:

"We first need to determine the force required to climb the incline."

Then continue with calculations and reasoning.

Use LaTeX normally inside the reasoning when appropriate.


============================================================
FINAL ANSWER
============================================================

The final answer should contain:

1. Important requirements
2. Assumptions
3. Relevant equations
4. Calculations
5. Results
6. Limiting conditions
7. Practical component considerations
8. Tradeoffs
9. Recommendations

Clearly distinguish calculated values from assumptions.

For engineering calculations, show enough work that the user can verify
the result.


============================================================
SAFETY
============================================================

Clearly identify important engineering safety considerations.

Do not claim that an estimate is a guaranteed specification.

Distinguish between:

- calculated requirements
- assumed values
- recommended ratings
- manufacturer specifications


============================================================
CASUAL QUESTIONS
============================================================

For casual conversation, respond naturally.

Do not force casual questions into an engineering format.

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
    # AI MESSAGE
    # --------------------------------------------------------

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
                        *st.session_state.messages,
                    ],

                    temperature=creativity,

                    reasoning_effort=reasoning_effort,

                    stream=True,
                )

            except Exception as e:

                st.error(
                    f"API request failed: {type(e).__name__}: {e}"
                )

                st.stop()


            # ------------------------------------------------
            # ENGINEERING PROCESS
            # ------------------------------------------------

            with st.expander(
                "Engineering Process",
                expanded=True,
            ):

                thinking_placeholder = st.empty()

                thinking_placeholder.markdown(
                    "*Analyzing problem...*"
                )


            # ------------------------------------------------
            # FINAL ANSWER
            # ------------------------------------------------

            answer_placeholder = st.empty()


            # ------------------------------------------------
            # TEXT STORAGE
            # ------------------------------------------------

            reasoning_text = ""
            answer_text = ""


            # ------------------------------------------------
            # RECEIVE STREAM
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
            # SAVE FINAL ANSWER
            # ------------------------------------------------

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
                            *st.session_state.messages,
                        ],

                        temperature=creativity,

                        reasoning_effort=reasoning_effort,
                    )

                except Exception as e:

                    st.error(
                        f"API request failed: {type(e).__name__}: {e}"
                    )

                    st.stop()


            # ------------------------------------------------
            # GET ANSWER
            # ------------------------------------------------

            answer_text = (
                response
                .choices[0]
                .message
                .content
            )


            # ------------------------------------------------
            # GET REASONING
            # ------------------------------------------------

            reasoning_text = getattr(
                response.choices[0].message,
                "reasoning",
                None,
            )


            # ------------------------------------------------
            # ENGINEERING PROCESS
            # ------------------------------------------------

            with st.expander(
                "Engineering Process",
                expanded=False,
            ):

                if reasoning_text:

                    st.markdown(
                        reasoning_text
                    )

                else:

                    st.markdown(
                        "*No reasoning was returned.*"
                    )


            # ------------------------------------------------
            # SHOW ANSWER
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