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

MAX_TOKENS = 4000
MAX_HISTORY_MESSAGES = 4


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
    Clean common LaTeX formatting problems without
    modifying normal Markdown such as headings.
    """

    if not text:
        return text

    # --------------------------------------------------------
    # Normalize escaped dollar signs
    # --------------------------------------------------------

    text = text.replace(r"\$", "$")

    # --------------------------------------------------------
    # Fix common corrupted [4pt] alignment endings
    # --------------------------------------------------------

    text = re.sub(
        r"(?:\\+|\$+)?\s*\$?\s*4pt\s*\]",
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
    # Fix aligned equations that are missing $$ delimiters
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
    Render model output as Streamlit Markdown.
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
        index=1,
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
        index=0,
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
        index=2,
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
#
# IMPORTANT:
#
# This is intentionally NOT an f-string.
#
# LaTeX contains many { } characters, for example:
#
# \text{N}
# \frac{a}{b}
# \begin{aligned}
#
# Using an f-string here would cause Python to interpret
# those braces as Python expressions.
# ============================================================

system_prompt = r"""
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

============================================================
RESPONSE SETTINGS
============================================================

Follow the settings provided below.

============================================================
ENGINEERING BEHAVIOR
============================================================

For substantial engineering problems:

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

For complex engineering questions, reason carefully before producing
the final answer.

============================================================
MATHEMATICAL FORMATTING
============================================================

Use REAL LaTeX whenever displaying mathematical equations.

The application uses Streamlit Markdown, which supports LaTeX.

Inline mathematics:

$F = ma$

Display mathematics:

$$
F = ma
$$

For multi-line equations:

$$
\begin{aligned}
F_{grade} &= mg\sin(\theta) \\
F_{rr} &= C_{rr}mg\cos(\theta) \\
F_{total} &= F_{grade} + F_{rr}
\end{aligned}
$$

Use LaTeX commands such as:

- \frac{a}{b}
- \sqrt{x}
- \sin(\theta)
- \cos(\theta)
- \tan(\theta)
- \times
- \approx
- \omega
- \theta
- \eta
- \text{N}
- \text{W}
- \text{kg}
- \text{m/s}

============================================================
LATEX SIMPLICITY RULE
============================================================

Prefer SIMPLE LaTeX.

For most calculations, use separate display equations:

$$
F_g = mg\sin(\theta)
$$

$$
F_g =
25 \times 9.81 \times \sin(10^\circ)
\approx 42.6\ \text{N}
$$

Only use the aligned environment when multiple equations genuinely
need to be vertically aligned.

When using aligned, ALWAYS use exactly this structure:

$$
\begin{aligned}
F_g &= mg\sin(\theta) \\
F_{rr} &= C_{rr}mg\cos(\theta) \\
F_{total} &= F_g + F_{rr}
\end{aligned}
$$

NEVER generate malformed constructs such as:

\begin{aligned} ... \$$4pt] ... \end{aligned}

NEVER generate:

[ equation ]

NEVER generate raw LaTeX outside a math delimiter.

Do not use [4pt] spacing unless it is actually needed.

Prefer normal line breaks with \\ inside aligned.

============================================================
VERY IMPORTANT LATEX RULES
============================================================

NEVER write mathematical equations using square brackets.

BAD:

[ F_{total} = F_{grade} + F_{rr} ]

BAD:

[ F = ma ]

GOOD:

$$
F_{total} = F_{grade} + F_{rr}
$$

GOOD:

$$
F = ma
$$

Do NOT use square brackets as an alternative to LaTeX delimiters.

Do NOT write raw LaTeX outside a Markdown math delimiter.

Do NOT put equations inside Markdown code blocks.

Do NOT escape LaTeX backslashes unnecessarily.

The final response should contain normal Markdown-compatible LaTeX.

For calculations, show the equation first and then substitute numbers.

Example:

$$
F_{grade} = mg\sin(10^\circ)
$$

$$
F_{grade}
=
25 \times 9.81 \times \sin(10^\circ)
\approx 42.5\ \text{N}
$$

============================================================
HEADINGS AND MARKDOWN
============================================================

Use normal Markdown headings when helpful.

For example:

### 4.2 Wheel torque

Do NOT escape the # characters.

Do NOT put headings inside code blocks.

Do NOT put headings inside LaTeX delimiters.

Tables may be used when they improve clarity.

============================================================
ENGINEERING PROCESS
============================================================

The application displays the model's reasoning separately inside an
"Engineering Process" box.

Do NOT write the heading "Engineering Process" yourself.

Do NOT put the phrase "Engineering Process" at the beginning of
your reasoning.

Begin the reasoning directly with the engineering analysis.

Example:

"We first need to determine the force required to climb the incline."

Then continue with calculations and reasoning.

Use proper LaTeX inside the reasoning whenever appropriate.

============================================================
FINAL ANSWER
============================================================

The final answer should contain, when appropriate:

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

For engineering calculations, show enough work that the user can
verify the result.

Do not unnecessarily repeat the entire engineering process in the
final answer if it has already been shown separately.

============================================================
REAL-WORLD LOSSES
============================================================

When relevant, explicitly consider:

- Rolling resistance
- Bearing losses
- Gearbox losses
- Motor efficiency
- Motor controller losses
- Wiring losses
- Battery internal resistance
- Voltage sag
- Aerodynamic drag
- Tire deformation
- Starting torque
- Acceleration
- Uneven terrain
- Thermal limits

Do not hide all losses behind a single efficiency number when the
user specifically asks about real-world performance.

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
# ADD CURRENT SETTINGS TO SYSTEM PROMPT
# ============================================================

system_prompt += f"""

============================================================
CURRENT USER SETTINGS
============================================================

Response length: {response_length}
Response style: {style}
Explanation level: {explanation_level}
Creativity: {creativity}
Units: {units}

Engineering priorities:

Cost priority: {cost_priority}
Performance priority: {performance_priority}
Reliability priority: {reliability_priority}
Safety priority: {safety_priority}

Reasoning effort: {reasoning_effort}
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

        render_markdown(prompt)

    # --------------------------------------------------------
    # LIMITED HISTORY
    # --------------------------------------------------------

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

                    max_completion_tokens=MAX_TOKENS,

                    stream=True,
                )

            except Exception as e:

                st.error(
                    f"API request failed: "
                    f"{type(e).__name__}: {e}"
                )

                st.session_state.messages.pop()

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
                        clean_latex(reasoning_text)
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
                        clean_latex(answer_text)
                    )

            # ------------------------------------------------
            # REASONING FALLBACK
            # ------------------------------------------------

            if not reasoning_text:

                thinking_placeholder.markdown(
                    "*No separate reasoning was returned by the model.*"
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

                        max_completion_tokens=MAX_TOKENS,
                    )

                except Exception as e:

                    st.error(
                        f"API request failed: "
                        f"{type(e).__name__}: {e}"
                    )

                    st.session_state.messages.pop()

                    st.stop()

            # ------------------------------------------------
            # GET ANSWER
            # ------------------------------------------------

            answer_text = (
                response
                .choices[0]
                .message
                .content
                or ""
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

                    render_markdown(
                        reasoning_text
                    )

                else:

                    st.markdown(
                        "*No reasoning was returned.*"
                    )

            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            render_markdown(answer_text)

            # ------------------------------------------------
            # SAVE ANSWER
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                }
            )