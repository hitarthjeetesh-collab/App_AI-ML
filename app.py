import streamlit as st
import os
import re

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

max_tokens = 6000


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
# LATEX / MARKDOWN HELPERS
# ============================================================

def clean_latex(text):
    """
    Clean common malformed LaTeX formatting produced by models.

    This does NOT modify normal Markdown or properly formatted LaTeX.
    """

    if not text:
        return text

    # --------------------------------------------------------
    # Convert common square-bracket equation formatting
    #
    # Example:
    #
    # [ F_{total} = F_{grade} + F_{rr} ]
    #
    # becomes:
    #
    # $$
    # F_{total} = F_{grade} + F_{rr}
    # $$
    # --------------------------------------------------------

    def replace_square_equation(match):
        equation = match.group(1).strip()

        # Only convert if it looks mathematical.
        math_indicators = [
            "\\",
            "=",
            "^",
            "_",
            "\\frac",
            "\\sin",
            "\\cos",
            "\\tan",
            "\\times",
            "\\approx",
            "∑",
            "√",
            "π",
        ]

        if any(indicator in equation for indicator in math_indicators):
            return f"\n\n$$\n{equation}\n$$\n\n"

        return match.group(0)

    # Only process single-line square bracket equations.
    text = re.sub(
        r"\[\s*([^\[\]\n]{3,500})\s*\]",
        replace_square_equation,
        text,
    )

    return text


def render_markdown(text):
    """
    Render model output as Markdown with Streamlit's
    built-in LaTeX support.
    """

    if not text:
        return

    cleaned = clean_latex(text)

    st.markdown(
        cleaned
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

IMPORTANT:

Use REAL LaTeX whenever displaying mathematical equations.

The application uses Streamlit's Markdown renderer, which supports
LaTeX.

Use:

Inline mathematics:

$F = ma$

Use display mathematics for important equations:

$$
F = ma
$$

For multi-line equations:

$$
\\begin{{aligned}}
F_{{grade}} &= mg\\sin(\\theta) \\\\
F_{{rr}} &= C_{{rr}}mg\\cos(\\theta) \\\\
F_{{total}} &= F_{{grade}} + F_{{rr}}
\\end{{aligned}}
$$

Use LaTeX commands such as:

- \\frac{{a}}{{b}}
- \\sin(\\theta)
- \\cos(\\theta)
- \\sqrt{{x}}
- \\times
- \\approx
- \\omega
- \\theta
- \\eta
- \\text{{N}}
- \\text{{W}}
- \\text{{kg}}
- \\text{{m/s}}

Examples:

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

$$
C = \\frac{{E}}{{V}}
$$


============================================================
VERY IMPORTANT LATEX RULES
============================================================

NEVER write mathematical equations using square brackets.

BAD:

[ F_{{total}} = F_{{grade}} + F_{{rr}} ]

BAD:

[ F = ma ]

GOOD:

$$
F_{{total}} = F_{{grade}} + F_{{rr}}
$$

GOOD:

$$
F = ma
$$

Do NOT use square brackets as an alternative to LaTeX delimiters.

Do NOT write raw LaTeX outside a Markdown math delimiter.

Do NOT put equations inside Markdown code blocks.

Do NOT escape the LaTeX backslashes in the final response.

The final response should contain actual Markdown-compatible LaTeX.

For example, output:

$$
F_{{grade}} = mg\\sin(\\theta)
$$

rather than:

[ F_{{grade}} = mg\\sin(\\theta) ]

For calculations, show the equation first and then substitute
numbers into it.

Example:

$$
F_{{grade}} = mg\\sin(10^\\circ)
$$

$$
F_{{grade}}
=
25 \\times 9.81 \\times \\sin(10^\\circ)
\\approx 42.5\\ \\text{{N}}
$$


============================================================
ENGINEERING PROCESS
============================================================

The application displays the model's reasoning separately inside
an "Engineering Process" box.

Do NOT write the heading "Engineering Process" yourself.

Do NOT put the phrase "Engineering Process" at the beginning
of your reasoning.

Begin the reasoning directly with the engineering analysis.

For example:

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

        render_markdown(
            prompt
        )

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

                    max_completion_tokens=max_tokens,

                    stream=True,
                )

            except Exception as e:

                st.error(
                    f"API request failed: {type(e).__name__}: {e}"
                )

                # Remove failed user message
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
            # ENGINEERING PROCESS FALLBACK
            # ------------------------------------------------

            if not reasoning_text:

                thinking_placeholder.markdown(
                    "*No separate reasoning was returned by the model.*"
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

                        max_completion_tokens=max_tokens,
                    )

                except Exception as e:

                    st.error(
                        f"API request failed: {type(e).__name__}: {e}"
                    )

                    # Remove failed user message
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
            # SHOW ANSWER
            # ------------------------------------------------

            render_markdown(
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