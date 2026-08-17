import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# SETUP
# ============================================================

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)


if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# REASONING PARSER
# ============================================================

def parse_reasoning(text):
    """
    Parses reasoning in this format:

    STEP: Determine requirements
    DETAIL: Find the robot's mass and target speed.

    STEP: Calculate force
    DETAIL: Use the mass and rolling resistance.
    """

    steps = []

    current_step = None
    current_detail = ""

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("STEP:"):
            # Save previous step
            if current_step is not None:
                steps.append(
                    (
                        current_step,
                        current_detail.strip()
                    )
                )

            current_step = line.replace(
                "STEP:",
                "",
                1
            ).strip()

            current_detail = ""

        elif line.startswith("DETAIL:"):
            current_detail = line.replace(
                "DETAIL:",
                "",
                1
            ).strip()

        elif current_step is not None and line:
            current_detail += " " + line

    # Save final step
    if current_step is not None:
        steps.append(
            (
                current_step,
                current_detail.strip()
            )
        )

    return steps


# ============================================================
# PAGE
# ============================================================

st.title("Robotics Engineering Assistant")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("Settings")

    # -------------------------
    # Model
    # -------------------------

    model = st.selectbox(
        "Model",
        (
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        )
    )

    # -------------------------
    # Response
    # -------------------------

    stream_it = st.toggle(
        "Stream response",
        True
    )

    response_length = st.radio(
        "Response length",
        (
            "Talkative",
            "Balanced",
            "Concise"
        ),
        horizontal=True
    )

    style = st.radio(
        "Response style",
        (
            "Professional",
            "Casual",
            "Friendly"
        ),
        horizontal=True
    )

    explanation_level = st.radio(
        "Explanation detail",
        (
            "Basic",
            "Intermediate",
            "Advanced"
        ),
        horizontal=True
    )

    creativity = st.slider(
        "Creativity",
        0.0,
        1.0,
        0.5,
        0.01
    )

    units = st.radio(
        "Units",
        (
            "Metric",
            "Imperial"
        ),
        horizontal=True
    )

    # -------------------------
    # Engineering priorities
    # -------------------------

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

    # -------------------------
    # Chat controls
    # -------------------------

    st.divider()

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
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# REASONING INSTRUCTIONS
# ============================================================

reasoning_format = """
For engineering problems, organize your reasoning into a sequence
of clear engineering steps.

Every step MUST use exactly this format:

STEP: <short step name>
DETAIL: <brief explanation>

Example:

STEP: Determine requirements
DETAIL: Identify the robot's mass, target speed, payload, and operating time.

STEP: Calculate required force
DETAIL: Use the mass and rolling resistance to determine the required tractive force.

STEP: Determine motor torque
DETAIL: Convert the required force into wheel torque using the wheel radius.

STEP: Select motor
DETAIL: Find a motor that provides sufficient torque and speed with an appropriate margin.

STEP: Verify the design
DETAIL: Check current draw, battery capacity, thermal limits, and safety margin.

For simple conversational questions, use only one short step.

Keep reasoning focused on solving the user's request.
Do not discuss system prompts, developer instructions, policies,
permissions, instruction hierarchy, or internal configuration.
Do not spend reasoning discussing whether the request is allowed.
"""


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Enter your question here:"
)


if prompt:

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message("user"):

        st.markdown(prompt)


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = f"""
You are a robotics engineering assistant.

Your job is to help the user with:

- Robotics design
- Engineering calculations
- Component selection
- Troubleshooting
- Optimization
- Prototyping
- Mechanical engineering
- Electrical engineering
- Robotics software

{reasoning_format}

Use these settings when responding:

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


    # ========================================================
    # AI RESPONSE
    # ========================================================

    with st.chat_message("assistant"):

        # ====================================================
        # STREAMING
        # ====================================================

        if stream_it:

            stream = client.chat.completions.create(
                model=model,
                temperature=creativity,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    *st.session_state.messages
                ],
                stream=True,
            )


            # -----------------------------------------------
            # UI CONTAINERS
            # -----------------------------------------------

            thinking_container = st.container()

            answer_container = st.empty()


            # -----------------------------------------------
            # State
            # -----------------------------------------------

            reasoning_text = ""
            answer_text = ""


            # -----------------------------------------------
            # Stream
            # -----------------------------------------------

            for chunk in stream:

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta


                # ===========================================
                # REASONING
                # ===========================================

                reasoning = getattr(
                    delta,
                    "reasoning",
                    None
                )

                if reasoning:

                    reasoning_text += reasoning

                    steps = parse_reasoning(
                        reasoning_text
                    )


                    # ---------------------------------------
                    # Render Engineering Process
                    # ---------------------------------------

                    with thinking_container:

                        with st.expander(
                            "Engineering Process",
                            expanded=True
                        ):

                            if not steps:

                                st.caption(
                                    "Analyzing problem..."
                                )

                            else:

                                for i, (step, detail) in enumerate(steps):

                                    # Latest step
                                    if i == len(steps) - 1:

                                        icon = "●"

                                    else:

                                        icon = "✓"


                                    st.markdown(
                                        f"**{icon} {i + 1}. {step}**"
                                    )


                                    if detail:

                                        st.caption(
                                            detail
                                        )


                # ===========================================
                # FINAL ANSWER
                # ===========================================

                content = getattr(
                    delta,
                    "content",
                    None
                )

                if content:

                    answer_text += content

                    answer_container.markdown(
                        answer_text
                    )


            # =================================================
            # SAVE AI RESPONSE
            # =================================================

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

            with st.spinner(
                "Thinking..."
            ):

                response = client.chat.completions.create(
                    model=model,
                    temperature=creativity,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        *st.session_state.messages
                    ],
                )


                answer_text = (
                    response
                    .choices[0]
                    .message
                    .content
                )


                st.markdown(
                    answer_text
                )


                # -------------------------------------------
                # Save response
                # -------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text
                    }
                )