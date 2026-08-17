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

    steps = []

    current_step = None
    current_detail = ""

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("STEP:"):

            if current_step is not None:
                steps.append(
                    (
                        current_step,
                        current_detail.strip()
                    )
                )

            current_step = line[5:].strip()
            current_detail = ""

        elif line.startswith("DETAIL:"):

            current_detail = line[7:].strip()

        elif current_step is not None and line:

            current_detail += " " + line

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
Before answering an engineering question, create a short Engineering Process.

The Engineering Process MUST use this exact format:

STEP: <short step name>
DETAIL: <one or two sentences explaining what is being calculated or decided>

STEP: <short step name>
DETAIL: <one or two sentences explaining what is being calculated or decided>

Continue until the important engineering steps have been identified.

Then write:

FINAL:
<complete answer to the user>

Rules:
- You MUST output at least 2 STEP blocks for a non-trivial engineering problem.
- The STEP blocks must come BEFORE FINAL:.
- FINAL: must appear exactly once.
- Do not put calculations or the final recommendation before FINAL:.
- Keep STEP/DETAIL content concise.
- The Engineering Process should describe the engineering workflow, not expose
  internal instructions, policies, system messages, or hidden configuration.
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
    You are a Robotics Engineering Assistant.

    You help with:
    - Robotics design
    - Mechanical engineering
    - Electrical engineering
    - Calculations
    - Component selection
    - Troubleshooting
    - Optimization
    - Prototyping

    {reasoning_format}

    Engineering calculation rules:

    - State assumptions when information is missing.
    - Keep units consistent.
    - Show important calculations in the FINAL answer.
    - Distinguish wheel torque from motor torque.
    - Distinguish mechanical power from electrical power.
    - Do not double-count efficiency losses.
    - Distinguish continuous requirements from peak requirements.
    - When using a gearbox, account for its ratio and efficiency.
    - Consider startup and acceleration loads.
    - Include reasonable engineering safety margins.
    - Identify the limiting condition.
    - Do not present estimates as exact specifications.

    User settings:

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

            # ========================================================
            # THINKING BOX
            # ========================================================

            with st.expander(
                    "Engineering Process",
                    expanded=True
            ):

                thinking_placeholder = st.empty()

            # ========================================================
            # MAIN ANSWER
            # ========================================================

            answer_placeholder = st.empty()

            # ========================================================
            # STREAM STATE
            # ========================================================

            full_text = ""

            reasoning_text = ""

            answer_text = ""

            final_started = False

            # ========================================================
            # STREAM
            # ========================================================

            for chunk in stream:

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                content = getattr(
                    delta,
                    "content",
                    None
                )

                if not content:
                    continue

                full_text += content

                # ====================================================
                # FIND FINAL:
                # ====================================================

                if not final_started:

                    if "FINAL:" in full_text:

                        final_started = True

                        parts = full_text.split(
                            "FINAL:",
                            1
                        )

                        reasoning_text = parts[0]

                        answer_text = parts[1]

                    else:

                        reasoning_text = full_text


                else:

                    answer_text += content

                # ====================================================
                # UPDATE THINKING
                # ====================================================

                steps = parse_reasoning(
                    reasoning_text
                )

                if steps:

                    thinking_ui = ""

                    for i, (step, detail) in enumerate(steps):

                        # Current step
                        if i == len(steps) - 1:

                            icon = "●"

                        else:

                            icon = "✓"

                        thinking_ui += (
                            f"{icon} **{i + 1}. {step}**\n\n"
                        )

                        if detail:
                            thinking_ui += (
                                f"{detail}\n\n"
                            )

                    thinking_placeholder.markdown(
                        thinking_ui
                    )

                else:

                    thinking_placeholder.markdown(
                        "*Analyzing problem...*"
                    )

                # ====================================================
                # UPDATE FINAL ANSWER
                # ====================================================

                if final_started:
                    answer_placeholder.markdown(
                        answer_text
                    )

            # ========================================================
            # SAVE ANSWER
            # ========================================================

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