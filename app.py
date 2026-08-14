import streamlit as st
from test_codes.main import question

if "x" not in st.session_state:
    st.session_state.x = 0


def submit():
    st.session_state.x += 1


st.title("RAG-Based AI")
with st.sidebar:
    st.subheader("Settings")
    response_style = st.radio("How are you want the ai to respond?", ("Talkative", "Neutral", "Concise"))
    creativity = st.slider("How creative do you want the ai to be?", 0.0, 1.0, 0.5, 0.01)

prompt = st.chat_input("Enter your question here: ")
response = question(prompt) if prompt else ("you haven't asked a question yet.")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("AI"):
        st.write(response)