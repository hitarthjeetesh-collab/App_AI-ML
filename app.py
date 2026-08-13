import streamlit as st
import chromadb

from main.py import add_memory

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("memories")

if "x" not in st.session_state:
    st.session_state.x = 0


def submit():
    st.session_state.x += 1


st.title("RAG-Based AI App")

st.subheader("Details")
st.write("Model: llama-3.3-70b-versatile")
st.write("Memory: ChromaDB")

st.subheader("Instructions")
st.write("Ask a question and the AI will answer based on the stored memories.")
with st.sidebar:
    st.subheader("Settings")
    response_style = st.radio("How are you want the ai to respond?", ("Talkative", "Neutral", "Concise"))
    creativity = st.slider("How creative do you want the ai to be?", 0.0, 1.0, 0.5, 0.01)



with st.sidebar:
    st.subheader("Memory")

    data = collection.get()

    if data["ids"]:
        st.dataframe({
            "ID": data["ids"],
            "Memory": data["documents"],
        }, hide_index=True)
    else:
        st.write("No memories stored.")
prompt = st.text_input("Enter your question here: ")

st.button("submit", key="submit", on_click=submit)
