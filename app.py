"""
Simple ChatGPT-like chatbot using a local open-source LLM (via Ollama) and Streamlit.

Run with:  streamlit run app.py
"""

import streamlit as st
import ollama

MODEL_NAME = "llama3.1:8b"
SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. Answer clearly and concisely. "
    "Use the prior conversation to understand follow-up questions. "
    "When a term or acronym is ambiguous, prefer its meaning within "
    "artificial intelligence, machine learning, and LLM/chatbot systems "
    "over other fields (e.g. RAG = Retrieval-Augmented Generation)."
)

st.set_page_config(page_title="Local Chatbot", page_icon="💬")
st.title("💬 Local Chatbot")
st.caption(f"Running on {MODEL_NAME} via Ollama — history is kept for this session only.")

# --- Session-based chat history -------------------------------------------------
# st.session_state is unique per browser session/tab, so each user/session
# gets its own independent conversation history.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if st.sidebar.button("🆕 New session"):
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.rerun()

# --- Render existing history -----------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle new user input --------------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        # The full session history (system + all prior turns) is sent every
        # time, so the model can resolve follow-ups like "explain the second one".
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=st.session_state.messages,
            stream=True,
        )
        for chunk in stream:
            full_response += chunk["message"]["content"]
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
