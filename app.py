"""
ChatGPT/Claude-style chatbot: Groq-hosted open-source LLM + ChromaDB
vector-store for persistent, browsable chat history across sessions.

Run with:  python run_app.py   (or: streamlit run app.py)
"""

import html
import json
import os
import queue
import random
import re
import threading
import uuid
from datetime import datetime

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. Answer clearly and concisely. "
    "Use the prior conversation to understand follow-up questions. "
    "When a term or acronym is ambiguous, prefer its meaning within "
    "artificial intelligence, machine learning, and LLM/chatbot systems "
    "over other fields (e.g. RAG = Retrieval-Augmented Generation)."
)

STARTER_QUESTION_POOL = [
    "Help me plan my day",
    "Give me 3 productivity tips",
    "Draft a polite email to reschedule a meeting",
    "Explain a random interesting fact",
    "What is RAG (Retrieval-Augmented Generation)?",
    "Explain supervised vs unsupervised learning",
    "Write a Python function to reverse a linked list",
    "Suggest a simple dinner recipe",
    "Give me a 5-minute desk stretch routine",
    "Help me write a short thank-you note",
]

TYPING_WORDS = [
    "Thinking", "Processing", "Crunching", "Noodling", "Pondering",
    "Percolating", "Puttering", "Mulling", "Simmering", "Digesting",
    "Untangling", "Assembling", "Contemplating", "Brewing", "Marinating",
    "Ruminating", "Deliberating", "Formulating", "Gathering thoughts",
    "Sifting", "Distilling", "Sketching it out", "Weighing options",
]

st.set_page_config(page_title="Local Chatbot", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
    .chat-row { display: flex; margin: 10px 0; }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .bubble {
        max-width: 85%; padding: 10px 16px; border-radius: 16px;
        line-height: 1.5; word-wrap: break-word;
    }
    .bubble.user {
        background-color: #2b6cb0; color: white; border-bottom-right-radius: 4px;
    }
    .bubble.assistant {
        background-color: rgba(128,128,128,0.18); color: inherit;
        border-bottom-left-radius: 4px;
    }
    .bubble pre {
        background-color: rgba(0,0,0,0.35); padding: 10px; border-radius: 8px;
        overflow-x: auto; white-space: pre-wrap;
    }
    .bubble code { font-family: monospace; }

    .typing-dots {
        display: inline-flex; gap: 4px; margin-right: 8px; vertical-align: middle;
    }
    .typing-dots span {
        width: 6px; height: 6px; border-radius: 50%;
        background-color: currentColor;
        animation: typing-bounce 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(1) { animation-delay: 0s; }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing-bounce {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40% { transform: scale(1); opacity: 1; }
    }
    .typing-word { font-style: italic; opacity: 0.8; }

    /* Keep the sidebar collapse arrow visible at all times, not just on hover */
    [data-testid="stSidebarCollapseButton"] {
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Hide Print / Record screen from the top-right menu for now, while
       keeping the theme switcher (System/Light/Dark) above them. */
    [data-testid="stMainMenuItem-print"],
    [data-testid="stMainMenuItem-recordScreencast"] {
        display: none !important;
    }
    [data-testid="stMainMenuDivider"] {
        display: none !important;
    }

    /* Never let the sidebar be dragged so narrow that its content becomes
       unusable — hold a sensible floor instead of chasing every width. */
    section[data-testid="stSidebar"] {
        min-width: 260px !important;
    }

    /* Keep each chat row (title + menu button) on one line.
       Scoped to the chat-row container only — these used to be sidebar-wide
       and leaked into the rename/delete-confirm rows' own columns, breaking
       their icon buttons (nth-of-type(1)'s display:block + overflow:hidden
       clipped the emoji since those columns aren't a title+menu pair). */
    div[class*="st-key-chatrow-"] div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
    }
    div[class*="st-key-chatrow-"] div[data-testid="stColumn"]:nth-of-type(1) {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }
    div[class*="st-key-chatrow-"] div[data-testid="stColumn"]:nth-of-type(1) button {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        display: block !important;
    }
    div[class*="st-key-chatrow-"] div[data-testid="stColumn"]:nth-of-type(2) {
        flex: 0 0 auto !important;
        min-width: 0 !important;
        width: auto !important;
    }
    div[class*="st-key-chatrow-"] div[data-testid="stColumn"]:nth-of-type(2) button {
        min-width: 36px !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    /* The "⋮" menu button gets an auto-added caret icon; hide it so the
       button is just the plain three dots, like a real kebab menu. */
    div[class*="st-key-chatrow-"] div[data-testid="stColumn"]:nth-of-type(2) [data-testid="stIconMaterial"] {
        display: none !important;
    }
    /* Center the delete-confirmation modal vertically, like ChatGPT/Claude —
       Streamlit's default dialog container aligns to the top (flex-start)
       instead of the viewport center. */
    div[data-testid="stDialog"] {
        align-items: center !important;
    }
    /* Light polish on the popover menu itself */
    div[data-testid="stPopoverBody"] {
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15) !important;
        padding: 6px !important;
        min-width: 160px !important;
    }
    /* Make the menu a tight, compact list instead of stretched-out blocks
       (Streamlit's default ~1rem gap between elements looked broken here).
       The delete button sits in its own nested container/vertical-block
       (needed for the red-color CSS hook below), so every layer of
       spacing has to be zeroed out, not just the outer one. */
    div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
        gap: 0.1rem !important;
    }
    div[data-testid="stPopoverBody"] [data-testid="stElementContainer"],
    div[data-testid="stPopoverBody"] [data-testid="stLayoutWrapper"],
    div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"],
    div[data-testid="stPopoverBody"] [data-testid="stMarkdown"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    /* Menu rows read as a plain list, not boxed buttons */
    div[data-testid="stPopoverBody"] button {
        border: none !important;
        background: transparent !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 6px 8px !important;
        border-radius: 6px !important;
    }
    div[data-testid="stPopoverBody"] button:hover {
        background: rgba(128,128,128,0.15) !important;
    }
    /* Destructive action styled red, like ChatGPT/Claude */
    div[class*="st-key-delete-btn-"] button {
        color: #e5484d !important;
    }
    .menu-divider {
        border: none; border-top: 1px solid rgba(128,128,128,0.25);
        margin: 0 !important;
    }

    /* Merge the title button + "⋮" menu button into a single row/box,
       like a real chat list item, instead of two separate boxes. */
    div[class*="st-key-chatrow-"] {
        border: 1px solid rgba(128,128,128,0.3) !important;
        border-radius: 8px !important;
        margin-bottom: 6px !important;
        overflow: hidden !important;
    }
    div[class*="st-key-chatrow-"] button {
        border: none !important;
        background: transparent !important;
        border-radius: 0 !important;
    }
    div[class*="st-key-chatrow-"]:hover {
        background: rgba(128,128,128,0.08) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Persistent, vector-DB backed chat store --------------------------------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="chat_history_db")
    return client.get_or_create_collection(name="chat_sessions")


collection = get_collection()


def save_session(session_id, messages, title):
    non_system = [m for m in messages if m["role"] != "system"]
    if not non_system:
        return
    doc_text = "\n".join(f'{m["role"]}: {m["content"]}' for m in non_system)
    collection.upsert(
        ids=[session_id],
        documents=[doc_text],
        metadatas=[{
            "title": title,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "messages": json.dumps(messages),
        }],
    )


def load_all_sessions():
    data = collection.get()
    sessions = [
        {"id": sid, "title": meta["title"], "created_at": meta["created_at"]}
        for sid, meta in zip(data["ids"], data["metadatas"])
    ]
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    return sessions


def load_session_messages(session_id):
    data = collection.get(ids=[session_id])
    if not data["metadatas"]:
        return None
    return json.loads(data["metadatas"][0]["messages"])


def delete_session(session_id):
    collection.delete(ids=[session_id])


def rename_session(session_id, new_title):
    messages = load_session_messages(session_id)
    if messages is not None:
        save_session(session_id, messages, new_title)


# --- Groq client --------------------------------------------------------------
def get_client():
    if not GROQ_API_KEY:
        return None
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def format_content(text):
    """Minimal markdown -> HTML: escape, then render ``` blocks and `code`."""
    escaped = html.escape(text)

    def code_block(m):
        return f'<pre><code>{m.group(1)}</code></pre>'

    escaped = re.sub(r"```(?:\w*\n)?(.*?)```", code_block, escaped, flags=re.DOTALL)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped.replace("\n", "<br>")


def render_bubble(role, html_content):
    st.markdown(
        f'<div class="chat-row {role}"><div class="bubble {role}">{html_content}</div></div>',
        unsafe_allow_html=True,
    )


def typing_indicator_html(word):
    dots = '<span class="typing-dots"><span></span><span></span><span></span></span>'
    return f'{dots}<span class="typing-word">{word}…</span>'


def stream_groq_response(client, model, messages, result_queue):
    try:
        stream = client.chat.completions.create(model=model, messages=messages, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                result_queue.put(("chunk", delta))
        result_queue.put(("done", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


# --- Session state ------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "title" not in st.session_state:
    st.session_state.title = None
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


def start_new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.title = None
    st.session_state.starter_questions = random.sample(STARTER_QUESTION_POOL, 4)


@st.dialog("Delete chat?")
def confirm_delete_dialog(session_id, title, is_active):
    st.write(f'Are you sure you want to delete "{title}"? This can\'t be undone.')
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pending_delete = None
            st.rerun()
    with col2:
        if st.button("Delete", type="primary", use_container_width=True):
            delete_session(session_id)
            if is_active:
                start_new_chat()
            st.session_state.pending_delete = None
            st.rerun()


if "starter_questions" not in st.session_state:
    st.session_state.starter_questions = random.sample(STARTER_QUESTION_POOL, 4)
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None
if "renaming_id" not in st.session_state:
    st.session_state.renaming_id = None


def open_chat(session_id, title):
    st.session_state.session_id = session_id
    st.session_state.messages = load_session_messages(session_id)
    st.session_state.title = title


# --- Sidebar: chat history ----------------------------------------------------
with st.sidebar:
    st.header("💬 Chats")
    if st.button("🆕 New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()
    for s in load_all_sessions():
        label = s["title"] or "Untitled chat"
        is_active = s["id"] == st.session_state.session_id

        # A single stable container per row, regardless of which of the three
        # modes below it's in — without this, Streamlit can leave ghost
        # widgets behind when the row's widget set changes between reruns
        # (e.g. title+menu -> text_input+icons), causing duplicated buttons.
        with st.container(key=f"row-{s['id']}"):
            if st.session_state.renaming_id == s["id"]:
                # Inline rename, like ChatGPT — not nested inside the popover,
                # since a popover doesn't reliably stay open across the rerun
                # triggered by clicking something inside it.
                # Wrapped in a form so pressing Enter in the text box submits
                # it (triggering Save, the first submit button) — a plain
                # text_input reruns on Enter but doesn't fire the button.
                with st.form(key=f"rename_form_{s['id']}", border=False):
                    rcol1, rcol2, rcol3 = st.columns([5, 1, 1])
                    with rcol1:
                        new_title = st.text_input(
                            "Rename chat", value=label, key=f"rename_input_{s['id']}",
                            label_visibility="collapsed",
                        )
                    with rcol2:
                        save_clicked = st.form_submit_button("✔️", help="Save", use_container_width=True)
                    with rcol3:
                        cancel_clicked = st.form_submit_button("❌", help="Cancel", use_container_width=True)

                if save_clicked:
                    final_title = new_title.strip() or label
                    rename_session(s["id"], final_title)
                    if is_active:
                        st.session_state.title = final_title
                    st.session_state.renaming_id = None
                    st.rerun()
                if cancel_clicked:
                    st.session_state.renaming_id = None
                    st.rerun()

            else:
                with st.container(key=f"chatrow-{s['id']}"):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        if st.button(
                            ("🟢 " if is_active else "") + label[:32],
                            key=f"open_{s['id']}",
                            use_container_width=True,
                        ):
                            open_chat(s["id"], s["title"])
                            st.rerun()
                    with col2:
                        pending = st.session_state.pending_delete
                        if pending and pending["id"] == s["id"]:
                            # Don't render the popover for this row while its
                            # delete dialog is open — a popover keeps its own
                            # open/closed state across reruns (it isn't reset
                            # just because the script reran), so it would
                            # otherwise stay visibly open behind the dialog.
                            # Unmounting it here means it comes back fresh
                            # (closed) once the dialog is dismissed.
                            st.button("⋮", key=f"menu_placeholder_{s['id']}", disabled=True, use_container_width=True)
                        else:
                            with st.popover("⋮", use_container_width=True):
                                # A real menu list, like ChatGPT/Claude — more options
                                # (Share, Pin, ...) can be added here later the same way.
                                if st.button("✏️ Rename", key=f"rename_{s['id']}", use_container_width=True):
                                    st.session_state.renaming_id = s["id"]
                                    st.rerun()
                                st.markdown('<hr class="menu-divider">', unsafe_allow_html=True)
                                with st.container(key=f"delete-btn-{s['id']}"):
                                    if st.button("🗑 Delete", key=f"del_{s['id']}", use_container_width=True):
                                        st.session_state.pending_delete = {
                                            "id": s["id"], "title": label, "is_active": is_active,
                                        }
                                        st.rerun()

    if st.session_state.pending_delete:
        pd = st.session_state.pending_delete
        confirm_delete_dialog(pd["id"], pd["title"], pd["is_active"])

has_real_messages = any(m["role"] != "system" for m in st.session_state.messages)

if has_real_messages:
    st.caption("💬 Local Chatbot")
else:
    st.title("👋 Hi! How can I help you today?")

client = get_client()
if client is None:
    st.error(
        "No GROQ_API_KEY found. Create a `.env` file next to app.py with:\n\n"
        "GROQ_API_KEY=your_key_here\n\n"
        "Get a free key at https://console.groq.com/keys, then restart the app."
    )
    st.stop()

# --- Render existing history ---------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    render_bubble(msg["role"], format_content(msg["content"]))

# --- Starter questions on an empty chat -----------------------------------------
if not has_real_messages:
    st.markdown("##### Try one of these:")
    for q in st.session_state.starter_questions:
        if st.button(q, use_container_width=True, key=f"starter_{q}"):
            st.session_state.pending_input = q
            st.rerun()

# --- Handle new user input -------------------------------------------------------
user_input = st.chat_input("Type your message...") or st.session_state.pending_input
st.session_state.pending_input = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    if st.session_state.title is None:
        st.session_state.title = user_input[:40]
    render_bubble("user", format_content(user_input))

    placeholder = st.empty()

    result_queue = queue.Queue()
    thread = threading.Thread(
        target=stream_groq_response,
        args=(client, GROQ_MODEL, st.session_state.messages, result_queue),
        daemon=True,
    )
    thread.start()

    full_response = ""
    error_message = None
    waiting_for_first_token = True
    word_index = 0

    while True:
        try:
            kind, payload = result_queue.get(timeout=0.5 if waiting_for_first_token else None)
        except queue.Empty:
            with placeholder:
                render_bubble("assistant", typing_indicator_html(TYPING_WORDS[word_index % len(TYPING_WORDS)]))
            word_index += 1
            continue

        if kind == "chunk":
            waiting_for_first_token = False
            full_response += payload
            with placeholder:
                render_bubble("assistant", format_content(full_response) + "▌")
        elif kind == "done":
            break
        elif kind == "error":
            error_message = payload
            break

    if error_message:
        with placeholder:
            st.error(f"Groq API error: {error_message}")
    else:
        with placeholder:
            render_bubble("assistant", format_content(full_response))
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_session(st.session_state.session_id, st.session_state.messages, st.session_state.title)
        st.rerun()
