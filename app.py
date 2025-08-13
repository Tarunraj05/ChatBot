import streamlit as st
from connect import build_qa_chain
import re

# Clean the LLM output to remove hidden reasoning
def clean_answer(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^think>.*", "", text, flags=re.IGNORECASE)
    return text.strip()

# Highlight query terms in text
def highlight_text(content, query):
    words = query.split()
    for word in words:
        content = re.sub(f"({re.escape(word)})", r"**\1**", content, flags=re.IGNORECASE)
    return content

# Load QA chain once and cache it
@st.cache_resource
def load_chain():
    return build_qa_chain()

# UI
st.title("📚 Ask Chatbot with Sources")
qa_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Chat input
prompt = st.chat_input("Write your message here")
if prompt:
    # Show user input
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from QA chain
    response = qa_chain.invoke({"query": prompt})
    answer = clean_answer(response["result"])
    sources = response.get("source_documents", [])

    # Show assistant answer
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Show sources with highlighting
    if sources:
        with st.expander("📄 Source Documents"):
            for i, doc in enumerate(sources, start=1):
                snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                snippet = highlight_text(snippet, prompt)
                st.markdown(f"**Source {i}:** `{doc.metadata.get('source', 'Unknown')}`\n\n{snippet}")
