"""
Streamlit App Example
====================

Simple chat interface for your RAG system.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st

from simple_api import ask_question_with_stages, search_documents

st.title("🔍 Vietnamese RAG System")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question in Vietnamese..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        # Use persistent spinner with dynamic text updates
        with st.spinner("Processing..."):
            # Create a placeholder for dynamic status updates inside the spinner
            status_placeholder = st.empty()

            def update_status(status_text):
                # Update the text inside the spinner area
                status_placeholder.markdown(f"**{status_text}**")

            # Show initial status
            update_status("🚀 Starting RAG process...")

            # Get response with stage updates
            result = ask_question_with_stages(prompt, status_callback=update_status)

            # Clear status placeholder
            status_placeholder.empty()

        # Show final result after spinner disappears
        st.write(result["answer"])

        # Show sources if available
        if result.get("sources"):
            st.subheader("Sources:")
            for i, source in enumerate(result["sources"][:3]):
                with st.expander(f"Source {i+1}: {source.get('filename', 'Unknown')}"):
                    st.write(source.get("content", "")[:300] + "...")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# Sidebar for document search
with st.sidebar:
    st.header("📚 Document Search")
    search_query = st.text_input("Search documents:")

    if search_query:
        docs = search_documents(search_query, top_k=3)
        for i, doc in enumerate(docs):
            with st.expander(f"Document {i+1} (Score: {doc['score']:.3f})"):
                st.write(f"**Source:** {doc['source']}")
                st.write(doc["content"][:200] + "...")
