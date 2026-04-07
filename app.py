import streamlit as st

tabs = st.tabs(["Chat", "Knowledge Base", "Configuration"])

with tabs[0]:
    st.header("Chat")
    st.write("This is the chat interface.")

    user_prompt = st.chat_input("Enter your configuration settings here...",
                                max_upload_size=1024 * 1024 * 5)  # Limit to 5MB

with tabs[1]:
    st.header("Knowledge Base")
    st.write("This is the knowledge base interface.")

    files = st.file_uploader("Upload files to the knowledge base", accept_multiple_files=True,
                             type=["txt", "pdf", "docx"])
    
    if files:
        for file in files:
            pass

with tabs[2]:
    st.header("Configuration")
    st.write("This is the configuration interface.")