import streamlit as st
import requests
from loguru import logger

API_ENDPOINT = "http://localhost:8000/process_text"

def call_api(input_text):
    response = requests.post(API_ENDPOINT, json={"question": input_text})
    return response.json()

st.sidebar.title("Chat Settings")
st.title("Langchain Chat App")

# Clear the conversation using a sidebar button for better accessibility
if st.sidebar.button("Clear Conversation"):
    st.session_state.chat_history = []

# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Creating a container for chat history to improve alignment and appearance
with st.container():
    for chat in st.session_state['chat_history']:
        # Use markdown with different styles for sender and message
        st.markdown(f"**{chat['sender']}**: {chat['message']}")

# Use a form for input and submit button, ensuring they are aligned at the bottom
with st.container():
    with st.form(key='user_input_form'):
        user_message = st.text_input("Enter your question:", key="input")
        submit_button = st.form_submit_button("Send")

# Logic to handle sending message
if submit_button:
    if not user_message.strip():
        st.warning("Please enter some text to send.")
    else:
        with st.spinner("Waiting for the response..."):
            response = call_api(user_message)
            if response:
                logger.info(response)
                response_text = response['answer']
                # Append to chat history and clear input
                st.session_state.chat_history.append({"sender": "You", "message": user_message})
                st.session_state.chat_history.append({"sender": "Bot", "message": response_text})

                # Display the Bot response separately
                st.markdown(f"**Bot**: {response_text}")