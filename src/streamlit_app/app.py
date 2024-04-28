import streamlit as st
import requests
from loguru import logger

API_ENDPOINT = "http://localhost:8000/process_text"

def call_api(input_text):
    response = requests.post(API_ENDPOINT, json={"question": input_text, "chat_history": st.session_state.chat_history})
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
        with st.chat_message(chat['role']):
            st.markdown(chat['content'])


if prompt := st.chat_input("Enter you question here"):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    response = call_api(prompt)
    response_text = response['answer']
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    logger.debug(st.session_state.chat_history)
    with st.chat_message("assistant"):
        st.markdown(response_text)
