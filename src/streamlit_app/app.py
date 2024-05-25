from loguru import logger
import streamlit as st
# import sys
# sys.path.append('/home/amstel/llm/src')
import requests
API_ENDPOINT = "http://localhost:8000/process_text"
from typing import List, Dict

def call_api(input_text: str, chat_history: List[Dict[str, str]]=[]):
    """
    input_text: str
    chat_history: List[Dict[str, str]]
    """
    response = requests.post(
        API_ENDPOINT,
        json={"question": input_text, "chat_history": chat_history}
    )
    r = response.json().get('choices')[0].get('text')
    return {"answer": r}


def create_preview_card(
        url="https://shop.by/stiralnye_mashiny/lg_f2j3ws2w/",
        title="Стиральная машина LG F2J3WS2W",
        image_url="https://shop.by/images/lg_f2j3ws2w_1.webp",
        description="Custom description"
):
    """Function to create a website preview card in Streamlit."""
    card_html = f"""
    <div style="display: flex; flex-direction: row; align-items: flex-start; gap: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 8px; box-shadow: 0 4px 6px 0 rgba(0,0,0,0.1);">
        <a href="{url}" target="_blank" style="text-decoration: none; color: #000;">
            <img src="{image_url}" alt="{title}" style="width: 120px; height: 120px; border-radius: 8px; object-fit: cover;">
        </a>
        <div style="flex-grow: 1;">
            <h4><a href="{url}" target="_blank" style="text-decoration: none; color: #000;">{title}</a></h4>
            <p>{description}</p>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


if __name__ == '__main__':

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
        if prompt:
            response = call_api(prompt, st.session_state.chat_history)
            response_text = response['answer']
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            logger.debug(st.session_state.chat_history)
            with st.chat_message("assistant"):
                st.markdown(response_text)

        # link to the selected product / products
        # todo: routing between scenarios
        # todo: change display_website_preview to parsing microdata in realtime
        # todo: display the results once, nice grid, from parameters.
        # todo: выделить в сценарий - sql to text
        # todo: Переключатель элементов
        # todo: выход из сценария - датафрэйм, сделать бэк который собирает данные (монго)
        # todo: логика сохранения рендеринга (по истории чата)
        # todo: citations
        # todo: inline elements - prefilters - how to create

        # create_preview_card()
