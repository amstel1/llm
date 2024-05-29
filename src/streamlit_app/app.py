import pandas as pd
from loguru import logger
import streamlit as st
import sys
sys.path.append('/home/amstel/llm/src')
import requests
from typing import List, Dict
from general_llm.llm_endpoint import call_generation_api, call_generate_from_history_api
from scenarios.scenario_router import ScenarioRouter
from scenarios.shopping_assistant import ShoppingAssistantScenario
from scenarios.just_chatting import JustChattingScenario


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


    scenario_router = ScenarioRouter()
    st.sidebar.title("Chat Settings")
    st.title("Langchain Chat App")

    # Clear the conversation using a sidebar button for better accessibility
    if st.sidebar.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.context = {
            # 'scenario': "",
            # 'current_step': "",
            'previous_steps': [],
        }

    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        st.session_state['context'] = {
            # 'scenario': "",
            # 'current_step': "",
            'previous_steps': [],
        }

    # Creating a container for chat history to improve alignment and appearance
    with st.container():
        for chat in st.session_state['chat_history']:
            with st.chat_message(chat['role']):
                st.markdown(chat['content'])


    if prompt := st.chat_input("Enter you question here"):
        logger.critical(f'context checkpoint 1: {st.session_state.context}')
        st.chat_message("user").markdown(prompt)
        # append before any processing so that the LLM call can be made with the chat history only
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        if prompt:
            if (not 'scenario_name' in st.session_state.context) or (st.session_state.context.get('scenario_name') == 'just_chatting'):
                logger.critical(f'context before scenario_router.route(): {st.session_state.context }')
                # execute once per scenario_name / scenraio_object
                logger.critical('We must see this only either at the specific scenario start (once) OR after every message in just_chatting')
                selected_route_str = scenario_router.route(
                    user_query=prompt, stop=['<|eot_id|>'],
                    grammar_path='/home/amstel/llm/src/grammars/scenario_router.gbnf'
                )
                st.session_state.context['scenario_name'] = selected_route_str
                logger.critical(f'context after scenario_router.route(): {st.session_state.context}')
            if st.session_state.context['scenario_name'] == 'shopping_assistant_washing_machine' and 'current_step' not in st.session_state.context:
                # initial
                st.session_state.scenario_object = ShoppingAssistantScenario()
                st.session_state.context['current_step'] = 'verify'
            elif st.session_state.context['scenario_name'] == 'just_chatting':
                st.session_state.scenario_object = JustChattingScenario()  # stateless (no current_step) by design

            # universal scenario logic
            data, context = st.session_state.scenario_object.handle(user_query=prompt, chat_history=st.session_state.chat_history, context=st.session_state.context)
            st.session_state.context = context
            assert 'scenario_name' in st.session_state.context
            logger.critical(f'context after scenario_object.handle(): {st.session_state.context }')
            if isinstance(data, str):
                response_text = data
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)
            elif isinstance(data, pd.DataFrame):
                # sql results - show table
                st.markdown(data)

            if 'current_step' in st.session_state.context and st.session_state.context['current_step'] == 'exit':
                logger.debug(f'current_step -- {st.session_state.context["current_step"]}')
                st.session_state.context.pop('scenario_name')  # when exited scenario, nullify
                st.session_state.context.pop('current_step')  # when exited scenario, nullify
                st.session_state.scenario_object = None


            # response_text = call_generate_from_history_api(prompt, st.session_state.chat_history)  # just chatting
            # st.session_state.chat_history.append({"role": "assistant", "content": response_text})



