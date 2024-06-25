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
from streamlit_app.cards import ItemDisplay, radiobutton_options
from streamlit_app.backend_ops import DataServer
from scenarios.sberbank_consultant import SberbankConsultant
from api.credit_interset_calculator import InterestCalculator


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

if __name__ == '__main__':


    scenario_router = ScenarioRouter()
    st.sidebar.title("Настройки")
    st.title("Прототип")

    # Clear the conversation using a sidebar button for better accessibility
    if st.sidebar.button("Очистить разговор"):
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
        sql_result_ix = 0  # int
        for chat in st.session_state['chat_history']:
            if chat['role'] in ('assistant', 'user'):
                with st.chat_message(chat['role']):
                    st.markdown(chat['content'])
            elif chat['role'] == 'html':
                items = chat.get('items')

                # repeated code
                top_item = items[0]
                top_item_price = top_item.get('price')
                calculator = InterestCalculator()
                duration_2_terms = {}
                for month_duration in radiobutton_options.values():
                    loan_terms = calculator.gpt4o(top_item_price, month_duration)
                    duration_2_terms[month_duration] = loan_terms

                # display here
                item_display = ItemDisplay(items, duration_2_terms=duration_2_terms, sql_result_ix=sql_result_ix)
                sql_result_ix += 1
                lgc = max(0, len(items) - 1)
                item_display.display_grid(lower_grid_cols=lgc)

    if prompt := st.chat_input("Enter you question here"):
        logger.critical(f'context checkpoint 1: {st.session_state.context}')
        st.chat_message("user").markdown(prompt)
        # append before any processing so that the LLM call can be made with the chat history only
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        if prompt:
            if (not 'scenario_name' in st.session_state.context) or (st.session_state.context.get('scenario_name') == 'just_chatting') or (st.session_state.context.get('current_state') == 'exit'):
                logger.critical(f'context before scenario_router.route(): {st.session_state.context }')
                # execute once per scenario_name / scenraio_object
                logger.critical('We must see this only either at the specific scenario start (once) OR after every message in just_chatting')
                selected_route_str = scenario_router.route(
                    user_query=prompt,
                    chat_history=st.session_state.chat_history,
                    stop=['<|eot_id|>'],
                    grammar_path='/home/amstel/llm/src/grammars/scenario_router.gbnf'
                )
                st.session_state.context['scenario_name'] = selected_route_str
                logger.critical(f'context after scenario_router.route(): {st.session_state.context}')
            if st.session_state.context['scenario_name'].startswith('shopping_assistant_'):
                # initial
                st.session_state.scenario_object = ShoppingAssistantScenario(scenario_name = st.session_state.context['scenario_name'])
                st.session_state.context['current_step'] = 'verify'
            elif st.session_state.context['scenario_name'] == 'sberbank_consultant':
                st.session_state.scenario_object = SberbankConsultant()
            elif st.session_state.context['scenario_name'] == 'just_chatting':
                st.session_state.scenario_object = JustChattingScenario()  # stateless (no current_step) by design

            # universal scenario logic
            data, context = st.session_state.scenario_object.handle(user_query=prompt, chat_history=st.session_state.chat_history, context=st.session_state.context)
            st.session_state.context = context
            assert 'scenario_name' in st.session_state.context
            if st.session_state.context.get('current_step') in ('sql','exit') and 'sql_items' in st.session_state:
                st.session_state.pop('sql_items')
            logger.critical(f'context after scenario_object.handle(): {st.session_state.context }')
            if isinstance(data, str):
                response_text = data
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)
            elif isinstance(data, pd.DataFrame):
                # sql results - show table
                # todo: get product_type_name from context variables
                data_server = DataServer(product_type_name='washing_machine')  # must pass this from scenario
                assert 'name' in data.columns
                assert data.shape[0] > 0
                logger.debug(data.shape)
                if 'sql_items' not in st.session_state:
                    items = data_server.collect_data(data['name'])
                    items = items[:4]
                    st.session_state['sql_items'] = items
                    st.session_state.chat_history.append({"role": "html", "items": items})
                items = st.session_state['sql_items']
                # show loan terms for the top option
                top_item = items[0]
                top_item_price = top_item.get('price')
                calculator = InterestCalculator()
                duration_2_terms = {}
                for month_duration in radiobutton_options.values():
                    loan_terms = calculator.gpt4o(top_item_price, month_duration)
                    duration_2_terms[month_duration] = loan_terms

                logger.debug(len(items))
                item_display = ItemDisplay(items, duration_2_terms=duration_2_terms, sql_result_ix=sql_result_ix)
                lgc = max(0, len(items)-1)
                item_display.display_grid(lower_grid_cols=lgc)



            # 10 06 temporarily disable
            # if 'current_step' in st.session_state.context and st.session_state.context['current_step'] == 'exit':
            #     logger.debug(f'current_step -- {st.session_state.context["current_step"]}')
            #     st.session_state.context.pop('scenario_name')  # when exited scenario, nullify
            #     st.session_state.context.pop('current_step')  # when exited scenario, nullify
            #     st.session_state.scenario_object = None



