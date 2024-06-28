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
from general_llm.utils import ChatHistory
CHAT_HISTORY_SIZE = 6
from text2sql.prod_sql_to_text import update_sql_statement

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
def stylish_citation_link(url, text):
    st.markdown(
            f"""
                <a href="{url}" target="_blank" style="
                display: inline-block;
                padding: 0.5rem 1rem;
                background-color: rgb(175, 175, 175);
                color: white;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                border-radius: 20px;
                transition: background-color 0.3s ease;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            ">{text}</a>
            """,
            unsafe_allow_html=True
        )

# Example usage


def render_df(df: pd.DataFrame):
    # sql results - show table
    # todo: get product_type_name from context variables
    # logger.warning(f'! important: {st.session_state.context}')
    assert st.session_state.context['sql_schema']  # must always exist
    data_server = DataServer(schema_name=st.session_state.context['sql_schema'])  # must pass this from scenario
    assert 'name' in data.columns
    assert data.shape[0] > 0
    logger.debug(data.shape)
    logger.debug(data.head(4))
    if 'sql_items' not in st.session_state:
        # sql_items must be updated every time sql is executed!
        items = data_server.collect_data(data['name'])
        logger.info(f'collect_data: {items}')
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
    logger.error(items)
    item_display = ItemDisplay(items, duration_2_terms=duration_2_terms, sql_result_ix=sql_result_ix)
    lgc = max(0, len(items) - 1)
    item_display.display_grid(lower_grid_cols=lgc)

if __name__ == '__main__':
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'context' not in st.session_state:
        st.session_state['context'] = {
            # 'scenario': "",
            # 'current_step': "",
            'previous_steps': ["", ""],
        }

    scenario_router = ScenarioRouter()

    st.title("Прототип")

    # Clear the conversation using a sidebar button for better accessibility
    with st.sidebar:
        st.title("Настройки")
        if st.button("Очистить разговор"):
            st.session_state.chat_history = []
            st.session_state.context = {
                # 'scenario': "",
                # 'current_step': "",
                'previous_steps': [None, None,],
            }
        if 'cited_sources' in st.session_state.context and 'citations_lookup' in st.session_state.context:
            logger.error('!!! WE ARE IN CITATIONS LOGIC')
            st.markdown('# Источники')
            sidebar_citation_columns = st.columns([1 for x in st.session_state.context['cited_sources']])
            for i, sitebar_citation_column in enumerate(sidebar_citation_columns):
                with sitebar_citation_column:
                    link = st.session_state.context['citations_lookup'].get(i).get('source')
                    stylish_citation_link(link, f"{i+1}")



    # initialize preemptive filters
    if 'filter_0_active' not in st.session_state:
        st.session_state.filter_0_active = False
    if 'filter_1_active' not in st.session_state:
        st.session_state.filter_1_active = False
    if 'filter_2_active' not in st.session_state:
        st.session_state.filter_2_active = False




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
        prompt = prompt.strip()
        logger.critical(f'context checkpoint 1: {st.session_state.context}')
        st.chat_message("user").markdown(prompt)
        # append before any processing so that the LLM call can be made with the chat history only
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        if prompt:
            if 'cited_sources' in st.session_state.context: st.session_state.context.pop('cited_sources')
            if 'citations_lookup' in st.session_state.context: st.session_state.context.pop('citations_lookup')
            non_html_chat_history = ChatHistory.truncate_exclude_last(chat_history=st.session_state.chat_history, n=CHAT_HISTORY_SIZE)
            if ((not 'scenario_name' in st.session_state.context) or
                    (st.session_state.context.get('scenario_name') == 'just_chatting') or
                    (st.session_state.context.get('scenario_name') == 'reroute') or
                    (st.session_state.context.get('current_state') == 'exit')):
                logger.critical(f'context before scenario_router.route(): {st.session_state.context }')
                # execute once per scenario_name / scenraio_object
                logger.critical('We must see this only either at the specific scenario start (once) OR after every message in just_chatting')
                logger.warning(f'prompt: {prompt}')
                logger.warning(f'non_html_chat_history: {non_html_chat_history}')
                # logger.warning(f'st.session_state.chat_history -- {st.session_state.chat_history}')
                selected_route_str = scenario_router.route(
                    user_query=prompt,
                    chat_history=non_html_chat_history,
                    stop=['<|eot_id|>'],
                    grammar_path='/home/amstel/llm/src/grammars/scenario_router.gbnf'
                )
                st.session_state.context['scenario_name'] = selected_route_str
                logger.critical(f'context after scenario_router.route(): {st.session_state.context}')
            if st.session_state.context['scenario_name'].startswith('shopping_assistant_'):
                # initial
                st.session_state.scenario_object = ShoppingAssistantScenario(scenario_name=st.session_state.context['scenario_name'])
                st.session_state.context['current_step'] = 'verify'
                if 'sql_query' in st.session_state.context: st.session_state.context.pop('sql_query')
            elif st.session_state.context['scenario_name'] == 'sberbank_consultant':
                st.session_state.scenario_object = SberbankConsultant()
            elif st.session_state.context['scenario_name'] == 'just_chatting':
                st.session_state.scenario_object = JustChattingScenario()  # stateless (no current_step) by design
                # if just_chatting, conversation template follows user-assistant temalate in turns, separated by technical tags
                non_html_chat_history = ChatHistory.truncate_include_last(chat_history=st.session_state.chat_history,
                                                                          n=CHAT_HISTORY_SIZE)
            # universal scenario logic
            data, context = st.session_state.scenario_object.handle(user_query=prompt, chat_history=non_html_chat_history, context=st.session_state.context)
            st.session_state.context = context
            assert 'scenario_name' in st.session_state.context
            if st.session_state.context.get('current_step') in ('sql','exit') and 'sql_items' in st.session_state:
                st.session_state.pop('sql_items')
            logger.critical(f'context after scenario_object.handle(): {st.session_state.context }')

            # Render LLM output
            if isinstance(data, str):
                response_text = data
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)
            elif isinstance(data, pd.DataFrame):
                render_df(data)

    # we finished sql-df rendering just now
    # show theses only if ( .get('previous_steps')[-1] == 'sql' and .get('scenario_name') == 'reroute' and .get('current_step') == 'exit' and .get('sql_schema')
    logger.debug(st.session_state.context)
    if (
        ('scenario_object' in st.session_state and isinstance(st.session_state.scenario_object, ShoppingAssistantScenario)) and
        st.session_state.context.get('previous_steps', ["", ""])[-1] == 'sql' and
        st.session_state.context.get('scenario_name') == 'reroute' and
        st.session_state.context.get('current_step') == 'exit' and
        st.session_state.context.get('sql_schema')
    ):
        if 'cited_sources' in st.session_state.context: st.session_state.context.pop('cited_sources')
        if 'citations_lookup' in st.session_state.context: st.session_state.context.pop('citations_lookup')
        # inline_filter = InlineFilter()
        st.markdown('##')
        st.markdown('##')
        st.markdown(
            """
            <style>
            div[data-testid="stButton"] button {
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 30px;
                font-weight: bold;
                width: 105%;
                transition: background-color 0.3s, opacity 0.3s;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        text_2_where = {
            'Цена от 3000 руб.': 'WHERE price >= 3000',
            'Загрузка от 8 до 10 кг.': 'WHERE max_load BETWEEN 8 AND 10',
            'Рейтинг от 4,9': 'WHERE rating_value >= 4.9',
        }
        _, col1, col2, col3, _ = st.columns([0.01, 0.3, 0.3, 0.3, 0.09])
        button_clicked = False
        with col1:
            rus_text, sql_where_clause = list(text_2_where.items())[0]
            if st.button(rus_text, type='primary', key=rus_text):
                logger.error(sql_where_clause)
                st.session_state.chat_history.append({"role": "user", "content": rus_text})
                assert st.session_state.context['sql_query'] is not None
                logger.debug(st.session_state.context['sql_query'])
                new_sql_value = update_sql_statement(
                    sql_statement=st.session_state.context['sql_query'],
                    new_where_clause=sql_where_clause)
                st.session_state.context['sql_query'] = new_sql_value
                st.session_state.context['current_step'] = 'sql'
                button_clicked = True


        with col2:
            rus_text, sql_where_clause = list(text_2_where.items())[1]
            if st.button(rus_text, type='primary', key=rus_text):
                logger.error(sql_where_clause)
                st.session_state.chat_history.append({"role": "user", "content": rus_text})
                assert st.session_state.context['sql_query']
                new_sql_value = update_sql_statement(
                    sql_statement=st.session_state.context['sql_query'],
                    new_where_clause=sql_where_clause)
                st.session_state.context['sql_query'] = new_sql_value
                st.session_state.context['current_step'] = 'sql'
                button_clicked = True


        with col3:
            rus_text, sql_where_clause = list(text_2_where.items())[2]
            if st.button(rus_text, type='primary', key=rus_text):
                logger.error(sql_where_clause)
                st.session_state.chat_history.append({"role": "user", "content": rus_text})
                assert st.session_state.context['sql_query']
                new_sql_value = update_sql_statement(
                    sql_statement=st.session_state.context['sql_query'],
                    new_where_clause=sql_where_clause)
                st.session_state.context['sql_query'] = new_sql_value
                st.session_state.context['current_step'] = 'sql'
                button_clicked = True

        if button_clicked:
            data, context = st.session_state.scenario_object.handle(user_query=None, chat_history=None, context=st.session_state.context)
            assert isinstance(data, pd.DataFrame)
            if 'sql_items' in st.session_state:
                # sql_items must be popped every time sql is executed!
                st.session_state.pop('sql_items')
            render_df(data)
