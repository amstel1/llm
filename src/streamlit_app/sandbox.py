import sys
sys.path.append('/home/amstel/llm/src')
import streamlit as st
from loguru import logger
import pandas as pd

from general_llm.utils import ChatHistory
CHAT_HISTORY_SIZE = 6
from streamlit_app.backend_ops import DataServer
from streamlit_app.cards import ItemDisplay
from scenarios.scenario_router import ScenarioRouter
from scenarios.shopping_assistant import ShoppingAssistantScenario
from scenarios.just_chatting import JustChattingScenario
from text2sql.prod_sql_to_text import extract_where_attributes
st.set_page_config(layout="wide")

def sort_products(products, criterion):
    if criterion == 'Сначала дороже':
        key = 'price'
        valid_products = [p for p in products if p.get(key) is not None]
        return sorted(valid_products, key=lambda x: x.get(key, -1), reverse=True)
    elif criterion == 'Сначала дешевле':
        key = 'price'
        valid_products = [p for p in products if p.get(key) is not None]
        return sorted(valid_products, key=lambda x: x.get(key, -1), reverse=False)
    elif criterion == 'По отзывам':
        key = 'rating_value'
        valid_products = [p for p in products if p.get(key) is not None]
        return sorted(valid_products, key=lambda x: x.get(key, -1), reverse=True)
    elif criterion == 'По популярности':
        key = 'rating_count'
        valid_products = [p for p in products if p.get(key) is not None]
        return sorted(valid_products, key=lambda x: x.get(key, -1), reverse=True)



def render_df(items: list,
              necessary_product_attributes_list: list[str],
              sort_criterion: str,
                ):


    # show loan terms for the top option
    # top_item = items[0]
    # logger.critical(f'0511 top item: {top_item}')
    # top_item_price = top_item.get('price')
    # duration_2_terms = {}

    # todo: display - redo 21 11 2024
    # sort and display data
    sorted_products = sort_products(items, sort_criterion)
    logger.critical(f'{type(sorted_products)}')  # continue from here!!!!
    create_grid(sorted_products)
    # sql results - show table
    # todo: get product_type_name from context variables
    # logger.warning(f'! important: {st.session_state.context}')
    # sql_result_ix = 0  # TEMP!!!!
    # item_display = ItemDisplay(items,
    #                            duration_2_terms=duration_2_terms,
    #                            sql_result_ix=sql_result_ix,  # wtf?
    #                            necessary_product_attributes_list=necessary_product_attributes_list
    #                            )
    # lgc = max(0, len(items) - 1)
    # item_display.display_grid(lower_grid_cols=lgc)
    # st.rerun()


def on_toggle_change(toggle_state):
    print(toggle_state)


def create_grid(products: list):
    logger.critical(len(products))
    logger.critical(products[0])
    # wildberries grid
    with st.container():
        grid_container = st.container()
        for i in range(4):
            cols = st.columns(5)
            with grid_container:
                for j, product in enumerate(products[i*5:(i+1)*5]):
                    with cols[j]:
                        st.image(product["product_image_url"], width=100)
                        st.write(f"**{product['name']}**")
                        st.write(product["price"])



if __name__ == '__main__':
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'sql_items' not in st.session_state:
        st.session_state['sql_items'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'context' not in st.session_state:
        st.session_state['context'] = {
            # 'scenario': "",
            # 'current_step': "",
            'previous_steps': ["", ""],
        }

    # Potentially display chat here
    # with st.sidebar:
    #     show_sidebar()

    scenario_router = ScenarioRouter()

    ### components ##########
    with st.container():
        _, main_column, _ = st.columns([1,1.5,1])
        with main_column:
            # toggle SQL: add or replace
            with st.container():
                toggle_column, toggle_text_column = st.columns([0.2, 0.75])
                with toggle_column:
                    toggle_state = st.toggle("Enable Feature", value=False, )
                with toggle_text_column:
                    pass

            # instruction
            st.title("Internet Shop")
            with st.expander(label="Instruction"):
                st.write("Here comes the instruction")

            # sorting
            with st.container():
                sort_criterion = st.radio("sort_criterion", ["Сначала дороже", "Сначала дешевле", "По отзывам", "По популярности"], horizontal=True, label_visibility='hidden')

            ### components ##########

    if prompt := st.chat_input("Enter your question here"):
        prompt = prompt.strip().lower()
        st.session_state['sql_items'] = None
        st.session_state['data'] = None

        _, aux_column, _ = st.columns([1, 1.5, 1])
        with aux_column:
            st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        if prompt:
            non_html_chat_history = ChatHistory.truncate_exclude_last(chat_history=st.session_state.chat_history, n=CHAT_HISTORY_SIZE)
            if ((not 'scenario_name' in st.session_state.context) or
                    (st.session_state.context.get('scenario_name') == 'just_chatting') or
                    (st.session_state.context.get('scenario_name') == 'reroute') or
                    (st.session_state.context.get('current_state') == 'exit')):
                # logger.critical(f'context before scenario_router.route(): {st.session_state.context}')
                # execute once per scenario_name / scenraio_object
                # logger.critical(
                #     'We must see this only either at the specific scenario start (once) OR after every message in just_chatting')
                # logger.warning(f'prompt: {prompt}')
                # logger.warning(f'non_html_chat_history: {non_html_chat_history}')
                # logger.warning(f'st.session_state.chat_history -- {st.session_state.chat_history}')
                selected_route_str = scenario_router.route(
                    user_query=prompt,
                    chat_history=non_html_chat_history,
                    stop=['<|eot_id|>'],
                    grammar_path='/home/amstel/llm/src/grammars/scenario_router.gbnf'
                )
                st.session_state.context['scenario_name'] = selected_route_str
                # logger.critical(f'context after scenario_router.route(): {st.session_state.context}')
            if st.session_state.context['scenario_name'].startswith('shopping_assistant_'):
                # initial
                st.session_state.scenario_object = ShoppingAssistantScenario(
                    scenario_name=st.session_state.context['scenario_name'])
                st.session_state.context['current_step'] = 'verify'
                if 'sql_query' in st.session_state.context: st.session_state.context.pop('sql_query')
            elif st.session_state.context['scenario_name'] == 'just_chatting':
                st.session_state.scenario_object = JustChattingScenario()  # stateless (no current_step) by design
                # if just_chatting, conversation template follows user-assistant temalate in turns, separated by technical tags
                non_html_chat_history = ChatHistory.truncate_include_last(chat_history=st.session_state.chat_history,
                                                                          n=CHAT_HISTORY_SIZE)
            # logger.debug(f'user_query -- {prompt}')
            # logger.debug(f'chat_history-- {non_html_chat_history}')
            # logger.debug(f'context-- {st.session_state.context}')
            # universal scenario logic
            data, context = st.session_state.scenario_object.handle(user_query=prompt, chat_history=non_html_chat_history,
                                                                    context=st.session_state.context)
            # logger.critical(f'05112024 does this have sql query?: {context}')
            st.session_state['data'] = data
            st.session_state.context = context
            assert 'scenario_name' in st.session_state.context
            if st.session_state.context.get('current_step') in ('sql', 'exit') and 'sql_items' in st.session_state:
                st.session_state.pop('sql_items')
            # logger.critical(f'context after scenario_object.handle(): {st.session_state.context}')

    # Render LLM output or stored data
    if isinstance(st.session_state['data'], str):
        data = st.session_state['data']
        response_text = data
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

    elif (isinstance(st.session_state['data'], pd.DataFrame) and st.session_state['data'].shape[0] > 0) or (st.session_state['sql_items']):
        data = st.session_state['data']
        necessary_product_attributes_list = extract_where_attributes(
            sql_query=st.session_state.context.get('sql_query'))
        # logger.error(f'1011 - necessary_product_attributes_list - {necessary_product_attributes_list}')
        assert st.session_state.context['sql_schema']  # must always exist
        data_server = DataServer(schema_name=st.session_state.context['sql_schema'])  # must pass this from scenario
        assert 'name' in data.columns
        if 'sql_items' not in st.session_state:
            # sql_items must be updated every time sql is executed!
            items = data_server.collect_data(data['name'])
            st.session_state['sql_items'] = items
            st.session_state.chat_history.append({"role": "html", "items": items})
        items = st.session_state['sql_items']

        # here we recreate display grid
        render_df(
            items,
            necessary_product_attributes_list=necessary_product_attributes_list,
            sort_criterion=sort_criterion
        )
    elif isinstance(st.session_state['data'], pd.DataFrame) and data.shape[0] == 0:
        data = st.session_state['data']
        response_text = "Извините, но я ничего не нашел."
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
