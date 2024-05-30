# ask - generate - verify - sql

# I'm writing a chatbot scenario called "ShoppingAssistantScenario". My task is: given the user query return the best product from a database.
# Here are the coarse steps and their respective method definitions:
#
# 1. Evaluate if text2sql from the user_query possible
# method: evaluate_ready(text) -> bool
# comment: this step is neccessary because the user can provide us either with a well defined query in natural language or a vague statement that requires to be enriched with details.
#
# 2. Perform text2sql
# method: sql_query(text) -> pd.DataFrame
# comment: translate the query into an sql statement. This method is called if the previous step returned true.
#
# 3. Ask more question, steer the user (via prompt?) to write the desirable conditions
# method: gather_specs(text) -> text
# comment: if the input is not appropriate to create an sql from, we need to find out more details about the desired properties of the product
#
# 4. Generate possible inline filters
# method: generate_specs(text) -> text
# comment: instead of asking the user to write their requirements, at this step we generate pre-defined filters
#
# 5. Translate them into language
# method: translate_specs_into_text(set | text) -> text
# comment: after we've gathered the specs either as text or as a set of predefined filters, we need to compress all the chat history in a single natural language statement.
#
# Create a blueprint for the steps above in python. Write well defined code in python. Create classes and method definitions. For each method in the comment docstring write the most important implementation aspects need to be resolved for the best quality possible.

import sys
sys.path.append('/home/amstel/llm/src')
import pandas as pd
from scenarios.base import BaseScenario, parse_markup_chat_history, get_llama3_template_from_history, get_llama3_template_from_user_query
from text2sql.prod_llama_fewshot import SqlToText
from typing import Iterable, Dict, List, Union, Optional, Any
import requests
from loguru import logger
from general_llm.llm_endpoint import call_generation_api, call_generate_from_history_api, call_generate_from_query_api

def chat_history_list_to_str(chat_history: list):
    str_chat_history = ''
    for i, d in enumerate(chat_history):
        if i < len(chat_history) - 1:
            str_chat_history += d.get('role') + ': ' + d.get('content') + '\n'
        else:
            # assert d.get('content') == user_query
            pass
            # debug here: last chat history statement must equal text
    return str_chat_history


class ShoppingAssistantScenario(BaseScenario):
    possible_filters = ""  # -> max_oad, min_price, max_load, is_drying

    def __init__(self, ):
        """
        Initialize the Shopping Assistant with a database.

        :param database: A pandas DataFrame or database connection containing product data.
        """
        pass

    def verify(self, user_query: str, chat_history: list, context: Any) -> bool:
        """
        Evaluate if the text can be directly translated into an SQL query.

        :param text: User query in natural language.
        :return: Boolean indicating if text2sql is possible.
        """
        # Implementation aspects:
        # 1. Use natural language processing (NLP) to determine if the query is well-defined.
        # 2. Check for keywords and structure that indicate a specific query.
        # 3. Handle edge cases where user input is ambiguous.
        # parsed_chat_history = parse_markup_chat_history(chat_history)

#         user_content = f"""You are tasked to decide whether the user query contains at least one attribute / characteristic / feature of any kind.
# Return exactly either true or false.
# Return false if the user did not mention any desired properties.
# Return true if the user mentioned at least one of the desired properties or attributes, e.g.: size, width, quality, brand, price, name, rating, product's specific features.
# Return true if you are specifically asked to perform a query or if you are not allowed to ask any more questions.
#
# Here is the user query you need to evaluate: {user_query}
# """

        user_content = f"""Ты должен решить, содержит ли запрос пользователя хотя бы один описательный атрибут / характеристику / определение любого рода.
Верни true или false.
Верни false, если запрос не содержит какие-либо желаемые свойства / характеристики / определения.
Верни true, если пользователь упомянул хотя бы одно из желаемых свойств, характеристик или атрибутов, например: размер, ширину, качество, бренд, цену, название, рейтинг, особенности продукта.

Запрос пользователя: {user_query}"""
        prompt = get_llama3_template_from_history(system_prompt_clean='Ты объективный семантический оценщик.', chat_history=[{'role': 'user', 'content': user_content}])
        return call_generation_api(prompt=prompt, grammar='root ::= "true" | "false"')


    def get_possible_filters(self) -> Iterable:
        possible_filters = ['цена', 'рейтинг', 'брэнд', 'максимальная загрукзка', 'наличие сушки']
        return possible_filters

    def ask(self, user_query: str, chat_history: Iterable, context: Any) -> str:
        """
        Ask more questions to gather detailed specifications from the user.

        :param text: User query in natural language.
        :return: Refined text with more detailed specifications.
        """
        # Implementation aspects:
        # 1. Identify missing details in the user's query.
        # 2. Generate follow-up questions to extract these details.
        # 3. Maintain the context of the conversation to avoid repetitive questions.
        possible_filters = self.get_possible_filters()
        if chat_history:
            str_chat_history = chat_history_list_to_str(chat_history)
            user_prompt = f"""Ответь на реплику пользователя так, чтобы выяснить, какие характеристики важны для пользователя.
Вот история чата:\n{str_chat_history}\n
Вот характеристики, которые ты можешь использовать: {', '.join(possible_filters)}.
Твой ответ должен быть кратким, но содержательным.

Вот реплика пользователя: {user_query}.

Твой ответ: """
        else:
            user_prompt = f"""Ответь на реплику пользователя так, чтобы выяснить, какие характеристики важны для пользователя.
Вот характеристики, которые ты можешь использовать: {', '.join(possible_filters)}.
Твой ответ должен быть кратким, но содержательным.

Вот реплика пользователя: {user_query}.

Твой ответ: """

        return call_generate_from_query_api(
            user_prompt=user_prompt,
            system_prompt='Ты вежливый, умный и эффективный ИИ-помощник. Ты всегда стараешься выполнять пожелания пользователя наилучшим образом.'
        )


    def reformulate(self, user_query: Union[List[str], str] = None, chat_history: Iterable = None, context=None) -> str:
        """
        Translate collected specifications into a natural language statement.

        :param specs: Collected specifications as a set or text.
        :return: A natural language statement summarizing the specifications.
        """
        # Implementation aspects:
        # 1. Combine all gathered details into a coherent natural language statement.
        # 2. Ensure the statement is concise and clear.
        # 3. Handle both textual and structured input for specifications.

        str_chat_history = chat_history_list_to_str(chat_history)
        user_prompt = f"""История чата:/n{str_chat_history}./n
        Последний запрос пользователя: {user_query}
        
        На основе информации выше сформулируй суть требований пользователя кратко, но сохраняя все важные детали."""

        return call_generate_from_query_api(
            user_prompt=user_prompt,
            system_prompt='Ты вежливый, умный и эффективный ИИ-помощник. Ты всегда стараешься выполнять пожелания пользователя наилучшим образом.'
        )


    def handle(self, user_query, chat_history, context) -> [Any, Dict]:
        current_step = context.get('current_step')
        logger.debug(f'current_step - {current_step}')
        assert current_step in ('verify', 'ask', 'sql', 'reformulate',)
        previous_steps = context.get('previous_steps')
        assert isinstance(previous_steps, list)
        if current_step == 'verify':
            ready_for_sql_str = self.verify(user_query=user_query, chat_history=chat_history, context=context)
            ready_for_sql = eval(ready_for_sql_str.capitalize())
            logger.debug(f'ready_for_sql - {ready_for_sql}')
            previous_steps.append(current_step)
            if ready_for_sql:
                current_step = 'sql'
                logger.debug(f'current step - {current_step}')
                context['current_step'] = current_step
                df = SqlToText.sql_query(user_query=user_query)
                previous_steps.append(current_step)
                context['previous_steps'] = previous_steps
                current_step = 'exit'
                context['current_step'] = current_step
                context['scenario_name'] = "just_chatting"
                return df, context
            else:
                current_step = 'ask'
                logger.debug(f'current step - {current_step}')
                # todo: inline_filter_list = self.generate_inline_filters(text=user_input)
                context['current_step'] = current_step
                response = self.ask(user_query=user_query, chat_history=chat_history, context=context)
                previous_steps.append(current_step)
                context['previous_steps'] = previous_steps
                current_step = 'reformulate'
                context['current_step'] = current_step
                return response, context
        elif current_step == 'ask':
            logger.debug(f'current step - {current_step}')
            response = self.ask(user_query=user_query, chat_history=chat_history, context=context)
            previous_steps.append(current_step)
            context['previous_steps'] = previous_steps
            current_step = 'reformulate'
            context['current_step'] = current_step
            return response, context
        elif current_step == 'reformulate':
            response = self.reformulate(user_query=user_query, chat_history=chat_history, context=context)
            logger.critical(f'reformulated: {response}')
            logger.debug(f'current step - {current_step}')
            previous_steps.append(current_step)
            current_step = 'verify'
            context['current_step'] = current_step
            ready_for_sql_str = self.verify(user_query=user_query, chat_history=chat_history, context=context)
            ready_for_sql = eval(ready_for_sql_str.capitalize())
            logger.debug(f'ready_for_sql - {ready_for_sql}')
            previous_steps.append(current_step)
            if ready_for_sql:
                current_step = 'sql'
                logger.debug(f'current step - {current_step}')
                context['current_step'] = current_step
                df = SqlToText.sql_query(user_query=response)  # pass reformulated response here
                previous_steps.append(current_step)
                context['previous_steps'] = previous_steps
                current_step = 'exit'
                context['current_step'] = current_step
                context['scenario_name'] = "just_chatting"
                return df, context
            else:
                current_step = 'ask'
                logger.debug(f'current step - {current_step}')
                # todo: inline_filter_list = self.generate_inline_filters(text=user_input)
                context['current_step'] = current_step
                response = self.ask(user_query=user_query, chat_history=chat_history, context=context)
                previous_steps.append(current_step)
                context['previous_steps'] = previous_steps
                current_step = 'reformulate'
                context['current_step'] = current_step
                return response, context
        elif current_step == 'sql':
            logger.debug(f'current step - {current_step}')
            df = SqlToText.sql_query(user_input=user_query)
            previous_steps.append(current_step)
            context['previous_steps'] = previous_steps
            current_step = 'exit'
            context['current_step'] = current_step
            context['scenario_name'] = "just_chatting"
            return df, context


if __name__ == '__main__':

    user_query = "подбери стиральную машину"
    chat_history = []
    current_scenario = ShoppingAssistantScenario()
    verification_result = current_scenario.verify(
        user_query=user_query,
        chat_history=chat_history,
        context={"scenario": "shopping_assistant_washing_machine", "previous_steps": [], "current_step": "verify"})
    print(verification_result)  #  :bool = true | false

    # ch = [{'role': 'user', 'content': 'Привет'},
    #       {'role': 'assistant', 'content': 'Привет! Как я могу помочь вам сегодня?'},
    #       {'role': 'user', 'content': 'подбери стиральную машину'}]
    # user_query = 'подбери стиральную машину'
    # current_scenario = ShoppingAssistantScenario()
    # ask_result = current_scenario.ask(
    #     user_query=user_query,
    #     chat_history=ch,
    #     context=None,
    # )
    # print(ask_result)  # Привет! Я готов помочь вам выбрать стиральную машину. Какие параметры вы хотели бы видеть в этом продукте? Пожалуйста, ответьте на вопрос: какие из следующих свойств вам важны - цена, рейтинг, бренд, максимальная загрузка или наличие сушки?


    # ch = [{'role': 'user', 'content': 'Привет'},
    #       {'role': 'assistant', 'content': 'Привет! Как я могу помочь вам сегодня?'},
    #       {'role': 'user', 'content': 'подбери стиральную машину'},
    #       {'role': 'assistant', 'content': "Привет! Я готов помочь вам выбрать стиральную машину. Какие параметры вы хотели бы видеть в этом продукте? Пожалуйста, ответьте на вопрос: какие из следующих свойств вам важны - цена, рейтинг, бренд, максимальная загрузка или наличие сушки?"},
    #       {'role': 'user', 'content': 'недорогая но хорошая'},
    #       ]
    # user_query = 'недорогая но хорошая'
    # current_scenario = ShoppingAssistantScenario()
    # reformulate_result = current_scenario.reformulate(
    #     user_query=user_query,
    #     chat_history=ch,
    #     context=None,
    # )
    # print(reformulate_result)

    # user_input = 'Недорогая стиральная машина с хорошими характеристиками.'
    # response = SqlToText.sql_query(user_input=user_input)
    # print(response)
    pass