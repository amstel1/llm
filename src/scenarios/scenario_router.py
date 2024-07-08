import sys
sys.path.append('/home/amstel/llm')
sys.path.append('/home/amstel/llm/src')
from scenarios.shopping_assistant import chat_history_list_to_str
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field
from llama_cpp import LlamaGrammar
from typing import Optional, List, Dict, Literal
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from general_llm.llm_endpoint import call_generate_from_query_api, MODEL_NAME
from general_llm.prompt_construction import Llama3PromptTemplate, Gemma2PromptTemplate
import json
from loguru import logger


class Route_(str, Enum):
    just_chatting = "just_chatting"
    shopping_assistant_washing_machine = "shopping_assistant_washing_machine"
    shopping_assistant_fridge = "shopping_assistant_fridge"
    shopping_assistant_tv = "shopping_assistant_tv"
    shopping_assistant_mobile = "shopping_assistant_mobile"
    sberbank_consultant = "sberbank_consultant"

class Route(BaseModel):
    Route_


class ScenarioRouter:
    def __init__(self,
                 ):

        self.assistant_starts_with = '\nJSON:'
        self.system_prompt_without_chat_history = 'You are a state-of-the-art intent classifer.'
        self.system_prompt_with_chat_history = "You are a state-of-the-art intent classifier. When analyzing user input, prioritize the user's most recent message over any previous messages in the chat history. For instance, the input 'What is the weather like?' should not be classified as 'sberbank_consultant'."
        self.user_prompt_without_chat_history_placeholder = """route mapping: 
just_chatting: разговор на любые темы,  которые напрямую не относятся к подбору потребительских товаров или Сбер Банку (банковским услугам, картам, манибэк, кэшбэк, кредитам, депозитам)
shopping_assistant_washing_machine: поиск, выбор, покупка стиральной или стирально-сушильной машины
shopping_assistant_fridge: поиск, выбор, покупка холодильника
shopping_assistant_tv: поиск, выбор, покупка телевизора
shopping_assistant_mobile: поиск, выбор, покупка мобильного телефона
sberbank_consultant: консультация по всем вопросам, связанным с накоплением и сбережением, банковскими продуктами (карта, депозит, кредит), услугами (покупай валюту, страховка) (Сбер Банк Беларусь)

user input:
{user_input}

From the provided user input and chat history, identify the route that best matches the user's current intent. Based on your reasoning, decide on the route as JSON."""



##########################################################################################
        self.user_prompt_with_chat_history_placeholder = """From the provided user input and chat history, identify the route that best matches the user's current intent. If the user input changes the topic, disregard the previous context and focus on the new request. Remember, your response must be the exact route name as a JSON string.

chat history:
{chat_history}

user_input:
{user_input}

route mapping: 
sberbank_consultant: консультация по всем вопросам о банке, в том числе, связанным с накоплением и сбережением, банковскими продуктами (карта, манибэк, кэшбэк, депозит, кредит), услугами (покупай валюту, страховка) (Сбер Банк Беларусь) - related to banking services and products
shopping_assistant_washing_machine: поиск, выбор, покупка стиральной или стирально-сушильной машины
shopping_assistant_fridge: поиск, выбор, покупка холодильника
shopping_assistant_tv: поиск, выбор, покупка телевизора
shopping_assistant_mobile: поиск, выбор, покупка мобильного телефона
just_chatting: разговор на любые темы,  которые НЕ относятся к подбору потребительских товаров или Сбер Банку (банковским услугам, картам, кредитам, депозитам). """

    def route(self,
              user_query: str,
              chat_history: Optional[List[Dict[str, str]]] = None,
              grammar_path: str = None,  # force json output
              stop: list[str] = [],
              ) -> Route:
        '''returns the name of the chosen route as str'''
        # if grammar_path and not grammar:
        #     with open(grammar_path, 'r') as f:
        #         grammar = f.read()
        if (chat_history is None) or (not chat_history) or (len(chat_history)==0):
            generation_result = call_generate_from_query_api(
                system_prompt=self.system_prompt_without_chat_history,
                user_prompt=self.user_prompt_without_chat_history_placeholder.format(**{"user_input": user_query,}),
                assistant_must_start_with=self.assistant_starts_with,
                grammar_path=grammar_path,
                stop=stop,
            )
        else:
            # done: chat_history must be parsed, already implemented at .shopping_assistant, reuse
            str_chat_history = chat_history_list_to_str(chat_history)
            generation_result = call_generate_from_query_api(
                system_prompt=self.system_prompt_with_chat_history,
                user_prompt=self.user_prompt_with_chat_history_placeholder.format(
                    **{"user_input": user_query, "chat_history": str_chat_history,}),
                assistant_must_start_with=self.assistant_starts_with,
                grammar_path=grammar_path,
                stop=stop,
            )
        logger.debug(f'2606 - router - {generation_result}')
        selected_route_dict = JsonOutputParser().parse(generation_result)
        selected_route = selected_route_dict.get('route')
        logger.warning(selected_route)
        if isinstance(selected_route, list):
            assert len(selected_route) == 1
            selected_route_str = selected_route[0]  # 'just_chatting' / 'shopping_assistant_washing_machine'
        elif isinstance(selected_route, str):
            # the only way since 26/06/2024
            selected_route_str = selected_route
        return selected_route_str


if __name__ == '__main__':
    # pass
    fake_history = [
        {'role': 'user', 'content': 'найди телевизор'},
        # {'role': 'user', 'content': 'диагональ 55'},
        # {'role': 'user', 'content': 'фирма LG'},
#         {'role': 'user', 'content': 'производитель TCL, 65 дюймов'},
#         {'role': 'user', 'content': 'что такое кубок стэнли?'},
#         {'role': 'assistant', 'content': """Кубок Стэнли (Stanley Cup) - это приз, который вручается победителю Национальной хоккейной лиги (НХЛ) в Северной Америке. Кубок был создан в 1892 году и является старейшим трофеем в профессиональном спорте в мире.
#
# Кубок Стэнли имеет уникальную историю и традицию. Он сделан из кристалла, серебра и золота, и его высота составляет 36,5 см, а диаметр - 15,2 см. Кубок весит около 17 кг.
#
# В НХЛ Кубок Стэнли вручается победителю плей-офф, который выигрывает финальную серию Кубка Стэнли. Команда, которая выигрывает Кубок Стэнли, получает право хранить его до следующего сезона, а затем он возвращается в хранилище НХЛ.
#
# Кубок Стэнли имеет важное символическое значение для игроков и команды, которые его выигрывают. Он является признанием их достижений и является одним из самых престижных трофеев в мире спорта."""},
#         {'role': 'user', 'content': 'как называется трофей за победу в НБА?'},
#         {'role': 'assistant', 'content': """Трофей за победу в Национальной баскетбольной ассоциации (НБА) называется Кубком Ларри О'Брайена (Larry O'Brien Championship Trophy). Он вручается команде, которая выигрывает финальную серию НБА."""},
        # {'role': 'user', 'content': 'из чего сделан убок стэнли?'},
    ]

    router = ScenarioRouter()
    route = router.route(user_query="диагональ 55",
                         chat_history=fake_history,
                         grammar_path='/home/amstel/llm/src/grammars/scenario_router.gbnf'
                         )
    print(route)