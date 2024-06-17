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
from general_llm.llm_endpoint import call_generation_api
import json
from loguru import logger

# route_to_description = (
#     "just_chatting: разговор на любые темы",
#     "shopping_assistant_washing_machine: поиск, выбор, покупка стиральной или стирально-сушильной машины",
#     "sberbank_consultant: консультация по всем вопросам, связанным с банковскими продуктами, услугами (Сбер Банк Беларусь)"
# )

class Route_(str, Enum):
    just_chatting = "just_chatting"
    shopping_assistant_washing_machine = "shopping_assistant_washing_machine"
    sberbank_consultant = "sberbank_consultant"

class Route(BaseModel):
    Route_


class ScenarioRouter:
    def __init__(self,
                 ):
        self.prompt_without_chat_history = """<|start_header_id|>system<|end_header_id|>
You are a state-of-the-art intent classifer.<|eot_id|><|start_header_id|>user<|end_header_id|>

route mapping: 
just_chatting: разговор на любые темы
shopping_assistant_washing_machine: поиск, выбор, покупка стиральной или стирально-сушильной машины
sberbank_consultant: консультация по всем вопросам, связанным с накоплением и сбережением, банковскими продуктами (карта, депозит, кредит), услугами (покупай валюту, страховка) (Сбер Банк Беларусь)

user input:
{user_input}

Given user input and mapping of possible routes and their descriptions select the most appropriate route for the user input.

Use step-by-step reasoning:
1. Analyze the content of the user input to determine if it essence relates to one of the route descriptions.
2. If not, assign just_chatting.

Based on your reasoning, decide on the route as JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nJSON:"""


        self.prompt_with_chat_history = """<|start_header_id|>system<|end_header_id|>
You are a state-of-the-art intent classifer.<|eot_id|><|start_header_id|>user<|end_header_id|>

chat history:
{chat_history}

user_input:
{user_input}

route mapping: 
just_chatting: разговор на любые темы
shopping_assistant_washing_machine: поиск, выбор, покупка стиральной или стирально-сушильной машины
sberbank_consultant: консультация по всем вопросам, связанным с накоплением и сбережением, банковскими продуктами (карта, депозит, кредит), услугами (покупай валюту, страховка) (Сбер Банк Беларусь)

Based on the user input and the chat history, identify which route the user's input most closely relates to. Your decision should take into account the context provided by the chat history. Respond with the most relevant route name from the given mapping.
Please respond with the most relevant route name as JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nJSON:"""

    def route(self,
              user_query: str,
              chat_history: Optional[List[Dict[str, str]]] = None,
              grammar: str = None,
              grammar_path: str = '/home/amstel/llm/src/grammars/scenario_router.gbnf',  # force json output
              stop: list[str] = ['<|eot_id|>'],
              ) -> Route:
        '''returns the name of the chosen route as str'''
        if grammar_path and not grammar:
            with open(grammar_path, 'r') as f:
                grammar = f.read()
        if (chat_history is None) or (not chat_history) or (len(chat_history)==1):
            generation_result = call_generation_api(
                prompt = self.prompt_without_chat_history.format(**{"user_input": user_query, }),
                grammar=grammar,
                stop=stop,
            )
        else:
            # done: chat_history must be parsed, already implemented at .shopping_assistant, reuse
            str_chat_history = chat_history_list_to_str(chat_history)
            generation_result = call_generation_api(
                prompt=self.prompt_with_chat_history.format(
                    **{"user_input": user_query, "chat_history": str_chat_history, }),
                grammar=grammar,
                stop=stop,
            )
        logger.debug(f'2805 - router - {generation_result}')
        selected_route_dict = json.loads(generation_result)
        selected_route = selected_route_dict.get('route')
        logger.warning(selected_route)
        if isinstance(selected_route, list):
            assert len(selected_route) == 1
            selected_route_str = selected_route[0]  # 'just_chatting' / 'shopping_assistant_washing_machine'
        elif isinstance(selected_route, str):
            selected_route_str = selected_route
        return selected_route_str


if __name__ == '__main__':
    # pass
    router = ScenarioRouter()
    route = router.route(user_query="какие кредиты есть в сбере?",
                         stop=['<|eot_id|>'],
                         # grammar_path='/home/amstel/llm/src/grammars/scenario_router.gbnf'
                         )
    print(route)