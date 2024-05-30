import sys
sys.path.append('/home/amstel/llm')
sys.path.append('/home/amstel/llm/src')
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field
from llama_cpp import LlamaGrammar
from typing import Optional, List, Dict, Literal
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from general_llm.llm_endpoint import call_generation_api
import json
from loguru import logger

route_to_description = (
    "just_chatting: разговор на любые темы",
    "shopping_assistant_washing_machine: поиск, выбор, покупка стиральной или стирально-сушильной машины",
    "sberbank_consultant: консультация по всем вопросам, связанным с банковскими продуктами, услугами (Сбер Банк Беларусь)"
)

class Route(str, Enum):
    just_chatting = "just_chatting"
    shopping_assistant_washing_machine = "shopping_assistant_washing_machine"
    sberbank_consultant = "sberbank_consultant"

class SelectedRoute(BaseModel):
    Route


class ScenarioRouter:
    def __init__(self,
                 llama_cpp_model_path = '/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q6_K.gguf',
                 llama_cpp_stop: List[str] = ['<|eot_id|>'],
                 route_to_description=route_to_description,
                 grammar_path='/home/amstel/llm/src/grammars/scenario_router.gbnf',
                 ):
        # self.llm = LlamaCpp(
        #     model_path=llama_cpp_model_path,
        #     n_gpu_layers=33,
        #     max_tokens=1024,
        #     n_batch=1024,
        #     n_ctx=2048,
        #     f16_kv=True,
        #     verbose=True,
        #     temperature=0.0,
        #     grammar_path=grammar_path,
        #     stop=llama_cpp_stop,
        # )

        self.route_to_description = route_to_description
        # self.prompt = PromptTemplate.from_template(
        #     template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        #               "You are a state-of-the-art intent classifer.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        #               "Given user_input and mapping of possible routes and their descriptions you must select the most appropriate route for the user_input.\n"
        #               "Here is the route to description mapping:\n{route_to_description}\n\n"
        #              "Given the user input below, think step-by-step to determine which route it should take:\n"
        #              "user_input:\n{user_input}\n"
        #             "Step-by-step reasoning:\n"
        #             "1. Analyze the content of the user input to determine if it essence relates to one of the route descriptions.\n"
        #             "2. If not, assign just_chatting.\n"
        #             "Based on your reasoning, decide on the route as JSON.\nJSON:"
        #               "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        # )
        # self.chain = self.prompt | self.llm | JsonOutputParser()
        self.prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n
        You are a state-of-the-art intent classifer.<|eot_id|><|start_header_id|>user<|end_header_id|>\n
        Given user_input and mapping of possible routes and their descriptions you must select the most appropriate route for the user_input.\n
        Here is the route to description mapping:\n{route_to_description}\n\n
        Given the user input below, think step-by-step to determine which route it should take:\n
        user_input:\n{user_input}\n
        Step-by-step reasoning:\n
        1. Analyze the content of the user input to determine if it essence relates to one of the route descriptions.\n
        2. If not, assign just_chatting.\n
        Based on your reasoning, decide on the route as JSON.\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>\nJSON:"""
    def route(self,
              user_query: str,
              chat_history: Optional[List[Dict[str, str]]] = None,
              grammar: str = None,
              grammar_path: str = '/home/amstel/llm/src/grammars/scenario_router.gbnf',  # force json output
              stop: list[str] = ['<|eot_id|>'],
              ) -> SelectedRoute:
        '''returns the name of the chosen route as str'''
        if grammar_path and not grammar:
            with open(grammar_path, 'r') as f:
                grammar = f.read()
        generation_result = call_generation_api(
            prompt= self.prompt.format(**{"user_input": user_query, "route_to_description": self.route_to_description}),
            grammar=grammar,
            stop=stop,
        )
        logger.debug(f'2805 - router - {generation_result}')
        selected_route_dict = json.loads(generation_result)
        selected_route = selected_route_dict.get('selected_route')
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