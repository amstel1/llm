'''
I'm building an LLM powered chat bot. Here is my coarse architecture. A user query goes into scenario choosing module. Here the LLM performs classification among available scenarios. For brevity suppose currently we have two scenarios: just chatting and shopping assistant. Just chatting is simply a generic chat with the LLM with message history. Shopping assistant is a scenario where two steps should occur: 1. Gathering the requirements: the user might say all the specs they what the desired product to be OR an LLM must find out what those specs are via the conversation; 2. The LLM translates the user query (possibly reformulated or extracted from several chat history messages) into an sql query. The result is a table with requested data,  e.g.: price, max load, model name.

I want you to:
1. Create a blueprint for files, classes and methods that should be implemented to solve this task. Do not write full code, but rather small project structure and/or class/methods definitions. Keep it simple, clean and obvious.
'''

from abc import abstractmethod, ABC
from typing import List, Dict, Iterable
from loguru import logger

def parse_markup_chat_history(chat_history: List[str]):
    """get rid of the tags, only (role, text) messages must remain"""
    pass

# instead of the functios below we can extend:
# https://api.python.langchain.com/en/latest/_modules/langchain_experimental/chat_models/llm_wrapper.html#Llama2Chat
def create_chatml_statement(role: str, content: str):
    assert role in ('user', 'assistant')
    assert content
    return f"<|{role}|>\n{content}<|end|>\n"

def create_llama3_statement(role: str, content: str):
    logger.debug(role)
    assert role in ('user', 'assistant')
    assert content
    return f"<|start_header_id|>{ role }<|end_header_id|>\n{ content }<|eot_id|>"

def get_chatml_template(chat_history: List[Dict[str, str]]):
    '''chat history ~ few shot'''
    template = ""
    final_assistant = "<|assistant|>\n"
    if chat_history:
        logger.warning(chat_history)
        for message in chat_history:
            role = message.get('role')
            content = message.get('content')
            current_template_part = create_chatml_statement(role, content)
            template += current_template_part
        template += final_assistant
        logger.warning(template)
    else:
        raise AttributeError  # should never be executed
        # final_user = create_statement('user', question)
        # template = final_user + final_assistant
        # logger.warning(template)
    return template

def get_llama3_template(system_prompt_clean:str, chat_history: List[Dict[str, str]]):
    """"""
    # assert chat_history
    final_assistant = "<|start_header_id|>assistant<|end_header_id|>"
    template = f'<|start_header_id|>system<|end_header_id|>\n{ system_prompt_clean }<|eot_id|>'
    if chat_history:
        for message in chat_history:
            role = message.get('role')
            content = message.get('content')
            current_template_part = create_llama3_statement(role, content)
            template += current_template_part
    template += final_assistant
    logger.warning(template)
    return template

class BaseScenario(ABC):
    def handler_query(self, user_input: str,):
        raise NotImplementedError