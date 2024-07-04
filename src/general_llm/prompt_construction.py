from abc import abstractmethod, ABC
from typing import List, Dict, Iterable, Any
from loguru import logger

def parse_markup_chat_history(chat_history: List[str]):
    """get rid of the tags, only (role, text) messages must remain -> List[("role", "content")]"""
    logger.info(chat_history)
    raise NotImplementedError


class Gemma2PromptTemplate:
    final_assistant = "<start_of_turn>model\n"

    # instead of the functios below we can extend:
    # https://api.python.langchain.com/en/latest/_modules/langchain_experimental/chat_models/llm_wrapper.html#Llama2Chat

    def create_statement(self, role: str, content: str):
        assert role in ('user', 'assistant')
        role = role.replace('assistant', 'model')
        assert content
        return f"<start_of_turn>{role}\n{content}<end_of_turn>\n"

    def create_prompt_from_history(self, system_prompt=None, chat_history: List[Dict[str, str]] = None, assistant_must_start_with: str = None):
        '''chat history ~ few shot'''
        template = ""
        if system_prompt:
            current_template_part = self.create_statement(role='user', content=system_prompt)
            template += current_template_part
        if chat_history:
            logger.warning(chat_history)
            for message in chat_history:
                role = message.get('role')
                content = message.get('content')
                current_template_part = self.create_statement(role, content)
                template += current_template_part
            template += self.final_assistant
            logger.warning(template)
        else:
            raise AttributeError  # should never be executed
            # final_user = create_statement('user', question)
            # template = final_user + final_assistant
            # logger.warning(template)
        if assistant_must_start_with:
            template += assistant_must_start_with
        return template

    def create_prompt_from_user_query(self, system_prompt_clean: str, user_query: str, assistant_must_start_with: str = None):
        template = self.create_statement(role='user', content=system_prompt_clean)  # for phi3 system = user
        current_template_part = self.create_statement(role='user', content=user_query)
        template += current_template_part
        template += self.final_assistant
        if assistant_must_start_with:
            template += assistant_must_start_with
        return template

############################################


class Llama3PromptTemplate:
    final_assistant = "<|start_header_id|>assistant<|end_header_id|>"

    def create_statement(self, role: str, content: str):
        assert role in ('system', 'user', 'assistant')
        assert content
        return f"<|start_header_id|>{ role }<|end_header_id|>\n{ content }<|eot_id|>"

    def create_prompt_from_history(self, system_prompt_clean:str, chat_history: List[Dict[str, str]], assistant_must_start_with: str = None):
        """"""
        # assert chat_history
        template = self.create_statement(role='system', content=system_prompt_clean)
        if chat_history:
            for message in chat_history:
                role = message.get('role')
                content = message.get('content')
                current_template_part = self.create_statement(role, content)
                template += current_template_part
        template += self.final_assistant
        if assistant_must_start_with:
            template += assistant_must_start_with
        return template

    def create_prompt_from_user_query(self, system_prompt_clean:str, user_query: str, assistant_must_start_with: str = None):
        template = self.create_statement(role='system', content=system_prompt_clean)
        current_template_part = self.create_statement(role='user', content=user_query)
        template += current_template_part
        template += self.final_assistant
        if assistant_must_start_with:
            template += assistant_must_start_with
        return template

############################################

class Phi3PromptTemplate:
    final_assistant = "<|assistant|>\n"

    # instead of the functios below we can extend:
    # https://api.python.langchain.com/en/latest/_modules/langchain_experimental/chat_models/llm_wrapper.html#Llama2Chat
    def create_statement(self, role: str, content: str):
        assert role in ('user', 'assistant')
        assert content
        return f"<|{role}|>\n{content}<|end|>\n"

    def create_prompt_from_history(self, system_prompt=None, chat_history: List[Dict[str, str]] = None, assistant_must_start_with: str = None):
        '''chat history ~ few shot'''
        template = ""
        if system_prompt:
            current_template_part = self.create_statement(role='user', content=system_prompt)
            template += current_template_part
        if chat_history:
            logger.warning(chat_history)
            for message in chat_history:
                role = message.get('role')
                content = message.get('content')
                current_template_part = self.create_statement(role, content)
                template += current_template_part
            template += self.final_assistant
            logger.warning(template)
        else:
            raise AttributeError  # should never be executed
            # final_user = create_statement('user', question)
            # template = final_user + final_assistant
            # logger.warning(template)
        if assistant_must_start_with:
            template += assistant_must_start_with
        return template

    def create_prompt_from_user_query(self, system_prompt_clean:str, user_query: str, assistant_must_start_with: str = None):
        template = self.create_statement(role='user', content=system_prompt_clean)  # for phi3 system = user
        current_template_part = self.create_statement(role='user', content=user_query)
        template += current_template_part
        template += self.final_assistant
        if assistant_must_start_with:
            template += assistant_must_start_with
        return template

############################################