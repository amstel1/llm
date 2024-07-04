'''
I'm building an LLM powered chat bot. Here is my coarse architecture. A user query goes into scenario choosing module. Here the LLM performs classification among available scenarios. For brevity suppose currently we have two scenarios: just chatting and shopping assistant. Just chatting is simply a generic chat with the LLM with message history. Shopping assistant is a scenario where two steps should occur: 1. Gathering the requirements: the user might say all the specs they what the desired product to be OR an LLM must find out what those specs are via the conversation; 2. The LLM translates the user query (possibly reformulated or extracted from several chat history messages) into an sql query. The result is a table with requested data,  e.g.: price, max load, model name.

I want you to:
1. Create a blueprint for files, classes and methods that should be implemented to solve this task. Do not write full code, but rather small project structure and/or class/methods definitions. Keep it simple, clean and obvious.
'''

from abc import abstractmethod, ABC
from typing import List, Dict, Iterable, Any
from loguru import logger

class BaseScenario(ABC):
    def handle(self, user_query: Any, chat_history: Any, context: Any):
        raise NotImplementedError