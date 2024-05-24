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
from base import BaseScenario, parse_markup_chat_history
from text2sql.prod_llama_fewshot import SqlToText
from typing import Iterable, Dict, List, Union, Optional
from streamlit_app.app import call_api

class AIResponse:
    pass

class ShoppingAssistantScenario(BaseScenario):
    possible_filters = ""  # -> max_oad, min_price, max_load, is_drying

    def __init__(self, ):
        """
        Initialize the Shopping Assistant with a database.

        :param database: A pandas DataFrame or database connection containing product data.
        """
        pass

    def evaluate_ready(self, text: str) -> bool:
        """
        Evaluate if the text can be directly translated into an SQL query.

        :param text: User query in natural language.
        :return: Boolean indicating if text2sql is possible.
        """
        # Implementation aspects:
        # 1. Use natural language processing (NLP) to determine if the query is well-defined.
        # 2. Check for keywords and structure that indicate a specific query.
        # 3. Handle edge cases where user input is ambiguous.
        prompt = """
        Here is the chat history: {chat_history}.
        Here is the user query: {text}.
        
        Evaluate it the user query contains enough information to be make a valid sql statement from it.
        Return exactly either true or false.
        """

    def get_possible_filters(self) -> Iterable:
        return possible_filters

    def ask_for_specs(self, text: str, chat_history: Iterable, possible_filters: Iterable) -> AIResponse:
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
        prompt = f"""
        Here is the chat history: {chat_history}.
        Here is the user query: {text}.
        Here are possible product attributes to be used as filters: {possible_filters}.
        
        Ask the user to specify one or more filters that you consider the most relevant.
        """
        return call_api(input_text=text, chat_history=chat_history)

    def generate_inline_filters(self, text: str) -> List[str]:
        """
        Generate possible inline filters based on the user query.

        :param text: User query in natural language.
        :return: A string containing possible filters.
        """
        # Implementation aspects:
        # 1. Analyze the user query to identify potential filters (e.g., price range, brand).
        # 2. Generate a list of pre-defined filters that match the query context.
        # 3. Ensure the filters are relevant and comprehensive.
        prompt = """
        Here is the user query: {text}.
        Here are possible product attributes to be used as filters: {possible_filters}.
        
        Generate a list of valid filter values in natural language. 
        """

    def translate_specs_into_text(self, specs: Union[[List[str], str], chat_history: Iterable = None) -> str:
        """
        Translate collected specifications into a natural language statement.

        :param specs: Collected specifications as a set or text.
        :return: A natural language statement summarizing the specifications.
        """
        # Implementation aspects:
        # 1. Combine all gathered details into a coherent natural language statement.
        # 2. Ensure the statement is concise and clear.
        # 3. Handle both textual and structured input for specifications.
        prompt = """
        Here is the chat history: {chat_history}
        Here are the required product specifications: {specs}
        
        Reformulate the context above as a concise, well-formulated user query while preserve all the important details.
        """

    def handler_query(self, user_input: str, chat_history: Iterable = None):
        ready_for_sql = self.evaluate_ready(text=user_input)
        if ready_for_sql:
            df = SqlToText.sql_query(user_input=user_input)
        else:
            # we can:
            # 1. ask the user to clarify their requirements
            # 2. generate inline filters and show them
            # todo: inline_filter_list = self.generate_inline_filters(text=user_input)
            if not chat_history:
                ai_response = self.ask_for_specs(text=user_input, chat_history=None,
                                                             possible_filters=self.possible_filters)
            elif chat_history:
                ai_response = self.ask_for_specs(text=user_input, chat_history=chat_history,
                                                             possible_filters=self.possible_filters)
            translate_specs_into_text = call_api(input_text=ai_response, chat_history=chat_history)
        return ai_response




# Example usage:
# database = pd.read_csv('products.csv')  # or some database connection
# assistant = ShoppingAssistantScenario(database)

# user_query = "I'm looking for a smartphone with a good camera."
# if assistant.evaluate_ready(user_query):
#     results = assistant.sql_query(user_query)
# else:
#     refined_query = assistant.gather_specs(user_query)
#     # Continue with further steps...