from typing import Optional, List, Dict, Literal
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


route_to_description = [
    {'just_chatting': "разговор на общие темы, не касаются поиска / выбора потребительских товаров"},
    {'shopping_assistant_washing_machine': "касается совета по поиску, выбору, покупке стиральной машины"},
]
Route = Literal['just_chatting', 'shopping_assistant_washing_machine']
print(Route)


class ScenarioRouter:
    def __init__(self,
                 ollama_model_name: str = 'llama3_q6_correct:latest',
                 ollama_stop: List[str] = ['<|eot_id|>', ],
                 route_to_description = route_to_description,
                 ):
        self.ollama_model_name = ollama_model_name
        self.ollama_stop = ollama_stop
        self.route_to_description = route_to_description
        self.llm = Ollama(
            model=self.ollama_model_name,
            stop=self.ollama_stop,
            num_gpu=-1,
            num_thread=-1,
            temperature=0,
            mirostat=0
        )
        self.prompt = PromptTemplate.from_template(
            template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                      "You are a state-of-the-art intent classifer.<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                      "Given user_input and mapping of possible routes and their descriptions you must select the most appropriate route for the user_input.\n\nRoute to description mapping:\n{route_to_description}\n\n"
                     "user_input:\n{user_input}"
                      "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            # partial_variables=['user_input']
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def route(self,
              user_query: str,
              chat_history: Optional[List[Dict[str, str]]] = None,
              ) -> Route:
        selected_route = self.chain.invoke({"user_input": user_query, "route_to_description": self.route_to_description})
        return selected_route


if __name__ == '__main__':
    router = ScenarioRouter()
    route = router.route(user_query="стиральная машина")
    print(route)