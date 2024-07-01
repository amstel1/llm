from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_llama_cpp_api_warpper import LlamaCppApiWrapper

from llama_cpp import Llama
from llama_cpp_agent.providers import LlamaCppPythonProvider

# https://python.langchain.com/v0.1/docs/use_cases/question_answering/citations/#cite-documents
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType

class cited_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )

if __name__ == '__main__':
    # langchain
    # llm = LlamaCpp(
    #     model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q6_K.gguf',
    #     n_gpu_layers=32,
    #     max_tokens=-1,
    #     n_batch=512,
    #     n_ctx=1024,
    #     f16_kv=False,
    #     verbose=True,
    #     temperature=0.0,
    #     stop=['<|eot_id|>'],
    # )

    # llm_with_tool = llm.bind_tools(
    #     [cited_answer],
    #     tool_choice="cited_answer",
    # )
    # example_q = """What Brian's height?
    #
    #     Source: 1
    #     Information: Suzy is 6'2"
    #
    #     Source: 2
    #     Information: Jeremiah is blonde
    #
    #     Source: 3
    #     Information: Brian is 3 inches shorted than Suzy"""
    # response = llm_with_tool.invoke(example_q)
    #
    # # user_content = f""""""
    # # prompt = PromptTemplate.from_template(template="<|start_header_id|>system<|end_header_id|>\nТы вежливый, умный и эффективный ИИ-помощник. Ты всегда стараешься выполнять пожелания пользователя наилучшим образом.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
    # # chain = prompt | llm
    # # response = chain.invoke({"question": user_content})
    # # response = llm._call(prompt=prompt.format(question=user_content))
    # print(response)



    # llama-cpp-python
    llama_model = Llama(
        model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q6_K.gguf',
        n_gpu_layers=33,
        max_tokens=-1,
        n_batch=512,
        n_ctx=2048,
        f16_kv=False,
        verbose=True,
        temperature=0.0,
        seed=0,
        # chat_format='llama-3'
    )
    provider = LlamaCppPythonProvider(llama_model)
    agent = LlamaCppAgent(
        provider,
        predefined_messages_formatter_type=12,  # llama 3
        add_tools_and_structures_documentation_to_system_prompt=False,

    )
    agent.get_text_response()