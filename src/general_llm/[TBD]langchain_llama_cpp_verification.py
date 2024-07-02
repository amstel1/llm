from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_llama_cpp_api_warpper import LlamaCppApiWrapper

if __name__ == '__main__':
    # llm = LlamaCpp(
    #     model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q6_K.gguf',
    #     n_gpu_layers=33,
    #     max_tokens=-1,
    #     n_batch=512,
    #     n_ctx=2048,
    #     f16_kv=False,
    #     verbose=True,
    #     temperature=0.0,
    #     stop=['<|eot_id|>'],
    # )
    llm = LlamaCppApiWrapper()
    # messages = [
    #     {"role":"system", "content": "Ты вежливый и интересный AI ассистент."},
    #     {"role":"user", "content": "Привет"},
    # ]

    user_content = f"""На основе информации ниже сформулируй суть требований пользователя кратко, но сохраняя все важные детали. Требования могут касаться только одного типа товаров.

История чата:
user: найди телевизор
user: диагональ 55
user: фирма LG

Последний запрос пользователя: производитель TCL, 65 дюймов"""
    prompt = PromptTemplate.from_template(template="<|start_header_id|>system<|end_header_id|>\nТы вежливый, умный и эффективный ИИ-помощник. Ты всегда стараешься выполнять пожелания пользователя наилучшим образом.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
    chain = prompt | llm
    response = chain.invoke({"question": user_content})
    # response = llm._call(prompt=prompt.format(question=user_content))
    print(response)