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
    user_query = "стиральная машина"
    user_content = f"""Here is the user query: {user_query}.

            Evaluate if the user query contains enough information to be make a valid sql statement from it.
            Here are the examples when it DOES NOT contain enough information:
            - подобрать стиральную машину
            - обзор стиральных машин
            - какую стиральную машину выбрать

            Here are the examples when it DOES contain enough information:
            - стриальная машина ширина до 43 загрузка от 6 кг
            - стиралка Атлант с отзывами недорого
            - надежная машинка с сушилкой

            Return exactly either true or false and nothing else.
            """
    prompt = PromptTemplate.from_template(template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
    chain = prompt | llm
    response = chain.invoke({"question": user_content})
    # response = llm._call(prompt=prompt.format(question=user_content))
    print(response)