from llama_cpp import Llama

if __name__ == '__main__':
    # llm = Llama(
    #     model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q6_K.gguf',
    #     n_gpu_layers=33,
    #     max_tokens=-1,
    #     n_batch=512,
    #     n_ctx=2048,
    #     f16_kv=False,
    #     verbose=True,
    #     temperature=0.0,
    #     seed=0,
    #     # chat_format='llama-3'
    # )

    # messages = [
    #     {"role":"system", "content": "Ты вежливый и интересный AI ассистент."},
    #     {"role":"user", "content": "Привет"},
    # ]
    user_query = "стиральная машина"
    user_content = f"""На основе информации ниже сформулируй суть требований пользователя кратко, но сохраняя все важные детали. Требования могут касаться только одного типа товаров.

История чата:
user: найди телевизор
user: диагональ 55
user: фирма LG

Последний запрос пользователя: производитель TCL, 65 дюймов
"""
    # response = llm(prompt="""<|start_header_id|>system<|end_header_id|>\nТы вежливый и интересный AI ассистент.<|eot_id|><|start_header_id|>user<|end_header_id|>\nПривет!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n""", max_tokens=-1, temperature=0.0, echo=True, stop=['<eot_id>'])
    response = llm(
        prompt= f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        suffix=None,
        max_tokens=-1,
        temperature=0.0,
        top_p=0.95,
        min_p=0.05,
        echo=True,
        stop=['<eot_id>'],
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.1,
        top_k=40,
        mirostat_mode=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
    )
    print(response)