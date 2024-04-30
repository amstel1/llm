from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from loguru import logger

template = PromptTemplate.from_template(template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n""")

llm = LlamaCpp(
    model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct.Q6_K.gguf',
    n_gpu_layers=33,
    max_tokens=512,
    n_batch=128,
    n_ctx=1024,
    f16_kv=True,
    verbose=False,
    temperature=0.0,
    # repeat_penalty=1.5,
    grammar_path='/home/amstel/llm/src/grammars/json.gbnf',
    stop=["<|eot_id|>", "<|start_header_id|>"],
)

class PersonalData(BaseModel):
    has_personal_data: bool = Field(description="boolean indicating whether personal data is present in the input")
    personal_data_contents: str = Field(description="personal data from the input")

json_parser = JsonOutputParser(pydantic_object=PersonalData)
# struct_parser = StructuredOutputParser()
str_parser = StrOutputParser()

chain = template | llm | json_parser

"""
фамилия, имя, отчество, пол, дата рождения, серия и номер паспорта, личный номер, адрес регистрации и проживания
email
"""

pdn = [
    'Интернет: 2050359',
    'Коммунальные платежи: 3719000920',
    'Пополнение карты (онлайн): 00754933003122/1174',
    'МТС по № телефона: 297718072',
    'Телефон: 3186190',
    'Машино-место: 099030',
    'Электроэнергия (физ.лица): 813020103',
    'Налог Кнорина: AB3870391',
    'Телефон: 4561693',
    'Коммун.платежи АИС Расчет-ЖКУ: 150041148',
    'Телефон: 2422154',
    'Коммун.платежи АИС Расчет-ЖКУ: 8662460541',
    'Коммунальные платежи: 11121390011',
    'Электроэнергия (физ.лица): 698731482',
    'МТС по № телефона: 297718072',
    'life:) по № телефона: 256078683',
    'Пополнение счета: 35295279476',
    'Коммун.платежи АИС Расчет-ЖКУ: 7600030107',
    'Электроэнергия (физ.лица): 184360546',
    'Разрешение (физ.лица): MAA1637739',
    'Газоснабжение: 495405',
    'Электроэнергия (физ.лица): 50-23',
    'Пополнение карты (онлайн): 3001779330055930/7698',
    'Коммунальные платежи: 11121390001',
    'Газоснабжение: 495405',
]

for pers_data in pdn:
    response = chain.invoke(input={
        "system_prompt": 'You are a world class entity extractor tasked with securing personal data of the customers. Output a JSON with two keys: has_personal_data - boolean, indicating whether personal data is present in the input; personal_data_contents - array of strings, containing the result of the user request.',
        "user_message": 'In the string below find all substrings containing personal data and return them EXACTLY as in the input.\n' + pers_data
    })
    logger.info(response)
