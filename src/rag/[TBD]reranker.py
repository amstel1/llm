from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from loguru import logger
from langchain.retrievers.document_compressors import FlashrankRerank

def decouple(d: dict):
    logger.warning(d)
    return d

runable_decouple = RunnableLambda(decouple)

class RelevancyScore(BaseModel):
    relevancy: int = Field(description="How relevant is Document to the question", required=True)

filter_system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nТы должен определить, насколько хорошо документ подходит для ответа на вопрос. Используй шкалу от 1 (ужасно) до 5 (превосходно). Возвращай JSON с единственным ключом relevancy.<|eot_id|>"
filter_template = "<|start_header_id|>user<|end_header_id|>\nдокумент: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nJSON:"
filter_prompt = PromptTemplate.from_template(template=filter_system+filter_template)
llm = OllamaFunctions(model="llama3", format="json", temperature=0)
# llm = OllamaFunctions(model="llama3_q6_32k", format="json", stop=["<|eot_id|>",], num_gpu=33, temperature=0, mirostat=0)
structured_llm = llm.with_structured_output(RelevancyScore)

structured_chain = filter_prompt | runable_decouple | structured_llm | StrOutputParser()


# doc = 'конекст: \nБазовые характеристики безотзывного депозита "к Совершеннолетию" в Сбербанке:\n\n**Валюта:**\n\n* BYN (Белорусский рубль)\n\n**Сроки:**\n\n* 2 года\n* 730 дней (около 2 лет)\n\n**Ставки:**\n\n* 9% годовых\n\n**Место открытия:**\n\n* В отделении банка\n\n**Отзывность/безотзывность:**\n\n* Безотзывный депозит\n\n**Дополнительные параметры:**\n\n* Автоматическое переоформление до достижения 18-летия\n* Пополнение депозита возможно в течение первоначального (следующего после переоформления) срока депозита, за исключением последних 35 дней\n* Начисленные проценты капитализируются ежемесячно в последний рабочий день месяца и могут быть востребованы в течение срока депозита\n\n**Условия открытия:**\n\n* Депозит может быть открыт:\n\t+ на имя Несовершеннолетнего (до 18 лет) лицом независимо от родственных отношений с Несовершеннолетним\n\t+ Несовершеннолетним от 14 до 18 лет на свое имя\n* В случае заключения договора, предусматривающего выплату депозита на текущий счет с использованием банковской платежной карточки, необходимо иметь такой счет, открытый в валюте депозита\n\n**Условия возврата:**\n\n* Возврат депозита возможен:\n\t+ при востребовании депозита Вкладчиком в любом работающем подразделении банка\n\t+ в случае прекращения автоматического переоформления депозита на аналогичный срок:\n\t\t- посредством перевода Банком на Счет\n\t\t- посредством перевода Банком на текущий счет, открываемый в автоматическом режиме при невостребовании депозита Вкладчиком в подразделении банка\n\n**Дополнительные сервисы:**\n\n* Вкладчики-пользователи СБОЛ могут воспользоваться сервисами СБОЛ с использованием любой своей банковской платежной карточки (в том числе в валюте, отличной от валюты депозита) по пополнению депозита, востребованию причисленных к депозиту процентов, востребованию депозита в дату наступления срока возврата либо средств депозита с текущего счета после прекращения его переоформления\n* Операция пополнения депозита может совершаться иными лицами\n\n**Преимущества:**\n\n* Автоматическое переоформление до достижения 18-летия\n\nВопрос:\nпроконсультируй по безотзывным депозитам в белорусских рублях на короткий срок.'
doc = '\n\nБезотзывный депозит "Сохраняй". Ключевые условия: онлайн-открытие, валюта - EUR, USD, RUB, ставка - 0,1%, срок - от 370 до 750 дней, сумма - от 100 до 100000 EUR, минимальная сумма первоначального взноса не ограничена. || **Условия начисления и выплаты процентов:**\n\n* Проценты причисляются к остатку денежных средств на депозите ежемесячно\n* Может быть востребовано в течение срока депозита'
q = "вопрос: проконсультируй по безотзывным депозитам в белорусских рублях на короткий срок"

print(len(doc))
resp = structured_chain.invoke({"input": doc+ q})

print(resp)