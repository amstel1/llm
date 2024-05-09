import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
from rag_config import SPLITTER_SEPARATORS, CHUNK_OVERLAP, CHUNK_SIZE, RAG_COLLECTION_NAME
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import ParentDocumentRetriever
import numpy as np

from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence

def text_cosine_similarity(text_a: str, text_b: str):
    a = embedding_model.embed_query(text_a)
    b = embedding_model.embed_query(text_b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

filtered = [doc for doc in docs if doc.metadata.get('source') == 'https://www.sber-bank.by/card/sbercard']
content = filtered[0].page_content

with open('docs_final.pkl', 'rb') as f:
    docs = pickle.load(f)

# Initialize the LLM
llm = Ollama(model="llama3_custom", stop=["<|eot_id|>"], num_gpu=-1, num_thread=-1, temperature=0, mirostat=0)

# Define the templates
llama_raw_template_system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Ты наилучшим образом делаешь то что тебе говорят.<|eot_id|>"
llama_raw_template_user = "<|start_header_id|>user<|end_header_id|>\nКонтекст:\n\n{context}\n\nВопрос:\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# Create the formatting prompt
formatting_prompt = PromptTemplate.from_template(template=llama_raw_template_system + llama_raw_template_user)

# Define the formatting chain
formatting_chain = formatting_prompt | llm | StrOutputParser()

# Define the summarization template and prompt
summarize_template = "<|start_header_id|>user<|end_header_id|>Описание:\n{input}\n\nВыше приведено описание банковского продукта или услуги. Напиши одно предложение которое максимально точно и полно характериует его суть и преимущества. Отвечай на русском языке.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
summarization_prompt = PromptTemplate.from_template(template=summarize_template)

# Define summarization chain, ensuring input is formatted as expected
summarization_chain = {"input": lambda x: {"input": x}} | summarization_prompt | llm | StrOutputParser()

# Combine the chains THIS IS WORKING!!!!!!!!!!!!!!!!!!!11
combined_chain = RunnableParallel({
    "s1":formatting_chain,
    "s2":summarization_chain
}
)

# Prepare the input

r = combined_chain.invoke({
    "question": "Выше приведено содержимое вебсайта. Данные извлечены из каждого тега и разделены символом | . Сделай эти данные более структурированными и соедини релевантные параграфы вместе. Ты можешь только менять форматирование документа. Отвечай на русском языке.",
    "context": content,
})

# print(r)

for k, v in r.items():
    print(k)
    print(v)
    print()
    print('########')
    print()