from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.llms import LlamaCpp
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from operator import itemgetter
import os
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from datetime import datetime
from pprint import pprint

trace_name = f'sql_{datetime.now()}'
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-a45f2a3d-4085-4170-b337-8cc2f1921aef"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-26acbd19-74af-4cb4-93e1-919526c13921"

langfuse = Langfuse()
langfuse_callback_handler = CallbackHandler(trace_name=trace_name)    
# client_id = 2607176

db = SQLDatabase.from_uri("sqlite:///fake_sbol.db")
# logger.info(f'db type: {type(db)}')
# db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM Artist LIMIT 10;"))

llm = LlamaCpp(
    # llama 13b saiga: -- '../models/model-q4_K(2).gguf'
    # roleplay - mixtral moe 8x7b: -- mixtral-8x7b-moe-rp-story.Q4_K_M
    # mixtral-8x7b-v0.1.Q4_K_M

    model_path='/home/amstel/llm/models/Publisher/Repository/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf',
    n_gpu_layers=8,  # 28 for llama2 13b, 10 for mixtral
    # model_path='/home/amstel/llm/models/Publisher/Repository/nsql-llama-2-7b.Q5_K_S.gguf',

    # model_path='/home/amstel/llm/models/model-q4_K(2).gguf',
    # n_gpu_layers=28,  # 28 for llama2 13b, 10 for mixtral
    max_tokens=500,
    n_batch=128,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
)




answer_prompt = PromptTemplate.from_template(
    """<s> [INST] Given the following user question, corresponding SQL query, and SQL result, answer the user question in a complete sentence. Do not give explanations. [/INST] </s>

Question: {question}
SQL Query: {query}
SQL Result: {result} 
Answer: """
)
answer = answer_prompt | llm | StrOutputParser()




top_k = 5


# template = '''Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Pay attention to the table schema.
# Return no more than {top_k} results unless prompted by the use. Use the following format:

# Question: "Question here" SQLQuery: "SQL Query to run" SQLResult: "Result of the SQLQuery" Answer: "Final answer here"

# Only use the following tables:
# {table_info}.

# Table schema:
# index | table index, must never be used
# local_amount | amount of money spent in local currency BYN
# currency_code | currency encoding
# category_id | category of transction
# subcategory_id | subcategory of transaction
# mcc_id | MCC of transaction
# transaction_datetime | datetime of transaction event
# event_id | unique identifier for each row
# client_id | client id

# Question: {input}''' 

dialect = 'sqlite'

table_info = '''CREATE TABLE "pfm_flat_table" (
  "index" INTEGER, 
  "local_amount" REAL,
  "currency_code" TEXT,
  "category_id" INTEGER,
  "subcategory_id" INTEGER,
  "mcc_id" INTEGER,
  "transaction_datetime" TIMESTAMP,
  "event_id" INTEGER,
  "client_id" INTEGER
)

/*
3 rows from pfm_flat_table table:
index | local_amount | currency_code | category_id | subcategory_id | mcc_id | transaction_datetime | event_id | client_id
0 | 203.71 | usd | 1004 | 45 | 4890 | 2023-08-15 21:11:00 | 66337344 | 2778777
0 | 43.64 | rub | 1009 | 41 | 4115 | 2023-05-03 00:56:00 | 66253780 | 4054162
0 | 427.61 | byn | 006 | 21 | 4965 | 2023-09-13 03:07:00 | 66687351 | 4324529
*/'''

examples = [
{"input": "How much money did I spend in each month?",
"query": "SELECT strftime('%m-%Y', transaction_datetime) AS month, SUM(local_amount) AS summa FROM pfm_flat_table GROUP BY month ORDER BY summa DESC;"
},
{"input": "How many transacation I make over previous three months?",
"query": "SELECT count(event_id) AS cnt FROM pfm_flat_table WHERE transaction_datetime BETWEEN DATE('now', 'start of month', '-3 months') and DATE('now', 'start of month', '-1 day');"
},
{"input": "How many transacation I make over previous three months in each month?",
"query": "SELECT strftime('%m-%Y', transaction_datetime) AS month, count(event_id) AS cnt FROM pfm_flat_table WHERE transaction_datetime BETWEEN DATE('now', 'start of month', '-3 months') and DATE('now', 'start of month', '-1 day') GROUP BY month ORDER BY cnt DESC;"
},
{"input": "How much money did I spend over previous three months?",
"query": "SELECT sum(local_amount) AS summa FROM pfm_flat_table WHERE transaction_datetime BETWEEN DATE('now', 'start of month', '-3 months') and DATE('now', 'start of month', '-1 day');"
},
{"input": "How many transactions did I make in every category and subcategory?",
"query": "SELECT category_id, subcategory_id, count(event_id) AS cnt FROM pfm_flat_table GROUP category_id, subcategory_id ORDER BY cnt DESC;"
},
{"input": "How many transactions did I make in each category?",
"query": "SELECT category_id, count(event_id) AS cnt FROM pfm_flat_table GROUP category_id ORDER BY cnt DESC;"
},
{"input": "What was the month with the most spending?",
"query": "SELECT month FROM (SELECT strftime('%m-%Y', transaction_datetime) AS month, sum(local_amount) AS summa FROM pfm_flat_table GROUP BY month ORDER BY summa DESC LIMIT 1);"
},
{"input": "sum by month and currency",
"query": "SELECT month FROM (SELECT strftime('%m-%Y', transaction_datetime) AS month, currency_code, sum(local_amount) AS summa FROM pfm_flat_table GROUP BY month, currency_code ORDER BY summa DESC LIMIT 1);"
},
{"input": "What was the spending by currency?",
"query": "SELECT currency_code, sum(local_amount) AS summa FROM pfm_flat_table GROUP BY currency_code ORDER BY summa DESC;"
},
{"input": "what are the top three spending categories in the last month?",
"query": "SELECT category_id, sum(local_amount) AS summa FROM pfm_flat_table WHERE transaction_datetime BETWEEN DATE('now', 'start of month', '-1 month') AND DATE('now', 'start of month', '-1 day') GROUP BY category_id ORDER BY summa DESC LIMIT 3;"
},
]

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

prompt = FewShotPromptTemplate(
    examples=examples[:20],
    example_prompt=example_prompt,
    prefix="<s> [INST] You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries. [/INST] </s>",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)

# template = '''<s> [INST] You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
# Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
# Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
# Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
# Pay attention to use date('now') function to get the current date, if the question involves "today".

# Use the following format:

# Input question: {input}
# SQLQuery: SQL Query to run
# SQLResult: Result of the SQLQuery
# Answer: Final answer here

# Only use the following tables:
# {table_info}

#  [/INST] </s>'''

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(
    llm, 
    db, 
    prompt=prompt,
    k=1
)


chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    # | answer
)

# no agent
# response = write_query.invoke({"question": "Get top 5 clients by then number of transactions"}, config={"callbacks":[langfuse_callback_handler]})
# response = write_query.invoke({"question": "How much money did client 2778777 spend in total?"}, config={"callbacks":[langfuse_callback_handler]})
response = write_query.invoke({
    "question": "what are the top 10 spending months in the last four months by the sum of operation and the respective sums",
    "dialect":dialect,
    "table_info":table_info,
    "top_k":top_k,
    }, config={"callbacks":[langfuse_callback_handler]})
# response = write_query.invoke({"question": "How much money did each client spend in two previous months?"}, config={"callbacks":[langfuse_callback_handler]})
# response = write_query.invoke({"question": "Get top 5 clients by then amount of transactions"}, config={"callbacks":[langfuse_callback_handler]})
pprint(response)