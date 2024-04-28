from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
import uvicorn
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger

llm = LlamaCpp(
    model_path='/home/amstel/llm/models/publisher/repository/Phi-3-mini-4k-instruct-q4.gguf',
    n_gpu_layers=33,
    temperature=0.0,
    stop=['<|end|>'],
)
template = PromptTemplate.from_template("""<|user|>\n{question} <|end|>\n<|assistant|>""")
app = FastAPI(redirection_slashes=False)

def log_data(x):
    logger.debug(x)
    return x

logger_runnable = RunnablePassthrough(log_data)
chain = template | logger_runnable | llm | StrOutputParser()

class Input(BaseModel):
    question: str

@app.get("/")
async def hello() ->dict[str, str]:
    return {"hello":"world"}

@app.post("/process_text")
async def process_text(input_data: Input) -> dict[str, Any]:
    result = chain.invoke({'question': input_data.question})
    return {"answer": result}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)