from fastapi import FastAPI

app = FastAPI()

@app.get("{param}/")
async def root(param: str):
    return {"message": "Hello World"}

