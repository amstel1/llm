from fastapi import FastAPI
from pydantic import BaseModel
import json
from loguru import logger

app = FastAPI()

with open('json_mapper.json', 'r') as f:
    lookup_json = json.load(f)

# class Item(BaseModel):
#     service_name: str

@app.get("/pay/")
async def create_item(item: str):
    logger.info(f'item: {item}')
    payment_details = lookup_json.get(item)
    logger.info(f'payment_details: {payment_details}')
    logger.warning('api called successfully')
    return {'results':payment_details}