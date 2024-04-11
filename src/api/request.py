import requests
import json

url = "http://localhost:8000/pay/"

headers = {"accept": "application/json", "Content-Type":"application/json"}
data_json={"service_name":"Пополнение баланса"}

# response = requests.post(
#     url, 
#     json=data_json,
#     headers=headers,
#     timeout=1000,
# )

# response = requests.get(
#     url, 
#     json=data_json,
#     # headers=headers,
#     timeout=1000,
# )

response = requests.get(
    url, 
    params={'item':'Коммунальные платежи'.encode('utf8')},
    headers=headers,
    timeout=1000,
)

print(response.text)