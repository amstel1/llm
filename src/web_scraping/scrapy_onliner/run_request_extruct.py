import extruct
import requests
import pprint
from w3lib.html import get_base_url
pp = pprint.PrettyPrinter(indent=2)
from extruct.jsonld import JsonLdExtractor

link = 'https://catalog.onliner.by/mobile/xiaomi/xi1412512bk'

def extract_json_ld(link:str):
    r = requests.get(link)
    jslde = JsonLdExtractor()
    data = jslde.extract(r.text)
    return data