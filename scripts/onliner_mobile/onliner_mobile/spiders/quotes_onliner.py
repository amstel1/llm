from pathlib import Path
import scrapy
import extruct
import requests
from extruct.jsonld import JsonLdExtractor
from scrapy.loader import ItemLoader
from onliner_mobile.items import OnlinerMobileItem

def extract_json_ld(link:str):
    r = requests.get(link)
    jslde = JsonLdExtractor()
    data = jslde.extract(r.text)
    return data

class OnlinerSpider(scrapy.Spider):
    name = "onliner_scraper"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_callback = kwargs.get('args').get('callback')
        self.output_data = []

    def start_requests(self):
        urls = [
            "https://catalog.onliner.by/mobile?page=1",
            "https://catalog.onliner.by/mobile?page=2",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, meta={"playwright": True})

    def parse(self, response):
        for selector in response.css("div.catalog-form__offers-flex"):
            l = ItemLoader(item=OnlinerMobileItem(), selector=selector)
            link = selector.css('a::attr(href)').get()
            json_data = extract_json_ld(link)
            l.add_value("link", link)
            l.add_value("json_data", json_data)
            self.output_data.append(dict(l.load_item()))
            yield l.load_item()

    def close(self, spider, reason):
        self.output_callback(self.output_data)
