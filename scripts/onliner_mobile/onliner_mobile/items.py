# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from itemloaders.processors import TakeFirst

class OnlinerMobileItem(scrapy.Item):
    json_data = scrapy.Field()
    link = scrapy.Field(output_processor=TakeFirst())
