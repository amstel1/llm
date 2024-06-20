from base import Job
from web_scraping.utils import EcomItemListRead
from utils import ItemListDo
from postgres.utils import PostgresDataFrameWrite
from loguru import logger
import sys

class Step1:
    def __init__(self, shop_name: str, product_name: str):
        # product_name - one of the products below
        # product_name - alias of one of the three shops
        self.name = shop_name
        kettle = {
            'product_type_name': 'Чайник',
            'schema_name': 'kettle',
            'shop_slug': 'chayniki',
            'shop_max_page': 106,
            'vek21_slug': 'teapots',
            'vek21_max_page': 31,
            'onliner_slug': 'kettle',
            'onliner_max_page': 120,
        }

        vacuumcleaner = {
            'product_type_name': 'Пылесос',
            'schema_name': 'vacuumcleaner',
            'shop_slug': 'pylesosy',
            'shop_max_page': 106,
            'vek21_slug': 'vacuum',
            'vek21_max_page': 24,
            'onliner_slug': 'vacuumcleaner',
            'onliner_max_page': 136,
        }

        headphones = {
            'product_type_name': 'Наушники',
            'schema_name': 'headphones',
            'shop_slug': 'naushniki',
            'shop_max_page': 106,
            'vek21_slug': 'headphones',
            'vek21_max_page': 42,
            'onliner_slug': 'headphones',
            'onliner_max_page': 226,
        }

        smartwatch = {
            'product_type_name': 'Умные часы',
            'schema_name': 'smartwatch',
            'shop_slug': 'umnie_chasi',
            'shop_max_page': 56,
            'vek21_slug': 'smart_watches',
            'vek21_max_page': 9,
            'onliner_slug': 'smartwatch',
            'onliner_max_page': 32,
        }

        dishwasher = {
            'product_type_name': 'Посудомойка',
            'schema_name': 'dishwasher',
            'shop_slug': 'posudomoechnye_mashiny',
            'shop_max_page': 61,
            'vek21_slug': 'dishwashers',
            'vek21_max_page': 10,
            'onliner_slug': 'dishwasher',
            'onliner_max_page': 77,
        }

        hob_cooker = {
            'product_type_name': 'Варочная панель',
            'schema_name': 'hob_cooker',
            'shop_slug': 'varochnie_paneli',
            'shop_max_page': 106,
            'vek21_slug': 'hobs',
            'vek21_max_page': 25,
            'onliner_slug': 'hob_cooker',
            'onliner_max_page': 167,
        }

        oven_cooker = {
            'product_type_name': 'Духовой шкаф',
            'schema_name': 'oven_cooker',
            'shop_slug': 'duhovye_shkafy',
            'shop_max_page': 106,
            'vek21_slug': 'ovens',
            'vek21_max_page': 18,
            'onliner_slug': 'oven_cooker',
            'onliner_max_page': 127,
        }

        iron = {
            'product_type_name': 'Утюг',
            'schema_name': 'iron',
            'shop_slug': 'utyugi',
            'shop_max_page': 56,
            'vek21_slug': 'irons',
            'vek21_max_page': 13,
            'onliner_slug': 'iron',
            'onliner_max_page': 57,
        }

        conditioner = {
            'product_type_name': 'Кондиционер',
            'schema_name': 'conditioner',
            'shop_slug': 'konditsionery',
            'shop_max_page': 102,
            'vek21_slug': 'conditioners',
            'vek21_max_page': 25,
            'onliner_slug': 'conditioners',
            'onliner_max_page': 37,
        }

        waterheater = {
            'product_type_name': 'Водонагреватель',
            'schema_name': 'waterheater',
            'shop_slug': 'vodonagrevateli',
            'shop_max_page': 84,
            'vek21_slug': 'waterheaters',
            'vek21_max_page': 27,
            'onliner_slug': 'waterheater',
            'onliner_max_page': 46,
        }

        microwave = {
            'product_type_name': 'Микроволновка',
            'schema_name': 'microwave',
            'shop_slug': 'mikrovolnovye_pechi',
            'shop_max_page': 68,
            'vek21_slug': 'microwaves',
            'vek21_max_page': 12,
            'onliner_slug': 'microvawe',
            'onliner_max_page': 73,
        }

        product_mapping = {
            'vacuumcleaner': vacuumcleaner,
            'headphones': headphones,
            'smartwatch': smartwatch,
            'dishwasher': dishwasher,
            'hob_cooker': hob_cooker,
            'oven_cooker': oven_cooker,
            'iron': iron,
            'conditioner': conditioner,
            'waterheater': waterheater,
            'microwave': microwave,
        }
        self.selected_product = product_mapping[product_name]


    def run(self):
        # step 1. ItemList from sites to Postgres. Not: all three for each new product
        product = self.selected_product
        logger.debug(self.name)
        if self.name == 'shop':
            product_type_url = [f'https://shop.by/{product["shop_slug"]}/?page_id={i}' for i in range(1, product["shop_max_page"])]  # 1,30
            this_job = Job(
                reader=EcomItemListRead(extractor_name='ShopByExtractor',
                                        product_type_url=product_type_url,
                                        product_type_name=product["product_type_name"]),
                processor=ItemListDo(),
                writer=PostgresDataFrameWrite(
                    schema_name=product["schema_name"],
                    table_name='product_item_list',  # product_item_list_to_fill, product_item_list
                    insert_unique=True,
                    index_column="product_url",
                    if_exists='append')
            )
            this_job.run()
        elif self.name == 'vek21' or self.name == '21vek':
            product_type_url=[f'https://www.21vek.by/{product["vek21_slug"]}/page:{i}/' for i in range(1, product["vek21_max_page"])]  # 2,11
            this_job = Job(
                reader=EcomItemListRead(extractor_name='Vek21Extractor',
                                        product_type_url=product_type_url,
                                        product_type_name=product["product_type_name"]),
                processor=ItemListDo(),
                writer=PostgresDataFrameWrite(
                    schema_name=product["schema_name"],
                    table_name='product_item_list',  # product_item_list_to_fill, product_item_list
                    insert_unique=True,
                    index_column="product_url",
                    if_exists='append'
                ),
            )
            this_job.run()
        elif self.name == 'onliner':
            product_type_url=[f'https://catalog.onliner.by/{product["onliner_slug"]}?page={i}' for i in range(1, product["onliner_max_page"])]  # 2,50
            this_job = Job(
                reader=EcomItemListRead(extractor_name='OnlinerExtractor',
                                        product_type_url=product_type_url,
                                        product_type_name=product["product_type_name"]),
                processor=ItemListDo(),
                writer=PostgresDataFrameWrite(
                    schema_name=product["schema_name"],
                    table_name='product_item_list',  # product_item_list_to_fill, product_item_list
                    insert_unique=True,
                    index_column="product_url",
                    if_exists='append')
            )
            this_job.run()


if __name__ == '__main__':
    raise NotImplementedError

