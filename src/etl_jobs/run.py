import sys
sys.path.append('/home/amstel/llm/src')
from typing import Optional, Dict, Any, List, Set, Tuple
from base import Job, Read, Do, Write, ChainedRead
import yaml
from loguru import logger

# step 1
from web_scraping.utils import EcomItemListRead
from postgres.utils import PostgresDataFrameWrite
from utils import ItemListDo

# step 2
from utils import ItemDetailsRead
from utils import ItemDetailsDo
from postgres.utils import PostgresDataFrameWrite
from utils import PickleDataRead, PickleDataWrite

# step 3
from mongodb.utils import MongoRead
from postgres.utils import PostgresDataFrameRead
from web_scraping.utils import SearchParseRead
from mongodb.utils import MongoWrite

class YandexMarketDo(Do):
    """yandex market product details and reviews - do"""
    @logger.catch
    @staticmethod
    def process(data: Dict) -> Tuple[List[Dict[str, Dict]], List[Dict[str, List[Dict]]]]:
        '''
        returns ? (products, reviews)
        insert the result to mongo
        '''
        product_details_output = []
        reviews_details_output = []
        for product_name, combined_data in data.items():
            assert product_name == combined_data[0]
            product_tuple = combined_data[1]
            reviews_tuple = combined_data[2]
            if product_tuple:
                product_request_url, product_details = product_tuple
                if product_details:
                    product_details['product_request_url'] = product_request_url
                    product_details_output.append({product_name: product_details})  # str -> Dict

            if reviews_tuple:
                reviews_request_url, reviews_details = reviews_tuple
                if reviews_details:
                    # reviews_details is list
                    for review in reviews_details:
                        review.update({'reviews_request_url': reviews_request_url})
                    reviews_details_output.append({product_name: reviews_details})  # str -> List[Dict]

        return product_details_output, reviews_details_output

class ChainedRead_SearchParse_Part1:
    '''implementation for search and parse reviews reader - get mongo (x2), get postgres, calculate urls to search'''
    def __init__(self, readers: List[Read]):
        self.container = {}
        self.readers = readers

    def prepare(self) -> Any:
        for i, reader in enumerate(self.readers):
            self.container[f'step_{i}'] = reader.read()
        return self.container

    def read(self) -> Any:
        # executed in job

        # step 0 is mongo (scraped_data.product_reviews)
        # step 1 is mongo (scraped_data.product_details)
        # step 2 is postgres (scraped_data.product_item_list)
        # step 3 is postgres (scraped_data.product_query_attempts)

        self.prepare()
        already_scraped_names = []
        for data in self.container.get('step_0'):
            already_scraped_names.extend([key for key in data.keys() if key != '_id'])
        for data in self.container.get('step_1'):
            already_scraped_names.extend([key for key in data.keys() if key != '_id'])
        already_scraped_names = list(set(already_scraped_names))
        item_list = self.container.get('step_2')['product_name'].values.tolist()
        already_queried = self.container.get('step_3')['attempt_product_name'].values.tolist()

        product_names_to_scrape = set(
            [x for x in item_list if x not in already_scraped_names and x not in already_queried])
        assert len(product_names_to_scrape) > 0
        return product_names_to_scrape

class ChainedRead_SearchParse_Part2:
    def __init__(self, readers: List[Read]):
        self.container = {}
        self.readers = readers

    def read(self,) -> Any:
        product_names_to_scrape = self.readers[0].read() # pickle data read
        self.container[f'step_0'] = product_names_to_scrape
        data = self.readers[1].read(product_names_to_scrape=product_names_to_scrape)
        return data




if __name__ == '__main__':
    pass

    # step 1. ItemList from sites to Postgres
    # logger.warning('Start - Job 1')
    # product_type_url = [f'https://shop.by/stiralnye_mashiny/?page_id={i}' for i in range(1, 30)]
    # product_type_name='Стиральная машина'
    # ItemlList_2_Postgres = Job(
    #     reader=EcomItemListRead(extractor_name='ShopByExtractor', product_type_url=product_type_url, product_type_name=product_type_name),
    #     processor=ItemListDo(),
    #     writer=PosgresDataFrameWrite(schema_name='scraped_data', table_name='product_item_list')
    # )
    # ItemlList_2_Postgres.run()
    # logger.warning('End - Job 1')

    # step 2. Read: ItemList from Postgres, Do: Scrapy ProductDetails, Write: to Postgres
    # logger.warning('Start - Job 2')
    # ItemDetails_2_Postgres = Job(
    #     reader=ItemDetailsRead(
    #         step1__table='scraped_data.product_item_list',
    #         step1__where=None,
    #         step1_utls_attribute='product_url'
    #     ),
    #     processor=ItemDetailsDo(),
    #     writer=PostgresDataFrameWrite(
    #         schema_name='scraped_data',
    #         table_name='item_details_washing_machine'),
    # )
    # ItemDetails_2_Postgres.run()
    # logger.warning('End - Job 2')


    ## step 3.A - prepare data, save to pickle
    ## Read: 3.1
    # mongo_read_product_reviews = MongoRead(operation='read', db_name='scraped_data', collection_name='product_reviews')
    ## Read 3.2
    # mongo_read_product_details = MongoRead(operation='read', db_name='scraped_data', collection_name='product_details')
    ## Read: 3.3
    # postgres_read_item_list = PostgresDataFrameRead(
    #     table='scraped_data.product_item_list',
    #     where="product_position <= 1 limit 10"
    # )
    ## Read: 3.4
    # postgres_read_query_attemps = PostgresDataFrameRead(table='scraped_data.product_query_attempts')
    ## Part 3A
    # logger.warning('Start - Job 3A')
    # scrape_internet_part_A = Job(
    #     reader=ChainedRead_SearchParse_Part1(readers=[
    #         mongo_read_product_reviews,
    #         mongo_read_product_details,
    #         postgres_read_item_list,
    #         postgres_read_query_attemps
    #     ]),
    #     writer=PickleDataWrite(filepath='temp_1605.pkl'),
    # )
    # scrape_internet_part_A.run()
    # scrape_internet_part_B = Job(
    #     reader=PickleDataRead(filepath='temp_1605.pkl'),
    #     writer=PickleDataWrite(filepath='temp_16052.pkl'),
    # )
    # scrape_internet_part_B.run()
    # logger.warning('End - Job 3A')

    # step 3B
    logger.warning('Start - Job 3B')
    pkl_reader = PickleDataRead(filepath='temp_16052.pkl')
    internet_reader = SearchParseRead()

    # to include, need refactoring so that output is: [{}, {}, {}]
    # ym_processor = YandexMarketDo()  # returns product_details_output, reviews_details_output
    # product_details_writer = MongoWrite(operation='write', db_name='scraped_data', collection_name='product_details')
    # reviews_details_writer = MongoWrite(operation='write', db_name='scraped_data', collection_name='review_details')

    scrape_internet_part_B = Job(
        reader=ChainedRead_SearchParse_Part2(readers=[pkl_reader, internet_reader]),
        writer=PickleDataWrite(filepath='temp_16053.pkl'),
    )
    scrape_internet_part_B.run()
    logger.warning('End - Job 3B')
