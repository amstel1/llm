import sys
from datetime import datetime
import pandas as pd
import sqlalchemy.types
from polyfuzz.models import TFIDF
import numpy as np
sys.path.append('/home/amstel/llm/src')
from typing import Optional, Dict, Any, List, Set, Tuple, Iterable
from base import Job, Read, Do, Write, ReadChain, WriteChain, DoChain, StepNum, DoChainGlobal
import yaml
from loguru import logger
from jobs import Step1




# step 1
from web_scraping.utils import EcomItemListRead
from postgres.utils import PostgresDataFrameWrite
from utils import ItemListDo

# step 2
from utils import ItemDetailsRead
from utils import ItemDetailsDo
from postgres.utils import PostgresDataFrameWrite
from utils import PickleDataRead, PickleDataWrite

# step 2.5
from postgres.utils import PostgresDataFrameRead
from web_scraping.utils import SearchRead
from utils import PopulateQueueDo
from postgres.utils import PostgresDataFrameWrite, PostgresDataFrameUpdate

# step 2.6
from postgres.utils import PostgresDataFrameRead
from web_scraping.utils import ParseRead   # parse details' pages of yandex market
from utils import SetSearchQueueProcessedDo
from mongodb.utils import MongoWrite

# step 3
from mongodb.utils import MongoRead
from postgres.utils import PostgresDataFrameRead
from web_scraping.utils import SearchParseRead
from mongodb.utils import MongoWrite

class SearchDo(Do):
    def process(self, data: Dict[StepNum, Any]) -> Dict[StepNum, Any]:
        d = data.get('step_1').get('step_0')
        df = pd.DataFrame(data=d.values(), columns=['search_query', 'product_yandex_name', 'searched',
                                                    'product_details_yandex_link', 'product_reviews_yandex_link',
                                                    'scraped'])
        df['searched'] = 1
        return {'step_0': df}

class YandexMarketDo(Do):
    #-> Tuple[Dict,List[Dict]]:
    def process(self, data: Dict[str, Tuple[str, Tuple[str, Dict], Tuple[str, List[Dict]]]]) -> Dict[str, List[Dict[str, List[Dict]]]]:
        '''all the logic to process scraping results from yandex market'''
        if 'step_1' in data:
            data = data.get('step_1')
        assert isinstance(data, dict)
        user_queries = list(data.keys())
        results = []
        for user_query in user_queries:
            triple = data.get(user_query)
            logger.critical(triple)
            try:
                assert len(triple) == 3
                item_name = triple[0]
                if triple[1]:
                    # product_details
                    # triple[1]: (url, {details_dict})
                    assert len(triple[1]) == 3
                    details_to_mongo = triple[1][1]
                    details_to_mongo.update({'query_url': triple[1][0], 'query_item_name': item_name})
                    # triple[2] contains bool - interrupted by captcha
                    if triple[2]:
                        # product_reviews
                        # triple[2]: (reviews_url, [{'review1_key1':'review1_value1'}, {'review2_key1':'review2_value1'}])
                        assert len(triple[2]) == 3
                        reviews_url = triple[2][0]
                        init_reviews_list = triple[2][1]
                        for review in init_reviews_list:
                            review.update(
                                {"reviews_url": reviews_url,
                                 "item_name": item_name}
                            )
                    else:
                        init_reviews_list = []
            except Exception as e:
                logger.error(e)
                details_to_mongo = {}
                init_reviews_list = []
            try:
                results.append({"step_0": [details_to_mongo], "step_1": init_reviews_list})
            except Exception as e:
                results.append({"step_0": [[]], "step_1": []})
        return {"step_0": [x.get('step_0') for x in results], "step_1": [x.get("step_1") for x in results]}

class AttemptProductsDo(Do):  # trace decendants after lunch
    def process(self, data: Iterable) -> Dict[StepNum, pd.DataFrame]:
        if isinstance(data, dict):
            if 'step_0' in data:
                data = data.get('step_0')
        df = pd.DataFrame(data, columns=['attempt_product_name'])
        logger.warning(df.head)
        logger.warning(df.shape)
        df['attempt_datetime'] = datetime.now()
        return {"step_0": df}

class ReadChainSearchParsePart1:
    '''implementation for search and parse reviews reader - get mongo (x2), get postgres, calculate urls to search'''
    def __init__(self, readers: List[Read]):
        self.data = {}
        self.readers = readers

    def prepare(self) -> Any:
        for i, reader in enumerate(self.readers):
            self.data[f'step_{i}'] = reader.read()
        return self.data

    def read(self) -> Any:
        # executed in job

        # step 0 is mongo (scraped_data.product_reviews)
        # step 1 is mongo (scraped_data.product_details)
        # step 2 is postgres (scraped_data.product_item_list)
        # step 3 is postgres (scraped_data.product_query_attempts)

        self.prepare()
        already_scraped_names = []
        for data in self.data.get('step_0').get('step_0'):
            already_scraped_names.append(data.get('item_name'))
        for data in self.data.get('step_1').get('step_0'):
            already_scraped_names.append(data.get('query_item_name'))
        already_scraped_names = list(set(already_scraped_names))
        item_list = self.data.get('step_2').get('step_0')['name'].values.tolist()
        if  self.data.get('step_3'):
            already_queried_data = self.data.get('step_3').get('step_0')
        else:
            already_queried_data = []
        # logger.debug(f'item list: {item_list}')
        # logger.debug(f'already_queried_data: {already_queried_data}')
        if already_queried_data is not None and len(already_queried_data) > 0:
            print(already_queried_data)
            already_queried = already_queried_data['attempt_product_name'].values.tolist()
        else:
            # postgres check is disabled
            already_queried = []
        product_names_to_scrape = set(
            [x for x in item_list if x not in already_scraped_names and x not in already_queried])
        assert len(product_names_to_scrape) > 0
        return {'step_0': product_names_to_scrape}


class ReadChainSearchParsePart2:
    def __init__(self, readers: List[Read]):
        self.data = {}
        self.readers = readers

    def read(self,) -> Dict[str, Any]:
        self.data[f'step_0'] = self.readers[0].read()
        self.data[f'step_1'] = self.readers[1].read(
            data=self.data[f'step_0']
        )
        return self.data


class FillInDo(Do):
    def process(self, data: Dict[StepNum, Any]) -> Dict[StepNum, Any]:
        assert isinstance(data, dict)
        item_details = data.get('step_0').get('step_0')  # source, not empty details
        product_item_list_to_fill = data.get('step_1').get('step_0')  # new, empty details
        product_item_list = data.get('step_2').get('step_0')  # old, linked with not empty details

        # etl:
        # step 1 - find exact match
        # step 2 - find aprox match
        # step 3 - assign them to __schema__.product_item_list_to_fill as "etalon_name"
        # step 4 - get them from __schema__.product_item_list_to_fill by product_url
        # step 5 - get __schema__.item_details_washing_machine
        # step 6 - select from step 4 by etalon name
        # step 7 - assign product_url, product_price as min_price, etalon_name as name,

        fill = set(product_item_list_to_fill['product_name'])
        known = set(product_item_list['product_name'])
        # step 1: exact_match
        intersect = fill & known
        fill -= intersect
        known -= intersect
        tfidf = TFIDF()
        mapping = {x.replace('  ', ' '): x for x in list(fill)}

        from_mapping = {x: x for x in list(fill)}
        to_mapping = {x: x for x in list(known)}
        temp_tfidf = tfidf.match(from_list=[x for x in list(fill)],
                                 to_list=[x for x in list(known)])


        # from_mapping = {x.replace('  ', ' '): x for x in list(fill)}
        # to_mapping = {x.replace('  ', ' '): x for x in list(known)}
        # temp_tfidf = tfidf.match(from_list=[x.replace(' ', '') for x in list(fill)],
        #                          to_list=[x.replace(' ', '') for x in list(known)])

        temp_tfidf['OriginalFrom'] = temp_tfidf['From'].map(from_mapping)
        temp_tfidf['OriginalTo'] = temp_tfidf['To'].map(to_mapping)
        temp_tfidf = temp_tfidf[temp_tfidf.Similarity > 0.71]
        product_item_list_to_fill['etalon_name'] = np.nan

        # step 3 - assing them to scraped_data.product_item_list_to_fill as "etalon_name"
        product_item_list_to_fill.loc[product_item_list_to_fill.product_name.isin(intersect), 'etalon_name'] = \
        product_item_list_to_fill.loc[product_item_list_to_fill.product_name.isin(intersect), 'product_name']

        product_item_list_to_fill.loc[
            (product_item_list_to_fill.etalon_name.isnull() & product_item_list_to_fill.product_name.isin(
                temp_tfidf.OriginalFrom)), 'etalon_name'
        ] = product_item_list_to_fill.loc[
            (product_item_list_to_fill.etalon_name.isnull() & product_item_list_to_fill.product_name.isin(
                temp_tfidf.OriginalFrom)), 'product_name'
        ].replace(temp_tfidf.set_index('OriginalFrom')['OriginalTo'].to_dict())

        product_item_list_to_fill.loc[
            (product_item_list_to_fill.product_name.isin(temp_tfidf.OriginalFrom)), 'Similarity'
        ] = product_item_list_to_fill.loc[
            (product_item_list_to_fill.product_name.isin(temp_tfidf.OriginalFrom)), 'product_name'
        ].replace(temp_tfidf.set_index('OriginalFrom')['Similarity'].to_dict())

        # step 4 - get  them from scraped_data.product_item_list_to_fill by product_url
        product_item_list_to_fill = product_item_list_to_fill[product_item_list_to_fill.etalon_name.notnull()]
        # details_new = item_details[
        #     item_details.name.isin(product_item_list_to_fill.etalon_name)]
        # details_new.drop(['product_url', 'offer_count', 'min_price', ], axis=1, inplace=True)
        # details_new = details_new.merge(
        #     product_item_list_to_fill[['etalon_name', 'product_url', 'product_price', 'Similarity']], 'left',
        #     left_on='name', right_on='etalon_name')
        # details_new.drop(['etalon_name', ], axis=1, inplace=True)
        # details_new.rename(columns={'product_price': 'min_price'}, inplace=True)
        # details_new['min_price'] = details_new['min_price'].astype(str).str.replace(" ", "").str.replace(u'\xa0', "").astype(float)
        # details_new['offer_count'] = 1
        # details_new['Similarity'].fillna(1, inplace=True)
        # details_new['ranking'] = details_new.groupby([details_new['product_url'].str[:20], details_new['name']])[
        #     'Similarity'].rank(ascending=False)
        # details_new = details_new[details_new.ranking == 1]
        # details_new.drop(['Similarity', 'ranking'], axis=1, inplace=True)
        # details_new = details_new[item_details.columns]
        product_item_list_to_fill = product_item_list_to_fill.merge(
            product_item_list[['product_url', 'product_name']].rename(columns={'product_url': 'etalon_url'}),
            'left',
            left_on='etalon_name',
            right_on='product_name'

        )
        product_item_list_to_fill = product_item_list_to_fill.merge(
            item_details[['name', 'product_url']].rename(
                columns={'product_url': 'etalon_url', 'name': 'correct_product_name'}),
            'left',
            left_on='etalon_url',
            right_on='etalon_url'

        )
        details_new = item_details \
            .merge(
            product_item_list_to_fill[['etalon_url', 'product_url', 'product_price']],
            # idk wtf 'product_image_url' is not used
            'inner',
            left_on='product_url',
            right_on='etalon_url') \
            .drop(['product_url_x', 'etalon_url', 'min_price'], axis=1) \
            .rename(columns={'product_url_y': 'product_url', 'product_price': 'min_price'}) \
            .assign(offer_count=1)
        details_new = details_new[item_details.columns]
        details_new.drop_duplicates(subset=['name', 'product_url'], inplace=True)
        return {'step_0': details_new}


class ReviewsProductDetailsDo(Do):
    def process(self, data: Dict[StepNum, Any]) -> Dict[StepNum, Any]:
        assert isinstance(data, dict)
        data = data.get('step_0')
        df = pd.DataFrame(data)
        df['product_rating_value'] = df['product_rating_value'].astype(float)
        df['product_rating_count'] = df['product_rating_count'].fillna(0).astype(int)
        df['product_review_count'] = df['product_review_count'].fillna(0).astype(int)
        df['inserted_datetime'] = datetime.now()
        logger.debug(df.columns)
        if 'ObjectId' in df.columns: df.drop('ObjectId', axis=1, inplace=True)
        if '_id' in df.columns: df.drop('_id', axis=1, inplace=True)

        return {"step_0": df}






if __name__ == '__main__':
    # config part
    SCHEMA_NAME = 'vacuumcleaner'
    PRODUCT_TYPE_NAME = 'Пылесос'  # Стиральная машина, Холодильник

    # step 1. ItemList from sites to Postgres. Not: all three for each new product
    #
    # product_type_url = [f'https://shop.by/chayniki/?page_id={i}' for i in range(1, 106)]  # 1,30
    # product_type_url=[f'https://www.21vek.by/teapots/page:{i}/' for i in range(2, 31)]  # 2,11
    # product_type_url=[f'https://catalog.onliner.by/kettle?page={i}' for i in range(86, 120)]  # 2,50
    #
    # ItemlList_2_Postgres = Job(
    #     # reader=EcomItemListRead(extractor_name='ShopByExtractor', product_type_url=product_type_url, product_type_name=PRODUCT_TYPE_NAME),
    #     # reader=EcomItemListRead(extractor_name='Vek21Extractor', product_type_url=product_type_url, product_type_name=PRODUCT_TYPE_NAME),
    #     # reader=EcomItemListRead(extractor_name='OnlinerExtractor', product_type_url=product_type_url, product_type_name=PRODUCT_TYPE_NAME),
    #     processor=ItemListDo(),
    #     writer=PostgresDataFrameWrite(
    #         schema_name=SCHEMA_NAME,
    #         table_name='product_item_list',  # product_item_list_to_fill, product_item_list
    #         insert_unique=True,
    #         index_column="product_url",
    #         if_exists='append'
    #     ),
    # )
    # ItemlList_2_Postgres.run()

    # step = Step1(product_name='waterheater', shop_name='shop')
    # step.run()
    #
    # step = Step1(product_name=SCHEMA_NAME, shop_name='vek21')
    # step.run()

    # step = Step1(product_name=SCHEMA_NAME, shop_name='onliner')
    # step.run()

    #
    # # step 2. Read: ItemList from Postgres, Do: Scrapy ProductDetails, Write: to Postgres
    # logger.warning('Start - Job 2')
    # ItemDetails_2_Postgres = Job(
    #     reader=ItemDetailsRead(
    #         step1__table=f'{SCHEMA_NAME}.product_item_list',
    #         step1__where="product_url ilike '%%shop.by%%'",
    #         step1_urls_attribute='product_url'
    #     ),
    #     processor=ItemDetailsDo(product_type_name=PRODUCT_TYPE_NAME),
    #     writer=PostgresDataFrameWrite(
    #         schema_name=SCHEMA_NAME,
    #         table_name=f'item_details_{SCHEMA_NAME}',
    #         insert_unique=True,
    #         if_exists='append'),
    # )
    # ItemDetails_2_Postgres.run()
    # logger.warning('End - Job 2')


    # # # todo: 2905 after lunch
    # Job 4 -> Fill in the details from the sites that have no product_details microdata
    # DetailsFillIn = Job(
    #     reader=ReadChain(readers=[
    #         PostgresDataFrameRead(table=f'{SCHEMA_NAME}.item_details_{SCHEMA_NAME}', where=''),
    #         PostgresDataFrameRead(table=f'{SCHEMA_NAME}.product_item_list_to_fill', where="product_url not ilike '%%shop.by%%'"),  # item_list with unknown details = fill
    #         PostgresDataFrameRead(table=f'{SCHEMA_NAME}.product_item_list', where="product_url ilike '%%shop.by%%'"),  # item_list with known details = known
    #     ]),
    #     processor=FillInDo(),
    #     writer=PostgresDataFrameWrite(schema_name=SCHEMA_NAME, table_name=f'item_details_{SCHEMA_NAME}', insert_unique=False)
    # )
    # DetailsFillIn.run()
    # logger.warning('step 4')




    # # step 2.5 - записать в очередь
    mongo_read_product_reviews = MongoRead(operation='read', db_name=SCHEMA_NAME, collection_name='product_reviews')
    mongo_read_product_details = MongoRead(operation='read', db_name=SCHEMA_NAME, collection_name='product_details')
    postgres_read_item_list = PostgresDataFrameRead(
        table=f'{SCHEMA_NAME}.item_details_{SCHEMA_NAME}',
        # where=" (offer_count is not null or offer_count is null) and offer_count > 10 and height_cm >= 195 order by min_price asc limit 2000"
    )
    create_queue_table = Job(
        reader=ReadChainSearchParsePart1(readers=[
            mongo_read_product_reviews,
            mongo_read_product_details,
            postgres_read_item_list,
        ]),
        processor=PopulateQueueDo(),
        writer=PostgresDataFrameWrite(
            schema_name=SCHEMA_NAME,
            table_name='search_queue',
            insert_unique=True,
            index_column='search_query',
            if_exists='append',
            dtypes={
                    'search_query': sqlalchemy.types.TEXT,
                    'product_yandex_name': sqlalchemy.types.TEXT,
                    'searched': sqlalchemy.types.BIGINT,
                    'product_details_yandex_link': sqlalchemy.types.TEXT,
                    'product_reviews_yandex_link': sqlalchemy.types.TEXT,
                    'scraped': sqlalchemy.types.BIGINT,
                    }
        ),
    )
    create_queue_table.run()


    # step 2.6 - populate with google search
    # populate_queue_table = Job(
    #     reader=ReadChain(
    #         readers=[
    #             PostgresDataFrameRead(table=f'{SCHEMA_NAME}.search_queue', where="(searched is null or searched = 0) and lower(search_query) similar to '%%(lg|bosch|electrolux|samsung)%%'"),
    #             SearchRead()  # output: Dict[user_query, tuple(fridge.search_queue attributes)]
    #         ]),
    #     processor=SearchDo(),
    #     writer=PostgresDataFrameUpdate(
    #         schema_name=SCHEMA_NAME,
    #         table_name='search_queue',
    #         where_postgres_attribute='search_query',
    #         where_dataframe_column_name='search_query',
    #     )
    # )
    # populate_queue_table.run()


    # step 2.7
    # parse_yandex = Job(
    #     reader=ReadChainSearchParsePart2(readers=[
    #         PostgresDataFrameRead(
    #             table=f'{SCHEMA_NAME}.search_queue',
    #             where='searched = 1 and scraped = 0 and LENGTH(product_details_yandex_link) >= 10'
    #         ),
    #         ParseRead(),
    #     ]),
    #     processor=DoChainGlobal(processors=[
    #         SetSearchQueueProcessedDo(),
    #         YandexMarketDo()
    #     ]),
    #     writer=WriteChain(writers=[
    #         PostgresDataFrameUpdate(schema_name=SCHEMA_NAME, table_name='search_queue',
    #                                 where_dataframe_column_name='product_details_yandex_link', where_postgres_attribute='product_details_yandex_link'),
    #         WriteChain(writers=[
    #             MongoWrite(operation='write', db_name=SCHEMA_NAME, collection_name='product_details'),
    #             MongoWrite(operation='write', db_name=SCHEMA_NAME, collection_name='product_reviews')
    #         ]),  # details & reviews
    #     ],
    #     )
    # )
    # parse_yandex.run()


    # ----------remove------------------------------------------------------------------------------------------------------
    # step 3.A - prepare data, save to pickle
    # Read: 3.1
    # mongo_read_product_reviews = MongoRead(operation='read', db_name='fridge', collection_name='product_reviews')
    # # Read 3.2
    # mongo_read_product_details = MongoRead(operation='read', db_name='fridge', collection_name='product_details')
    # # Read: 3.3
    # postgres_read_item_list = PostgresDataFrameRead(
    #     table='fridge.item_details_fridge',
    #     where=" (offer_count is not null or offer_count is null) and offer_count > 10 and height_cm >= 195 order by min_price asc limit 2000"
    # )
    # # Read: 3.4
    # postgres_read_query_attempts = PostgresDataFrameRead(table='fridge.product_query_attempts')
    # # Part 3A
    # logger.warning('Start - Job 3A')
    # scrape_internet_part_A = Job(
    #     reader=ReadChainSearchParsePart1(readers=[
    #         mongo_read_product_reviews,
    #         mongo_read_product_details,
    #         postgres_read_item_list,
    #         # postgres_read_query_attempts
    #     ]),
    #     writer=PickleDataWrite(filepath='temp_2005_A.pkl'),
    # )
    # scrape_internet_part_A.run()
    # logger.warning('End - Job 3A')
    #
    # # # step 3B
    # logger.warning('Start - Job 3B')
    # internet_reader = SearchParseRead()
    # scrape_internet_part_C = Job(
    #     reader=ReadChainSearchParsePart2(readers=[
    #         PickleDataRead(filepath='temp_2005_A.pkl'),
    #         internet_reader,
    #     ]),
    #     processor=DoChain(processors=[
    #         AttemptProductsDo(),
    #         YandexMarketDo()
    #     ]),
    #     writer=WriteChain(writers=[
    #             PostgresDataFrameWrite(schema_name='fridge', table_name='product_query_attempts', insert_unique=True),  # todo: bug when False
    #             WriteChain(writers=[
    #                 MongoWrite(operation='write', db_name='fridge', collection_name='product_details'),
    #                 MongoWrite(operation='write', db_name='fridge', collection_name='product_reviews')
    #             ]),  # details & reviews
    #         ],
    #     )
    # )
    # scrape_internet_part_C.run()
    # logger.warning('End - Job 3B')
    # ----------------remove------------------------------------------------------------------------------------------------



    #
    # # Job 5 - Mongo Details -> PostgresDetails
    # ReviewsProductDetails = Job(
    #     reader=MongoRead(operation='read', db_name='scraped_data', collection_name='product_details'),
    #     processor=ReviewsProductDetailsDo(),
    #     writer=PostgresDataFrameWrite(
    #         schema_name='scraped_data',
    #         table_name='reviews_product_details',
    #         insert_unique=True,
    #         index_column='product_url'
    #     )
    # )
    # ReviewsProductDetails.run()
    #
    # # todo: correct empty rating from postgres view - decide in Job 3 what to scrape based on that
    #
    # # Job 6 - to complete -
