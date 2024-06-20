# todo: ? save html bodies to mongo

import sys

import pandas as pd

sys.path.append('/home/amstel/llm')
import concurrent
import concurrent.futures as pool


from scrapy.crawler import CrawlerProcess
from loguru import logger
import extruct
import scrapy
from scrapy.spiders import CrawlSpider
# from bs4 import BeautifulSoup
# import requests
from typing import Dict, List, Any, Set, Tuple
from scrapy.exceptions import CloseSpider
import logging
import pickle
# from src.postgres.postgres_utils import select_data
ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
from abc import abstractmethod
from etl_jobs.base import Read, StepNum
from .search_and_parse_reviews import thread_work
from .search_and_parse_reviews import search_google_parse_results, parse_yandex_market

class JsonldExtractor:
    """
    Must be Jsonld
    Returns Name of web page proudct
    """

    @staticmethod
    def page_product(jdata: Dict) -> str:
        """Run at step 1(a) - get category name"""
        for el in jdata.get('json-ld'):
            if el.get('@type') == 'Product':
                return el.get('name')

    @staticmethod
    def get_jdata(body):
        return extruct.extract(body, syntaxes=['json-ld'])


class MicrodataExtractor:
    """extract microdayta"""

    @staticmethod
    def get_mdata(body):
        return extruct.extract(body, syntaxes=['microdata'])

    @staticmethod
    def breadcrumbs(mdata: Dict) -> List[Dict[str, str]]:
        '''
        Must be Microdata
        Run at step 1(b) - get website structure
        '''
        website_structure = []
        for el in mdata['microdata']:
            if el.get('type') == 'http://schema.org/BreadcrumbList':
                _properties = el.get('properties')
                _items_list = _properties.get('itemListElement')
                for _item in _items_list:
                    item_properties = {}
                    _item_properties = _item.get('properties')
                    item_properties['product_url'] = _item_properties.get('item')
                    item_properties['name'] = _item_properties.get('name')
                    item_properties['position'] = _item_properties.get('position')
                    website_structure.append(item_properties)
        return website_structure

    @logger.catch
    @staticmethod
    def item_list(mdata: Dict,
                  base_url: str,
                  product_type_url: str = None,
                  product_type_name: str = None,
        ) -> List[Dict[str, str]]:
        '''
        Must be Microdata
        Run as step 1(c) - get item list

        :param mdata = extuct.extract Dict
        :param product_type_url -- shop.by/washing_machine
        :param product_type_name -- Стиральные машины
        :returns results List
        '''
        results = []
        for el in mdata['microdata']:
            if el.get('type') == 'https://schema.org/ItemList':
                _items = el['properties'].get('itemListElement')
                for item in _items:
                    if item.get('type') == 'https://schema.org/ListItem':
                        item_features = {}
                        item_properties = item.get('properties')
                        rel_item_url = item_properties.get('url')
                        item_features['product_url'] = base_url + rel_item_url.rstrip('#shop')
                        item_features['product_name'] = item_properties.get('name')
                        item_features['product_position'] = item_properties.get('position')
                        item_features['product_type_url'] = product_type_url
                        item_features['product_type_name'] = product_type_name
                        try:
                            item_features['product_image_url'] = base_url + item_properties.get('image').get('properties').get('image')
                        except:
                            item_features['product_image_url'] = None
                        results.append(item_features)
        return results

    @staticmethod
    def product(mdata: Dict) -> Dict[str, str]:
        '''
        Run as step 2 - get product details

        :param mdata = extuct.extract Dict
        :returns item_features Dict
        '''
        item_features = {}  # to put in sql table
        for el in mdata['microdata']:
            if el.get('type') == 'https://schema.org/Product':
                _item_properties = el['properties']
                item_features['name'] = _item_properties.get('name')
                item_features['product_url'] = _item_properties.get('url')
                # item_features['image_url'] = _item_properties.get('image')  # let's get image url from item_list
                _offers_properties = _item_properties.get('offers').get('properties')
                item_features['offer_count'] = _offers_properties.get('offerCount')
                item_features['min_price'] = _offers_properties.get('lowPrice')
                _additional_properties = _item_properties.get('additionalProperty')
                for ap_dict in _additional_properties:
                    kv = ap_dict.get('properties')
                    key = kv.get('name')
                    val = kv.get('value')
                    item_features[key] = val
        return item_features

# todo: extend MicrodataExtractor for other sites
class HtmlExtractor:
    def __init__(self, product_type_url: str | List[str], product_type_name):
        '''
        :param product_type_url: str or list of str -- links to parse. Pages must contain item lists
        :param product_type_name:
        '''
        self.product_type_url = product_type_url
        self.product_type_name = product_type_name


    @abstractmethod
    def item_list(self, mdata, response, product_type_url, product_type_name) -> List[Dict[str, str]]:
        """
        Extract from body.

        Output dict keys:
        product_url
        product_name
        product_position
        self.product_type_url
        self.product_type_name
        """
        ...

class OnlinerExtractor(HtmlExtractor):
    base_url = 'https://catalog.onliner.by/washingmachine'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def item_list(
            self,
            mdata,
            response,
            product_type_url,
            product_type_name,
        ) -> List[Dict[str, str]]:
        results = []
        for i, selector in enumerate(response.css('div.catalog-form__offers-unit')):
            item_features = {}
            # maybe a.catalog-form__link_primary-additional.catalog-form__link_base-additional
            item_features['product_url'] = selector.css('a.catalog-form__link_base-additional::attr(href)').get()
            item_features['product_image_url'] = selector.css('img.catalog-form__image::attr(src)').get()
            item_features['product_name'] = selector.css('a.catalog-form__link_base-additional::text').get().strip()
            # item_features['product_price'] = selector.css('a.catalog-form__link span:not([class^="catalog-form__description"])::text').get().replace('\xa0р.','').replace(',','.')
            item_features['product_type_url'] = product_type_url
            item_features['product_type_name'] = self.product_type_name
            item_features['product_position'] = i
            results.append(item_features)
        return results

class Vek21Extractor(HtmlExtractor):
    base_url = 'https://www.21vek.by'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def item_list(
            self,
            mdata,
            response,
            product_type_url,
            product_type_name,
    ) -> List[Dict[str, str]]:
        results = []
        logger.warning(f'response type {type(response)}')
        logger.warning(f'response {(response)}')
        for i, selector in enumerate(response.css('div[class^="ListingProduct_product"]')):
            item_features = {}
            item_features['product_url'] = self.base_url + selector.css('p[class^="CardInfo"] a::attr(href)').get()
            item_features['product_image_url'] = selector.css('div[class^="CardMedia_mediaContainer"] img::attr(src)').get()
            item_features['product_name'] = selector.css('p[class^="CardInfo"] a::text').get()
            # item_features['product_price'] = selector.css('p[class^="CardPrice_currentPrice"]::text').get().replace('р.', '').replace(',','.').strip()
            item_features['product_type_url'] = product_type_url
            item_features['product_type_name'] = product_type_name
            item_features['product_position'] = i
            results.append(item_features)
        return results

class ShopByExtractor(HtmlExtractor):
    base_url = 'https://shop.by'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def item_list(
            self,
            mdata,
            response,
            product_type_url,
            product_type_name,
        ) -> List[Dict[str, str]]:

        results = MicrodataExtractor.item_list(
            mdata=mdata,
            base_url=self.base_url,
            product_type_url=product_type_url,
            product_type_name=product_type_name,
        )
        return results

class ItemListSpider(CrawlSpider):
    name = "item_list"

    def __init__(self, *a, **kw):
        super(ItemListSpider, self).__init__(*a, **kw)
        logger.warning(f'args: {a}')
        logger.warning(f'kwargs: {kw}')

        self.page_product = []
        self.breadcrumbs = []
        self.item_list = []
        self.output_callback = kw.get('args').get('callback')
        self.extractor_instances = kw.get('args').get('extractor_instances')
        logger.warning(f'extractor_instances: {self.extractor_instances}')


    def start_requests(self):
        # todo: how to check if it's already scraped whem the site redirect the out-of-bound page number to the last valid
        # todo: construct urls

        self.url_2_extractor_instance = {}
        urls = []
        for extractor_instance in self.extractor_instances:
            assert extractor_instance.product_type_url
            if isinstance(extractor_instance.product_type_url, str):
                urls.append(extractor_instance.product_type_url)
                self.url_2_extractor_instance[extractor_instance.product_type_url] = extractor_instance
            elif isinstance(extractor_instance.product_type_url, list):
                urls.extend(extractor_instance.product_type_url)
                for extractor_instance_product_type_url in extractor_instance.product_type_url:
                    self.url_2_extractor_instance[extractor_instance_product_type_url] = extractor_instance

        for url in urls:
            self.current_url = url
            logger.warning(url)
            yield scrapy.Request(url=url, callback=self.parse,)

    def parse(self, response):
        if response.status == 404:
            raise CloseSpider('Receive 404 response')

        extractor_instance = self.url_2_extractor_instance[self.current_url]
        logger.info(type(response))
        with open('response.pkl', 'wb') as f:
            pickle.dump(response.text, f)

        jdata = JsonldExtractor.get_jdata(response.body)
        mdata = MicrodataExtractor.get_mdata(response.body)
        product_type_name = JsonldExtractor.page_product(jdata)
        breadcrumbs = MicrodataExtractor.breadcrumbs(mdata)

        # todo: get extractor instances
        item_list = extractor_instance.item_list(
            mdata=mdata,
            response=response,
            product_type_url=self.current_url,
            product_type_name=extractor_instance.product_type_name,
        )

        self.page_product.append([extractor_instance.product_type_name, self.current_url])
        self.breadcrumbs.append(breadcrumbs)
        self.item_list.append(item_list)
        self.output = self.item_list
        # logger.warning(product_type_name)

    def close(self, spider, reason):
        self.output_callback(
            # (
                # self.page_product,
                # self.breadcrumbs,
                self.item_list
            # )
        )

class ProductSpider(CrawlSpider):
    name = 'product'

    def __init__(self, *a, **kw):
        logger.critical(kw)
        # get_url_from_db = kw.get('args').get('get_url_from_db')
        # sql_from_table = kw.get('args').get('sql_from_table')
        # where_clause = kw.get('args').get('where_clause')
        #
        # logger.info(f'get_url_from_db: {get_url_from_db}')

        super(ProductSpider, self).__init__(*a, **kw)
        self.products = []
        self.output_callback = kw.get('args').get('callback')

        if kw.get('args').get('urls'):
            self.urls = kw.get('args').get('urls')
            # assert sql_from_table
            # assert where_clause
            # logger.debug(sql_from_table, where_clause)
            # df = select_data(table=sql_from_table, where=where_clause)
            # if df.shape[0] > 0:
            #     self.urls = df['product_url'].tolist()
        else:
            self.urls = []

    def start_requests(self):
        if not self.urls:
            self.urls = [
                'https://shop.by/stiralnye_mashiny/atlant_sma_60u1214_01/'
            ]
        for url in self.urls:
            yield scrapy.Request(url=url, callback=self.parse)

    @logger.catch
    def parse(self, response):
        if response.status == 404:
            raise CloseSpider('Recieve 404 response')
        mdata = MicrodataExtractor.get_mdata(response.body)
        product = MicrodataExtractor.product(mdata,)
        self.products.append(product)

    def close(self, spider, reason):
        self.output_callback(self.products)

class CustomCrawler:
    def __init__(self, **kwargs):
        self.output = None
        self.crawl_args_update = kwargs
        # logger.warning(f'kwargs: {kwargs}')
        self.process = CrawlerProcess({
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36',
            'FEED_FORMAT': 'csv',
            'FEED_URI': 'output.csv',
            'DEPTH_LIMIT': 2,
            # 'CLOSESPIDER_PAGECOUNT': 3,
            # 'DOWNLOAD_DELAY': 2,  # minimum download delay
        })

    def yield_output(self, data):
        self.output = data

    def crawl(self, cls):
        # self.process.get_url_from_db = self.get_url_from_db
        crawl_args = {'callback': self.yield_output}
        crawl_args.update(self.crawl_args_update)
        self.process.crawl(cls, args=crawl_args)
        self.process.start()

def crawl_static(cls, **kwargs):
    # get_url_from_db = kwargs.get('get_url_from_db')
    crawler = CustomCrawler(**kwargs)
    crawler.crawl(cls)
    return crawler.output

class EcomItemListRead(Read):
    '''ETL Read interface for item list'''
    def __init__(self, extractor_name: str, product_type_url: str | List[str], product_type_name: str):
        if extractor_name == 'ShopByExtractor':
            extractor_instance = ShopByExtractor(product_type_url=product_type_url, product_type_name=product_type_name)
        elif extractor_name == 'Vek21Extractor':
            extractor_instance = Vek21Extractor(product_type_url=product_type_url, product_type_name=product_type_name)
        elif extractor_name == 'OnlinerExtractor':
            extractor_instance = OnlinerExtractor(product_type_url=product_type_url, product_type_name=product_type_name)
        else:
            raise ValueError
        self.extractor_name = extractor_name
        self.extractor_instance = extractor_instance

    def read(self,) -> List[Dict]:
        data = crawl_static(
            ItemListSpider,
            extractor_instances=[self.extractor_instance,]
        )
        if isinstance(data, list) and isinstance(data[0], list):
            data = [item for sublist in data for item in sublist]
        return {"step_0": data}

class EcomProductRead(Read):
    def read(self, urls: List[str]) -> Dict[StepNum, Any]:
        out = crawl_static(
            ProductSpider,
            urls=urls
        )
        return out

    # def read_threading(self, product_names_to_scrape: Set) -> Any:
    #     ex = pool.ThreadPoolExecutor(max_workers=1,
    #                                  thread_name_prefix='thread_',
    #                                  initializer=None, initargs=())
    #
    #     # Read 5
    #     with ex as executor:
    #         future_to_url = {executor.submit(thread_work, user_query): user_query for user_query in product_names_to_scrape}
    #     triplets = {}
    #     for future in concurrent.futures.as_completed(future_to_url):
    #         url = future_to_url[future]
    #         try:
    #             data = future.result()
    #             triplets[url] = data
    #         except Exception as exc:
    #             print(f'{url} сгенерировано исключение: {exc}')


class SearchParseRead(Read):
    def read(self, data: Dict[StepNum, str]) -> Tuple[str, Tuple[str, Dict], Tuple[str, List[Dict]]]:
        assert isinstance(data, dict)
        self.triplets = {}
        product_names_to_scrape = data.get('step_0')
        for user_query in product_names_to_scrape:
            # Tuple[str, Tuple[str, Dict], Tuple[str, List[Dict]], bool]
            try:
                output = thread_work(user_query)
                logger.critical(output)
                triplet = output[:3]
                INTERRUPT = output[-1]
                self.triplets[user_query] = triplet
                logger.critical(f"interrupt? {user_query} -- {INTERRUPT}")
                if INTERRUPT:
                    break
            except Exception as e:
                pass
        return self.triplets

class SearchRead(Read):
    # the same as SearchParseRead but only search
    def read(self, data: Dict[StepNum, str]) -> Dict[StepNum, Dict[str, tuple]]:
        assert isinstance(data, dict)

        # cols: search_query, product_yandex_name, processed, product_details_yandex_link, product_reviews_yandex_link
        product_url_2_details = {}
        search_queue_df = data.get('step_0').get('step_0')
        logger.debug(search_queue_df.columns)
        for user_query in search_queue_df['search_query'].unique():
            try:
                product_url, reviews_url, product_yandex_name, interrupt = search_google_parse_results(user_query)
            except Exception as e:
                product_url_2_details[user_query] = (user_query, product_yandex_name, 1, product_url, reviews_url, 0)
            if interrupt:
                return {'step_0': product_url_2_details}
            if product_url:
                product_url_2_details[user_query] = (user_query, product_yandex_name, 1, product_url, reviews_url, 0)
            else:
                product_url_2_details[user_query] = (user_query, product_yandex_name, 0, product_url, reviews_url, 0)
        return {'step_0': product_url_2_details}




class ParseRead(Read):
    def read(self, data: Dict[StepNum, str]) -> Tuple[str, Tuple[str, Dict], Tuple[str, List[Dict]]]:
        assert isinstance(data, dict)
        self.triplets = {}
        logger.debug(type(data))
        df = data.get('step_0')
        assert isinstance(df, pd.DataFrame)
        df = df[df['product_details_yandex_link'].notnull()]
        df.drop_duplicates(subset='product_details_yandex_link', inplace=True)
        # data structure is: df with columns = (user_query, product_yandex_name, processed[int], product_url, reviews_url)
        for i, row in df.iterrows():
            user_query, product_yandex_name, searched, product_url, reviews_url, scraped = row
            if scraped == 0:
                try:
                    output = parse_yandex_market(user_query=user_query, product_url=product_url, reviews_url=reviews_url)
                    logger.critical(output)
                    triplet = output[:3]
                    INTERRUPT = output[-1]
                    self.triplets[user_query] = triplet
                    logger.critical(f"interrupt? {user_query} -- {INTERRUPT}")
                    if INTERRUPT:
                        break
                except Exception as e:
                    pass
        return self.triplets


if __name__ == '__main__':
    # logging.getLogger(__name__).setLevel(logging.CRITICAL)
    # onliner_extractor = OnlinerExtractor(
    #     product_type_url='https://catalog.onliner.by/washingmachine',
    #     product_type_name='Стиральная машина'
    # )
    # vek_21_extractor = Vek21Extractor(
    #     product_type_url='https://www.21vek.by/washing_machines/',
    #     product_type_name='Стиральная машина'
    # )
    # shopby_extractor = ShopByExtractor(
    #     product_type_url=[f'https://shop.by/stiralnye_mashiny/?page_id={i}' for i in range (1,30)],
    #     product_type_name='Стиральная машина',
    # )
    # # ItemList
    # out = crawl_static(
    #     ItemListSpider,
    #     extractor_instances=[
    #         shopby_extractor,
    #         # onliner_extractor,
    #         # vek_21_extractor
    #     ]
    # )
    #
    # with open('/home/amstel/llm/out/scrape_ecom__shopby_extractor__washing_machine.pkl', 'wb') as f:  #ItemListSpider
    #     pickle.dump(out, f)
    # logger.info(len(out))
    # logger.info(out[:5])

    # ProductSpider
    # параметры определяют какие айтемы парсим
    # out = crawl_static(
    #     ProductSpider,
    #     get_url_from_db=True,
    #     sql_from_table='scraped_data.product_item_list',
    #     where_clause='crawl_id <= 2',
    # )

    # with open('output_products.pkl', 'wb') as f:
    #     pickle.dump(out, f)
    # logger.info(len(out))


    # 15 05 2024 example usage
    raise ValueError
    processor_1 = EcomItemListRead(extractor_name, product_type_url, product_type_name)
    processor_2 = EcomProductRead(urls)
