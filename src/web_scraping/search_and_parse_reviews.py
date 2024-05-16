# todo: ? save html bodies to mongo
import sys
sys.path.append('/home/amstel/llm/src')
IS_HEADLESS = False
BROWSER_SELECT = 'chromium'
import os

# import sys
# sys.path.append('/home/amstel/llm')
import asyncio
import concurrent.futures

import re
import bs4
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from loguru import logger
from typing import Dict, List, Callable, Any
from bs4 import BeautifulSoup
import extruct
# from src.postgres.utils import select_data  # todo: refactor
import rapidfuzz
import concurrent.futures as pool
Url = str
import random
import numpy as np

from mongodb.utils import MongoConnector
# from postgres.utils import insert_data  # todo: refactor
import pandas as pd
from datetime import datetime


uas = [
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
]
# curl -x 152.170.208.188:8080  http://example.com
PROXIES = [
    {"server": "152.170.208.188:8080"},
    {"server": "51.210.19.141:80"},
]

@logger.catch
def parse_product(mdata) -> Dict[str, str]:
    '''return product details from microdata'''
    for el in mdata:
        if el.get('type') == 'https://schema.org/Product':
            if 'properties' in el:
                _properties = el.get('properties')
                product_category = _properties.get('category')
                product_brand = _properties.get('brand')
                product_name = _properties.get('name')
                product_url = _properties.get('url')
                product_image_url = _properties.get('image')
                _aggregate_rating = _properties.get('aggregateRating') #.get('properties')
                if _aggregate_rating and 'properties' in _aggregate_rating:
                    _aggregate_rating_properties = _aggregate_rating.get('properties')
                    product_rating_value = _aggregate_rating_properties.get('ratingValue')
                    product_best_rating = _aggregate_rating_properties.get('bestRating')
                    product_worst_rating = _aggregate_rating_properties.get('worstRating')
                    product_rating_count = _aggregate_rating_properties.get('ratingCount')
                    product_review_count = _aggregate_rating_properties.get('reviewCount')
                else:
                    product_rating_value = None
                    product_best_rating = None
                    product_worst_rating = None
                    product_rating_count = None
                    product_review_count = None
                return {
                    'product_category':product_category,
                    'product_brand':product_brand,
                    'product_name':product_name,
                    'product_url':product_url,
                    'product_image_url':product_image_url,
                    'product_rating_value':product_rating_value,
                    'product_best_rating':product_best_rating,
                    'product_worst_rating':product_worst_rating,
                    'product_rating_count':product_rating_count,
                    'product_review_count':product_review_count,
                }

@logger.catch
def parse_reviews(mdata) ->  List[dict[str, Any]]:
    '''returns reviews'''
    for el in mdata:
        if el.get('type') == 'https://schema.org/Product':
            if 'properties' in el:
                total_reviews = []
                _properties = el.get('properties')
                _reviews = _properties.get('review')
                if hasattr(_reviews, '__iter__'):
                    for review in _reviews:
                        review_details = {}
                        if 'properties' in review:
                            _review_properties = review.get('properties')
                            review_details['review_date_published'] = _review_properties.get('datePublished')
                            review_details['review_description'] = _review_properties.get('description')
                            if 'reviewRating' in _review_properties:
                                _review_properties_rating = _review_properties.get('reviewRating')
                                if 'properties' in _review_properties_rating:
                                    _review_rating_properties = _review_properties_rating.get('properties')
                                    review_details['review_ratng_value'] = _review_rating_properties.get('ratingValue')
                                    review_details['review_best_rating'] = _review_rating_properties.get('bestRating')
                        total_reviews.append(review_details)
                return total_reviews

@logger.catch
def scrape_page_playwright(url_path: Url, parse_func: Callable) -> List[tuple[Url, Dict | List[Dict]]]:
    parsing_output = []
    with sync_playwright() as p:
        browser_select = random.sample(['firefox',], 1)[0]  #  'firefox',
        browser_select = BROWSER_SELECT
        if browser_select == 'chromium':
            proxy = random.choice(PROXIES)
            logger.critical(f'chosen proxy: {proxy}')
            browser = p.chromium.launch(headless=IS_HEADLESS,
                                        # proxy=proxy
                                        )
        elif browser_select == 'firefox':
            proxy = random.choice(PROXIES)
            logger.critical(f'chosen proxy: {proxy}')
            browser = p.firefox.launch(headless=IS_HEADLESS,
                                       # proxy=proxy
                                       )
        ua = random.sample(uas, 1)[0]
        context = browser.new_context(
            user_agent=ua,
            #
            java_script_enabled=True,  # ?
            bypass_csp=True,
            # extra_http_headers={},
        )
        logger.warning(ua)
        # context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()
        # page.wait_for_timeout(int(np.random.uniform(0, 1000, 1)[0]))
        page.goto(url_path, referer='https://yandex.by/',)

        page.wait_for_timeout(int(np.random.uniform(2500, 5000, 1)[0]))
        # page.screenshot(path="now.png")
        content = page.content()
        try:
            with open(f"/home/amstel/llm/out/htmls/{url_path.replace('/', '-')}.html", 'w') as f:
                f.write(content)
        except:
            pass
        soup = BeautifulSoup(content, "html.parser")
        mdata = extruct.extract(soup.prettify(), syntaxes=['microdata']).get('microdata')
        # if parse_func==parse_product: parsing_output -> Dict
        # if parse_func==parse_review: parsing_output -> List[Dict]
        parsing_output = parse_func(mdata)
        # assert parsing_output  # if error, parsing failed, check microdata of the page
        context.close()
        browser.close()
    return (url_path, parsing_output)


def clear_str(s):
    return s.replace("Стоит ли покупать", "").replace("? Отзывы на Яндекс Маркете", "").strip().lower()


def google_find_link(soup, user_query) -> str:





    # articles = soup.find('div', id='rso').find_all(class_='MjjYud')
    articles = soup.find_all(class_='MjjYud')
    # articles = soup.find_all(attrs={'class': ['MjjYud', 'hlcw0c']})
    # articles = soup.find_all(has_any_class)

    id_2_href = {}
    id_2_text_primary = {}  # results containing 'reviews' - reliable
    id_2_text_secondary = {}  # results not containing 'reviews' - unreliable

    id_2_score_primary = {}  # simple relevancy score
    id_2_score_secondary = {}
    # logger.debug(f'len of articles: {len(articles)}')
    for article_id, article in enumerate(articles):
        try:
            article_href = article.find('a').get('href').replace('search?q=', '').replace('url?q=', '').strip('/')
            article_text = article.find('h3').text
            # step 1 - check if link is good
            # logger.debug(f'candidate reviews link: {article_href}')
            if ('market.yandex.' in article_href) and \
                    ('/reviews' in article_href) and \
                    ('text?search' not in article_href) and \
                    ('search?text' not in article_href):
                # primary search branch - contains reviews
                # logger.debug(f'passed PRIMARY the check for reviews: {article_href} -- {article_text}', )
                id_2_href[article_id] = article_href
                id_2_text_primary[article_id] = article_text  #-- што эта за хня
                # https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html
                id_2_score_primary[article_id] = rapidfuzz.fuzz.token_set_ratio(
                    user_query,
                    article_href,  # seacrh against href because in google end text is truncated with dots
                    processor=rapidfuzz.utils.default_process
                )
            elif ('market.yandex.' in article_href) and \
                    ('/reviews' not in article_href) and \
                    ('text?search' not in article_href) and \
                    ('search?text' not in article_href):
                # secondary seatch branch
                # logger.debug(f'passed SECONDARY the check for reviews: {article_href} -- {article_text}', )
                id_2_href[article_id] = article_href
                id_2_text_secondary[article_id] = article_text
                # https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html
                id_2_score_secondary[article_id] = rapidfuzz.fuzz.token_set_ratio(
                    user_query,
                    article_href,  # seacrh against href because in google end text is truncated with dots
                    processor=rapidfuzz.utils.default_process
                )
        except:
            pass
    if len(id_2_score_primary) > 0:
        # logger.critical(sorted(id_2_text.items(), key=lambda x: x[1], reverse=True))
        best_id, best_score = sorted(id_2_score_primary.items(), key=lambda x: x[1], reverse=True)[0]
        best_text = id_2_text_primary[best_id]
        best_href = id_2_href[best_id]
    elif len(id_2_score_secondary) > 0:
        best_id, best_score = sorted(id_2_score_secondary.items(), key=lambda x: x[1], reverse=True)[0]
        best_text = id_2_text_secondary[best_id]
        best_href = id_2_href[best_id]
    else:
        # no results found
        best_id = None
        best_score = None
        best_text = None
        best_href = None
    logger.info(f'{best_id}, {best_score}, {best_text}, {best_href}')
    return best_href


def search_google(user_query: str) -> str:
    # todo: playwright-scrapy

    # duck
    # url = 'https://duckduckgo.com/'
    # params = {
    #     't': 'h_',
    #     'q': 'site:market.yandex.ru отзывы ' + user_query,
    #     'ia': 'web',
    # }

    # google
    url = 'https://google.com/search'
    params = {
        'q': 'site: market.yandex.ru отзывы ' + user_query,
    }

    for i, (k, v) in enumerate(params.items()):
        if i == 0:
            prefix = '?'
        else:
            prefix = '&'
        url += prefix + str(k) + '=' + str(v).replace(' ', '+')

    with sync_playwright() as p:
        # browser_select = random.sample(['firefox',], 1)[0]  # 'firefox'
        browser_select = BROWSER_SELECT
        if browser_select == 'chromium':
            proxy = random.choice(PROXIES)
            logger.critical(f'chosen proxy: {proxy}')
            browser = p.chromium.launch(headless=IS_HEADLESS,
                                        # proxy=proxy
                                        )
        else:
            proxy = random.choice(PROXIES)
            logger.critical(f'chosen proxy: {proxy}')
            browser = p.firefox.launch(headless=IS_HEADLESS,
                                       # proxy=proxy
                                       )
        ua = random.sample(uas, 1)[0]
        context = browser.new_context(
            user_agent=ua,
            #
            java_script_enabled=True,  # ?
            bypass_csp=True,
            # extra_http_headers={},
        )
        logger.warning(ua)
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()
        # page.wait_for_timeout(int(np.random.uniform(0, 1000, 1)[0]))
        page.goto(url, referer='https://yandex.by/',)
        page.wait_for_timeout(int(np.random.uniform(2500, 5000, 1)[0]))
        content = page.content()
        try:
            with open(f"/home/amstel/llm/out/htmls/{user_query.replace('/', '-')}.html", 'w') as f:
                f.write(content)
        except:
            pass
        # with open("duck_res.html") as fp:
        #     soup = BeautifulSoup(fp, "html.parser")
        soup = BeautifulSoup(content, "html.parser")
        # logger.debug('in google_find_link')
        best_link = google_find_link(soup=soup, user_query=user_query)
        # logger.debug(f'best link: {best_link}')
        context.close()
        browser.close()
    return best_link
    # except Exception as e:
    #     logger.critical(f'error: {e}')
    #     return ''


@logger.catch
def thread_work(user_query):
    '''
    We don't know if the returned value actually links to reviews
    '''
    some_url = search_google(user_query=user_query)  # may return None if search fails
    # logger.critical(f'some url: {some_url}')
    if some_url:
        if '/reviews' in some_url:
            logger.info('branch reviews')
            logger.info(user_query, )
            reviews_url  = some_url[:some_url.find('/reviews')]
            product_url = some_url[:some_url.find('/reviews')] + '/reviews'
            logger.info(f'input_url: {some_url}')
            logger.info(f'reviews_url: {reviews_url}')
            logger.info(f'product_url: {product_url}')
        elif '?sku=' in some_url:
            logger.warning('branch ?sku=')
            logger.warning(user_query, )
            product_url = some_url[:some_url.find('?sku=')]
            reviews_url = product_url + '/reviews'
            logger.warning(f'input_url {some_url}')
            logger.warning(f'reviews_url: {reviews_url}')
            logger.warning(f'product_url: {product_url}')
        else:
            logger.critical(f'v1 - no results for: {user_query}')
            return (user_query, [], [])
        product_details_list = scrape_page_playwright(url_path=product_url, parse_func=parse_product)
        review_details_list = scrape_page_playwright(url_path=reviews_url, parse_func=parse_reviews)
        if not review_details_list[-1]:  # list of reviews
            # this may happen if the reviews_url logic is faulty or when product names are inconsistent and redirects happen
            prod_det = product_details_list[-1]
            if prod_det:
                correct_product_url = prod_det.get('product_url')
                # logger.warning(correct_product_url)
                if correct_product_url:
                    correct_reviews_url = correct_product_url + '/reviews'
                    review_details_list = scrape_page_playwright(url_path=correct_reviews_url, parse_func=parse_reviews)
                    # logger.warning(correct_reviews_url)
                    # logger.warning(review_details_list)
        return (user_query, product_details_list, review_details_list)
    logger.critical(f'v2 - no results for: {user_query}')
    return (user_query, [], [])


if __name__ == '__main__':
    # todo: make composable read class

    # steps:
    # 1. get product_details & product_reviews - we do not need to query them again
    # # READ 1
    # con_product_reviews = MongoConnector(operation='read', db_name='scraped_data', collection_name='product_reviews')
    # cursor_product_reviews = con_product_reviews.read_many({})
    # product_reviews = list(cursor_product_reviews)
    #
    # # READ 2
    # con_product_details = MongoConnector(operation='read', db_name='scraped_data', collection_name='product_details')
    # cursor_product_details = con_product_details.read_many({})
    # product_details = list(cursor_product_details)

    # already_scraped_names = []
    # for data in product_reviews:
    #     already_scraped_names.extend([key for key in data.keys() if key != '_id'])
    # for data in product_details:
    #     already_scraped_names.extend([key for key in data.keys() if key != '_id'])
    # already_scraped_names = list(set(already_scraped_names))

    # Read 3
    # 2. query postgres
    # sql_from_table = ' scraped_data.product_item_list '
    # where_clause = " product_position = 1 limit 70"
    # df = select_data(table=sql_from_table, where=where_clause)  # todo: refactor
    # assert df.shape[0] > 0
    # # df = df.sample(frac=1.0)
    # logger.debug(df.columns)
    # product_names = df['product_name'].values.tolist()
    #
    # # Read 4
    # # 2.5 get attempts
    # attempts_df = select_data(table=' scraped_data.product_query_attempts ')  # todo: refactor
    # attempt_product_names = attempts_df['attempt_product_name'].values.tolist()
    # # attempt_product_names = []
    #
    # # 3. exclude (1) from (2)
    #
    # # Read 4.5 aux
    # product_names_to_scrape = set([x for x in product_names if x not in already_scraped_names and x not in attempt_product_names])
    # assert len(product_names_to_scrape) > 0
    # ex = pool.ThreadPoolExecutor(max_workers=1,
    #                              thread_name_prefix='thread_',
    #                              initializer=None, initargs=())
    #
    # # Read 5
    # with ex as executor:
    #     future_to_url = {executor.submit(thread_work, user_query): user_query for user_query in product_names_to_scrape}
    # triplets = {}
    # for future in concurrent.futures.as_completed(future_to_url):
    #     url = future_to_url[future]
    #     try:
    #         data = future.result()
    #         triplets[url] = data
    #     except Exception as exc:
    #         print(f'{url} сгенерировано исключение: {exc}')

    # Write 1
    attempts_df = pd.DataFrame(product_names_to_scrape, columns=['attempt_product_name'])
    attempts_df['attempt_datetime'] = datetime.now()

    # Write 1
    # todo: refactor
    insert_data(attempts_df, schema_name='scraped_data', table_name='product_query_attempts')
    logger.info(type(triplets))
    logger.info(type(data))
    new_path = '/home/amstel/llm/out/QueryDetailsReviews.pkl'

    # Write 2
    if os.path.exists(new_path):
        new_path = new_path.replace('.pkl', '')+'1'+'.pkl'
    with open(new_path, 'wb') as f:
        pickle.dump(triplets, f)
        logger.debug(f'saved to: {new_path}')
    logger.info(len(triplets))

