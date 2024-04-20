import asyncio
import concurrent.futures
import pickle
import re
import bs4
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from loguru import logger
from typing import Dict, List, Callable, Any
from bs4 import BeautifulSoup
import extruct
from db import select_from_db
import rapidfuzz
import concurrent.futures as pool
Url = str
import random
import numpy as np
# with open('desktop_user_agent.txt', 'r') as f:
#     uas = f.readlines()
# print(len(uas))
uas = [
    # 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0',
    # 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36',
    # 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36 OPR/38.0.2220.41',

    # 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.1234.56 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.0.0 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.0.0 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.0.0 Safari/537.36',

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
                        review_details['review_date_published'] = review.get('properties').get('datePublished')
                        review_details['review_description'] = review.get('properties').get('description')
                        _review_rating_properties = review.get('properties').get('reviewRating').get('properties')
                        review_details['review_ratng_value'] = _review_rating_properties.get('ratingValue')
                        review_details['review_best_rating'] = _review_rating_properties.get('bestRating')
                        total_reviews.append(review_details)
                return total_reviews

@logger.catch
def scrape_page_playwright(url_path: Url, parse_func: Callable) -> List[tuple[Url, Dict | List[Dict]]]:
    parsing_output = []
    with sync_playwright() as p:
        browser_select = random.sample(['firefox',], 1)[0]  #  'firefox',
        if browser_select == 'chromium':
            browser = p.chromium.launch(headless=False)
        elif browser_select == 'firefox':
            browser = p.firefox.launch(headless=False)
        ua = random.sample(uas, 1)[0]
        context = browser.new_context(user_agent=ua)
        logger.warning(ua)
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()
        page.wait_for_timeout(int(np.random.uniform(0, 1000, 1)[0]))
        page.goto(url_path)
        page.wait_for_timeout(int(np.random.uniform(2500, 5000, 1)[0]))
        # page.screenshot(path="now.png")
        content = page.content()
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

    id_2_href = {}
    id_2_text_primary = {}  # results containing 'reviews' - reliable
    id_2_text_secondary = {}  # results not containing 'reviews' - unreliable

    id_2_score = {}  # simple relevancy score
    # logger.debug(f'len of articles: {len(articles)}')
    for article_id, article in enumerate(articles):
        try:
            article_href = article.find('span').find('a').get('href')
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
                id_2_text_primary[article_id] = article_text
                # https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html
                id_2_text_primary[article_id] = rapidfuzz.fuzz.token_set_ratio(
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
                id_2_text_secondary[article_id] = rapidfuzz.fuzz.token_set_ratio(
                    user_query,
                    article_href,  # seacrh against href because in google end text is truncated with dots
                    processor=rapidfuzz.utils.default_process
                )
        except:
            pass
    if len(id_2_text_primary) > 0:
        # logger.critical(sorted(id_2_text.items(), key=lambda x: x[1], reverse=True))
        best_id, best_score = sorted(id_2_text_primary.items(), key=lambda x: x[1], reverse=True)[0]
        best_text = id_2_text_primary[best_id]
        best_href = id_2_href[best_id]
    elif len(id_2_text_secondary) > 0:
        best_id, best_score = sorted(id_2_text_secondary.items(), key=lambda x: x[1], reverse=True)[0]
        best_text = id_2_text_secondary[best_id]
        best_href = id_2_href[best_id]
    else:
        # no results found
        best_id = None
        best_score = None
        best_text = None
        best_href = None
    # logger.info(best_id, best_score, best_text, best_href)
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
        browser_select = random.sample(['firefox',], 1)[0]  # 'firefox'
        if browser_select == 'chromium':
            browser = p.chromium.launch(headless=False)
        else:
            browser = p.firefox.launch(headless=False)
        ua = random.sample(uas, 1)[0]
        context = browser.new_context(user_agent=ua)
        logger.warning(ua)
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()
        page.wait_for_timeout(int(np.random.uniform(0, 1000, 1)[0]))
        page.goto(url)
        page.wait_for_timeout(int(np.random.uniform(2500, 5000, 1)[0]))
        content = page.content()
        try:
            with open(f"./htmls/{user_query.replace('/', '-')}.html", 'w') as f:
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
            # logger.info('branch reviews')
            # logger.info(user_query, )
            reviews_url  = some_url
            product_url = some_url[:some_url.find('/reviews')] + '/reviews'
            # logger.info(f'some_url: {some_url}')
            # logger.info(f'reviews_url: {reviews_url}')
            # logger.info(f'product_url: {product_url}')
        elif '?sku=' in some_url:
            # logger.warning('branch ?sku=')
            # logger.warning(user_query, )
            product_url = some_url[:some_url.find('?sku=')]
            reviews_url = product_url + '/reviews'
            # logger.warning(f'some_url: {some_url}')
            # logger.warning(f'reviews_url: {reviews_url}')
            # logger.warning(f'product_url: {product_url}')
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
    sql_from_table = ' scraped_data.product_item_list '
    where_clause = " crawl_id >= 1 limit 8 "
    df = select_from_db.select_data(table=sql_from_table, where=where_clause)
    df = df.sample(frac=1.0)
    product_names = df['product_name'].values.tolist()

    ex = pool.ThreadPoolExecutor(max_workers=1,
                                 thread_name_prefix='thread_',
                                 initializer=None, initargs=())

    with ex as executor:
        future_to_url = {executor.submit(thread_work, user_query): user_query for user_query in product_names}
    pairs = {}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
            pairs[url] = data
        except Exception as exc:
            print(f'{url} сгенерировано исключение: {exc}')

    logger.info(type(pairs))
    logger.info(type(data))
    logger.info(len(pairs))
    with open('../out/future_pairs2.pkl', 'wb') as f:
        pickle.dump(pairs, f)
