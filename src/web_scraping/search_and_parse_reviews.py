import asyncio
import pickle
import re
import bs4
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from loguru import logger
from typing import Dict, List, Callable
from bs4 import BeautifulSoup
from typing import Dict
import extruct
from db import select_from_db
import rapidfuzz
ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'


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
def parse_reviews(mdata) -> str:
    '''returns reviews delimited by /n from microdata'''
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
async def scrape_page_playwright(url_path: str, parse_func: Callable) -> List[Dict]:
    parsing_output = []
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url_path)
        # page.screenshot(path="now.png")
        content = await page.content()
        # with open('product.html', 'w') as f:
        #     f.write(content)
        soup = BeautifulSoup(content, "html.parser")
        mdata = extruct.extract(soup.prettify(), syntaxes=['microdata']).get('microdata')
        parsing_output = parse_func(mdata)
        # assert parsing_output  # if error, parsing failed, check microdata of the page
        await browser.close()
    return parsing_output


def clear_str(s):
    return s.replace("Стоит ли покупать", "").replace("? Отзывы на Яндекс Маркете", "").strip().lower()


def google_find_link(soup, user_query) -> str:
    # articles = soup.find('div', id='rso').find_all(class_='MjjYud')
    articles = soup.find_all(class_='MjjYud')

    id_2_href = {}
    id_2_text = {}
    id_2_score = {}  # simple relevancy score
    logger.debug(f'len of articles: {len(articles)}')
    for article_id, article in enumerate(articles):
        try:
            article_href = article.find('span').find('a').get('href')
            article_text = article.find('h3').text
            # step 1 - check if link is good
            logger.debug(f'candidate reviews link: {article_href}')
            if ('market.yandex.' in article_href) and \
                    ('/reviews' in article_href) and \
                    ('text?search' not in article_href):
                logger.debug(f'passed the check for reviews: {article_href} -- {article_text}', )
                id_2_href[article_id] = article_href
                id_2_text[article_id] = article_text
                # https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html
                id_2_text[article_id] = rapidfuzz.fuzz.token_set_ratio(
                    user_query,
                    article_href,  # seacrh against href because in google end text is truncated with dots
                    processor=rapidfuzz.utils.default_process
                )
        except:
            pass
    if len(id_2_text) > 0:
        logger.critical(sorted(id_2_text.items(), key=lambda x: x[1], reverse=True))
        best_id, best_score = sorted(id_2_text.items(), key=lambda x: x[1], reverse=True)[0]
        best_text = id_2_text[best_id]
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
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        content = page.content()
        with open(f'{user_query}.html', 'w') as f:
            f.write(content)
        # with open("duck_res.html") as fp:
        #     soup = BeautifulSoup(fp, "html.parser")
        soup = BeautifulSoup(content, "html.parser")
        logger.debug('in google_find_link')
        best_link = google_find_link(soup=soup, user_query=user_query)
        browser.close()
    return best_link
    # except Exception as e:
    #     logger.critical(f'error: {e}')
    #     return ''


if __name__ == '__main__':
    sql_from_table = 'scraped_data.product_item_list'
    where_clause = 'crawl_id <= 2 order by product_name asc limit 5'
    df = select_from_db.select_data(table=sql_from_table, where=where_clause)
    product_names = df['product_name'].values.tolist()

    # user_query = 'Xiaomi 14'
    pairs = []
    for i, user_query in enumerate(product_names):
        logger.warning(user_query)
        reviews_url = search_google(user_query=user_query)  # may return None if DGG search fails
        if reviews_url:
            product_url = reviews_url.strip('/reviews')
            review_details_list = asyncio.run(scrape_page_playwright(url_path=reviews_url, parse_func=parse_reviews))
            product_details_list = asyncio.run(scrape_page_playwright(url_path=product_url, parse_func=parse_product))
            pairs.append((product_details_list, review_details_list))
            logger.info(product_details_list)
    with open('../out/pairs.pkl', 'wb') as f:
        pickle.dump(pairs, f)
    # logger.warning(pairs)
    logger.warning(len(pairs))