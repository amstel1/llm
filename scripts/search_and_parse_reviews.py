import pickle

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
import requests
import re

from bs4 import BeautifulSoup
import bs4
from playwright.sync_api import sync_playwright
from loguru import logger
from typing import Dict, List, Callable
from bs4 import BeautifulSoup
from typing import Dict
from extruct.w3cmicrodata import MicrodataExtractor
from extruct.microformat import MicroformatExtractor
import pprint
import extruct
ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
from db import select_from_db

def parse_product(mdata) -> Dict[str, str]:
    '''return product details from microdata'''
    for el in mdata:
        if el.get('type') == 'https://schema.org/Product':
            if 'properties' in el:
                product_category = el.get('properties').get('category')
                product_brand = el.get('properties').get('brand')
                product_name = el.get('properties').get('name')
                product_url = el.get('properties').get('url')
                product_image_url = el.get('properties').get('image')
                _aggredagate_rating = el.get('properties').get('aggregateRating').get('properties')
                product_rating_value = _aggredagate_rating.get('ratingValue')
                product_best_rating = _aggredagate_rating.get('bestRating')
                product_worst_rating = _aggredagate_rating.get('worstRating')
                product_rating_count = _aggredagate_rating.get('ratingCount')
                product_review_count = _aggredagate_rating.get('reviewCount')
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


def parse_reviews(mdata) -> str:
    '''returns reviews delimited by /n from microdata'''
    for el in mdata:
        if el.get('type') == 'https://schema.org/Product':
            if 'properties' in el:
                total_reviews = []
                _reviews = el.get('properties').get('review')
                for review in _reviews:
                    review_details = {}
                    review_details['review_date_published'] = review.get('properties').get('datePublished')
                    review_details['review_description'] = review.get('properties').get('description')
                    _review_rating_properties = review.get('properties').get('reviewRating').get('properties')
                    review_details['review_ratng_value'] = _review_rating_properties.get('ratingValue')
                    review_details['review_best_rating'] = _review_rating_properties.get('bestRating')
                    total_reviews.append(review_details)
    return total_reviews


def scrape_page_playwright(url_path: str, parse_func: Callable) -> List[Dict]:
    try:
        playwright = sync_playwright().start()
        browser = playwright.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url_path)
        # page.screenshot(path="now.png")
        content = page.content()
        # with open('product.html', 'w') as f:
        #     f.write(content)
        soup = BeautifulSoup(content, "html.parser")
        mdata = extruct.extract(soup.prettify(), syntaxes=['microdata']).get('microdata')
        parsing_output = parse_func(mdata)
        browser.close()
        playwright.stop()
        return parsing_output
    except:
        browser.close()
        playwright.stop()
        return []


def search_term_share(search_term: str,   result_text: str):
    len_search = len(search_term)
    len_result = len(result_text)
    start_pos = result_text.find(search_term)
    if start_pos != -1:
        return len_search / len_result
    else:
        return 0.0


def clear_str(s):
    return s.replace("Стоит ли покупать", "").replace("? Отзывы на Яндекс Маркете", "").strip().lower()


def search_dgg(user_query: str) -> bs4.element:
    try:
        url = 'https://duckduckgo.com/'
        params = {
            't': 'h_',
            'q': 'яндекс маркет отзывы ' + user_query,
            'ia': 'web',
        }
        for i, (k, v) in enumerate(params.items()):
            if i == 0:
                prefix = '?'
            else:
                prefix = '&'
            url += prefix + str(k) + '=' + str(v).replace(' ', '+')
        playwright = sync_playwright().start()
        browser = playwright.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        content = page.content()
        # with open('duck_res.html', 'w') as f:
        #     f.write(content)
        # with open("duck_res.html") as fp:
        #     soup = BeautifulSoup(fp, "html.parser")
        soup = BeautifulSoup(content, "html.parser")
        # top_result = soup.find("article", id='r1-0')
        first_result = soup.find("article", id='r1-0')
        second_result = soup.find("article", id='r1-1')
        first_result_text = first_result.find("h2").find("span").text
        second_result_text = second_result.find("h2").find("span").text
        search_term = user_query.lower()
        first_result_text = clear_str(first_result_text).lower()
        second_result_text = clear_str(second_result_text).lower()
        first_result_text.find(search_term)
        first_share = search_term_share(search_term, first_result_text)
        second_share = search_term_share(search_term, second_result_text)
        if first_share >= second_share:
            chosen_result = first_result
        else:
            chosen_result = second_result
        browser.close()
        playwright.stop()
        return chosen_result
    except:
        browser.close()
        playwright.stop()
        return []


if __name__ == '__main__':



    sql_from_table = 'scraped_data.product_item_list'
    where_clause = 'crawl_id <= 2 limit 5'
    df = select_from_db.select_data(table=sql_from_table, where=where_clause)
    product_names = df['product_name'].values.tolist()

    # user_query = 'Xiaomi 14'
    pairs = []
    for user_query in product_names:
        logger.warning(user_query)
        chosen_result = search_dgg(user_query=user_query)
        assert chosen_result
        reviews_url = chosen_result.find("a", {'href': re.compile(r'reviews')}).get('href')
        product_url = reviews_url.strip('/reviews')
        review_details_list = scrape_page_playwright(url_path=reviews_url, parse_func=parse_reviews)
        product_details_list = scrape_page_playwright(url_path=product_url, parse_func=parse_product)
        pairs.append((product_details_list, review_details_list))
        logger.info(product_details_list)
        with open('pairs.pkl', 'wb') as f:
            pickle.dump(pairs, f)