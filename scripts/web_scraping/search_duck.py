from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
import requests
import re

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from loguru import logger


def search_term_share(
        search_term: str,
        result_text: str
):
    len_search = len(search_term)
    len_result = len(result_text)
    start_pos = result_text.find(search_term)
    if start_pos != -1:
        return len_search / len_result
    else:
        return 0.0


def clear_str(s):
    return s.replace("Стоит ли покупать", "").replace("? Отзывы на Яндекс Маркете", "").strip().lower()

if __name__ == '__main__':

    user_query = 'Xiaomi 14'
    url = 'https://duckduckgo.com/'
    params={
            't':'h_',
            'q':'яндекс маркет отзывы ' + user_query,
            'ia':'web',
            # 'lr':157,
            # 'search_source':'yaby_desktop_common',
            # 'src':'suggest_B',
        }

    for i, (k,v ) in enumerate(params.items()):
        if i == 0:
            prefix = '?'
        else:
            prefix = '&'
        url += prefix + str(k) + '=' + str(v).replace(' ', '+')

    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto(url)
    content = page.content()

    with open('duck_res.html', 'w') as f:
        f.write(content)
    # with open("duck_res.html") as fp:
    #     soup = BeautifulSoup(fp, "html.parser")
    soup = BeautifulSoup(content, "html.parser")
    top_result = soup.find("article", id='r1-0')

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

    link = chosen_result.find("a", {'href': re.compile(r'reviews')}).get('href')
    print(link)

    page.screenshot(path="example2.png")
    browser.close()
    playwright.stop()
