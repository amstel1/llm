from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

import requests
from bs4 import BeautifulSoup

url = 'https://yandex.by/search/'
params={
        'text':'Xiaomi 14',
        'lr':157,
        'search_source':'yaby_desktop_common',
        'src':'suggest_B',
    }

for i, (k,v ) in enumerate(params.items()):
    if i == 0:
        prefix = '?'
    else:
        prefix = '&'
    url += prefix + str(k) + '=' + str(v).replace(' ', '+')

from playwright.sync_api import sync_playwright
playwright = sync_playwright().start()
# Use playwright.chromium, playwright.firefox or playwright.webkit
# Pass headless=False to launch() to see the browser UI
browser = playwright.chromium.launch(headless=True)
page = browser.new_page()
page.goto(url)
page.screenshot(path="example2.png")
browser.close()
playwright.stop()
