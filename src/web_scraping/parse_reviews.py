
reviews_url = 'https://market.yandex.ru/product--xiaomi-14/1943226316/reviews'
# reviews_url = 'https://market.yandex.by/product/1943226316'

import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from loguru import logger

from playwright.sync_api import sync_playwright
playwright = sync_playwright().start()
browser = playwright.firefox.launch(headless=True)
page = browser.new_page()
page.goto(reviews_url)
content = page.content()
with open('product_res.html', 'w') as f:
    f.write(content)
page.screenshot(path="example3.png")
browser.close()
playwright.stop()