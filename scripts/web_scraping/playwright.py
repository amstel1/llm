from playwright.sync_api import sync_playwright
playwright = sync_playwright().start()
# Use playwright.chromium, playwright.firefox or playwright.webkit
# Pass headless=False to launch() to see the browser UI
browser = playwright.chromium.launch(headless=True)
page = browser.new_page()
page.goto('https://market.yandex.by/product--smartfon-samsung-galaxy-a55-5g/84300549/reviews?sku=102875249860')
page.screenshot(path="example.png")
browser.close()
playwright.stop()