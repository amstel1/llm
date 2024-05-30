import logging
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from onliner_mobile.spiders.quotes_onliner import OnlinerSpider
from onliner_mobile.settings import ROOT_PROJECT
from loguru import logger
import pickle
import os


class CustomCrawler:
    def __init__(self):
        self.output = None
        self.process = CrawlerProcess(get_project_settings(), )

    def yield_output(self, data):
        self.output = data

    def crawl(self, cls):
        self.process.crawl(cls, args={'callback': self.yield_output})
        self.process.start()


def crawl_static(cls):
    crawler = CustomCrawler()
    crawler.crawl(cls)
    return crawler.output


if __name__ == '__main__':

    output_filepath = os.path.join(ROOT_PROJECT, 'onliner_mobile/output')
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    logging.getLogger(__name__).setLevel(logging.INFO)
    process = CrawlerProcess(get_project_settings())

    # out = crawl_static(SantegoSpider)
    # with open(os.path.join(output_filepath, 'faucets_santego.pkl'), 'wb') as f:
    #     pickle.dump(out, f)

    # out = crawl_static(Vek21Spider)
    # with open(os.path.join(output_filepath, 'faucets_21vek.pkl'), 'wb') as f:
    #     pickle.dump(out, f)

    # out = crawl_static(SanitSpider)
    # with open(os.path.join(output_filepath, 'faucets_sanit.pkl'), 'wb') as f:
    #     pickle.dump(out, f)

    out = crawl_static(OnlinerSpider)
    with open(os.path.join(output_filepath, 'onliner.pkl'), 'wb') as f:
        pickle.dump(out, f)

    logger.info(len(out))

