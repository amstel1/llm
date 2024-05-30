import pickle
from loguru import logger
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from typing import Any, Iterator, List, Sequence, cast
from langchain_core.documents import Document
import multiprocessing
import asyncio
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import scrapy
import pandas as pd

def get_navigable_strings(element: Any) -> Iterator[str]:
    """Get all navigable strings from a BeautifulSoup element.

    Args:
        element: A BeautifulSoup element.

    Returns:
        A generator of strings.
    """

    from bs4 import NavigableString, Tag

    for child in cast(Tag, element).children:
        if isinstance(child, Tag):
            yield from get_navigable_strings(child)
        elif isinstance(child, NavigableString):
            yield child.strip()


class CustomSpider(scrapy.Spider):
    name = "custom_scraper"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug(kwargs)
        self.output_callback = kwargs.get('args').get('callback')
        self.output_data = []
        self.urls = urls

    def start_requests(self):
        for url in self.urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                meta={"playwright": True}
            )

    def parse(self, response):
        if response.status in (200,301):
            document = Document(page_content=response.text, metadata={"source": response.url})
            self.output_data.append(document)

    def close(self, spider, reason):
        self.output_callback(self.output_data)

class ScrapySberbankScraper:
    def __init__(self, urls):
        self.output = None
        self.process = CrawlerProcess(get_project_settings(), )
        self.urls = urls

    def yield_output(self, data):
        self.output = data

    def crawl(self, cls):
        self.process.crawl(cls, args={'callback': self.yield_output})
        self.process.start()

def crawl_static(cls, urls):
    crawler = ScrapySberbankScraper(urls)
    crawler.crawl(cls)
    return crawler.output


class CustomBeautifulSoupTransformer(BeautifulSoupTransformer):
    """
    """  # noqa: E501
    def transform_documents(
        self,
        documents: Sequence[Document],
        unwanted_tags: List[str] = ["script", "style", "button", "header"],
        tags_to_extract: List[str] = ["p", "li", "div", "a"],
        unwanted_class_names_like: List[str] = [
            "Footer", "Prefooter", "TopMenu", "CookiesMessageBlock", "IconDescription", "BPSsiteSberPrimeForm",
            "VotesBlock", "DownloadFileWithTitle",
        ],
        remove_lines: bool = True,
        separator: str = " | ",
        **kwargs: Any,
    ) -> Sequence[Document]:
        """
        Transform a list of Document objects by cleaning their HTML content.

        Args:
            documents: A sequence of Document objects containing HTML content.
            unwanted_tags: A list of tags to be removed from the HTML.
            tags_to_extract: A list of tags whose content will be extracted.
            remove_lines: If set to True, unnecessary lines will be
            removed from the HTML content.

        Returns:
            A sequence of Document objects with transformed content.
        """
        for doc in documents:
            cleaned_content = doc.page_content
            cleaned_content = self.remove_by_class_names(cleaned_content, unwanted_class_names_like)
            cleaned_content = self.remove_unwanted_tags(cleaned_content, unwanted_tags)
            cleaned_content = self.extract_tags(cleaned_content, tags_to_extract, separator=separator)
            if remove_lines:
                cleaned_content = self.remove_unnecessary_lines(cleaned_content)
            doc.page_content = cleaned_content

        return documents

    @staticmethod
    def remove_by_class_names(html_content: str, unwanted_class_names_like: List[str]):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in unwanted_class_names_like:
            for element in soup.find_all(class_=lambda x: x and tag in x):
                element.decompose()
        return_str = str(soup)
        return_str = return_str.replace('\xa0', ' ')
        return return_str

    @staticmethod
    def extract_tags(html_content: str, tags: List[str], separator: str =" | ") -> str:
        """
        Extract specific tags from a given HTML content.

        Args:
            html_content: The original HTML content string.
            tags: A list of tags to be extracted from the HTML.

        Returns:
            A string combining the content of the extracted tags.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        text_parts: List[str] = []
        for element in soup.find_all():
            if element.name in tags:
                # Extract all navigable strings recursively from this element.
                text_parts += get_navigable_strings(element)
                # To avoid duplicate text, remove all descendants from the soup.
                element.decompose()

        return separator.join([x.strip() for x in text_parts if len(x.strip()) >= 1])  # из-за А1


class MultiAsyncChromiumLoader(AsyncChromiumLoader):
    def __init__(self, n_cpu: int=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_cpu = n_cpu

    def get_document_from_url(self, url: str):
        html_content = asyncio.run(self.ascrape_playwright(url))
        metadata = {"source": url}
        return Document(page_content=html_content, metadata=metadata)

    def lazy_load(self) -> List[Document]:
        with multiprocessing.Pool(self.n_cpu) as pool:
            documents = pool.map(self.get_document_from_url, self.urls)
        logger.debug(len(documents))
        return documents


# def parse_sberbank_urls(urls: List[str], n_cpu: int = 4) -> List[str]:
#     '''
#     Turns html into text
#
#     :param urls: urls to be scraped, must start with https://www.sber-bank.by/
#     :return: retrieved text
#     '''
#
#     loader = MultiAsyncChromiumLoader(n_cpu=n_cpu, urls=urls)
#     docs = loader.load()
#     logger.debug(f'{type(docs)}, {len(docs)}')
#     transformer = CustomBeautifulSoupTransformer()
#     docs_transformed = transformer.transform_documents(
#         docs,
#         tags_to_extract=["p", "h1", 'h2', 'h3', 'h4', 'h5', ],
#         unwanted_tags=[
#             'a',
#         #     'button',
#         #     'footer',
#         #     'header',
#         #     'ul',
#         #     'span',
#         #     "script",
#         #     "style",
#         #     'div'
#         ],
#         remove_lines=False,
#     )
#     return docs_transformed

def parse_sberbank_docs(docs: List[Document], ) -> List[str]:
    transformer = CustomBeautifulSoupTransformer()
    docs_transformed = transformer.transform_documents(
        docs,
        tags_to_extract=["a", "p", "h1", 'h2', 'h3', 'h4', 'h5', "li", "div", "span"],
        unwanted_tags=["script", "style", "button", "input"],
        unwanted_class_names_like=[
            "Footer", "Prefooter", "TopMenu", "CookiesMessageBlock", "IconDescription", "BPSsiteSberPrimeForm",
            "VotesBlock", "DownloadFileWithTitle", "DepositCalculator", "SelectorDropDown"
        ],
        remove_lines=False,
        separator=" | ",
    )
    return docs_transformed

def preprocess(fpath) -> List[str]:
    """preprocesses tree (create_tree_link.py)"""
    with open(fpath, 'rb') as f:
        link_tree = pickle.load(f)

    links = []
    links.extend(list(link_tree.keys()))
    links.extend(list(v) for v in link_tree.values())

    links = [x for x in links if x]

    total = []
    for x in links:
        total.extend(x)

    total = set(total)

    final = []
    for link in total:
        if link.startswith('https://www.sber-bank.by') and \
                (not link.endswith('.pdf')) and \
                (not link.endswith('.docx')) and \
                (not link.endswith('.doc')) and \
                (not link.endswith('.xlsx')) and \
                (not link.endswith('.apk')) and \
                (not link.endswith('.zip')) and \
                (not link.endswith('.rar')) and \
                (not link.endswith('.svg')) and \
                (not link.endswith('.xls')):
            final.append(link)

    final = [x for x in final if 'news' not in x]
    final = [x for x in final if 'business' not in x]
    final = [x for x in final if 'biznes' not in x]
    final = [x for x in final if '+375' not in x]
    final = [x for x in final if '#_ftn' not in x]
    final = [x for x in final if 'files/up' not in x]
    final = [x for x in final if 'loginsbol' not in x]
    final = [x for x in final if '?selected-insurance-id=' not in x]
    final = [x for x in final if 'token-scope' not in x]
    final = [x for x in final if '?selected-credit' not in x]
    final = [x for x in final if '?request=' not in x]
    final = [x for x in final if 'Dostavych' not in x]
    final = [x for x in final if 'zarplatnyj-proekt' not in x]

    return final

if __name__ == '__main__':
    logger.info('started')

    df = pd.read_excel('all.xlsx')
    urls = df['aux'].values.tolist()

    # urls = preprocess(fpath='/home/amstel/llm/src/web_scraping/bank_scraper/tree.pkl')

    urls = sorted(urls)#[:4]

    # urls = [
    #     'https://www.sber-bank.by/page/oferta'
    # ]
    # urls = ['https://www.sber-bank.by/deposit/adulthood-local/BYN/attributes']
    project_settings = get_project_settings()
    logger.info(project_settings)
    process = CrawlerProcess(project_settings)

    documents = crawl_static(CustomSpider, urls)
    transformed_docs = parse_sberbank_docs(docs=documents)
    # logger.info(transformed_docs)

    with open('docs_all_12052024.pkl', 'wb') as f:
        pickle.dump(transformed_docs, f)

    # for doc in transformed_docs:
    #     meta = doc.metadata
    #     source = meta.get('source')
    #     logger.critical(doc)
    #     print()


