import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import pickle
from loguru import logger

def build_link_tree(url, max_depth=2, allowed_domain='https://www.sber-bank.by'):
    """
    Builds a tree of links from a given URL up to a specified depth, respecting robots.txt.
    """
    tree = {}
    visited = {
        'https://www.sber-bank.by/person_business',
        'https://www.sber-bank.by/person',
        'https://www.sber-bank.by/loginsbol',
    }  # non-empty == blacklisted
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = RobotFileParser(robots_url)
    rp.read()

    def crawl(url, depth=0):
        if depth > max_depth or url in visited:  # Use full URLs for the check
            return

        if not rp.can_fetch("*", url):
            print(f"Skipping URL due to robots.txt: {url}")
            return

        visited.add(url)  # Add the full URL to the visited set
        try:
            response = requests.get(url)
        except Exception as e:
            logger.debug(f'{url} - err during request: {e}')
            return

        if response.status_code != 200:
            logger.debug(f'{url} - Non-200 status code: {response.status_code}')
            return

        soup = BeautifulSoup(response.content, "html.parser")
        links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
        tree[url] = links
        logger.warning(f'depth:{depth}--caller:{url}--n_this_level_urls:{len(links)}--tree_size:{len(tree)}')
        for link in links:
            if (link.startswith(allowed_domain)) and (not link.endswith('pdf')) and (not link.endswith('apk')) and (not link.endswith('xlsx')):
                logger.info(f'start scraping: {link}')
                crawl(link, depth+1)

    try:
        crawl(url)
    except Exception as e:
        logger.debug(f'Error during crawl: {e}')
    return tree



if __name__ == '__main__':
  # Example usage
  url = "https://www.sber-bank.by"
  link_tree = build_link_tree(url, max_depth=3)
  print(link_tree)

  with open('tree.pkl', 'wb') as f:
      pickle.dump(link_tree, f)

  # logger.warning(link_tree)