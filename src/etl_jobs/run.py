import sys
sys.path.append('/home/amstel/llm/src')
from typing import Optional, Dict, Any, List
from base import Job, Read, Do, Write
import yaml
from loguru import logger

# step 1
from web_scraping.utils import EcomItemListRead
from postgres.utils import PostgresDataFrameWrite
from utils import ItemListDo

#step 2
from utils import ItemDetailsRead
from utils import ItemDetailsDo
from postgres.utils import PostgresDataFrameWrite
from utils import PickleDataRead, PickleDataWrite

# def load_component(class_name: str, params: Optional[Dict] = None) -> Any:
#     """Load a component dynamically given its class name and parameters."""
#     # classes = {
#     #     "TextFileRead": TextFileRead,
#     #     "PickleDictRead": PickleDictRead,
#     #     "UppercaseDo": UppercaseDo,
#     #     "FilterDo": FilterDo,
#     #     "TextFileWrite": TextFileWrite,
#     #     "PrintWrite": PrintWrite
#     # }
#     component_class = classes[class_name]
#     return component_class(**params) if params else component_class()


# def load_jobs_from_config(config_path: str) -> List[Job]:
#     """Load jobs from a YAML configuration file."""
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#
#     jobs = []
#     for job_config in config['jobs']:
#         reader = load_component(job_config['reader']['type'], job_config['reader'].get('params'))
#         processor = load_component(job_config['processor']['type'], job_config['processor'].get('params'))
#         writer = load_component(job_config['writer']['type'], job_config['writer'].get('params'))
#         jobs.append(Job(reader, processor, writer))
#
#     return jobs


if __name__ == '__main__':

    # jobs = load_jobs_from_config('job_config.yaml')
    # for job in jobs:
    #     job.run()

    # step 1. ItemList from sites to Postgres
    # logger.warning('Start - Job 1')
    # product_type_url = [f'https://shop.by/stiralnye_mashiny/?page_id={i}' for i in range(1, 30)]
    # product_type_name='Стиральная машина'
    # ItemlList_2_Postgres = Job(
    #     reader=EcomItemListRead(extractor_name='ShopByExtractor', product_type_url=product_type_url, product_type_name=product_type_name),
    #     processor=ItemListDo(),
    #     writer=PosgresDataFrameWrite(schema_name='scraped_data', table_name='product_item_list')
    # )
    # ItemlList_2_Postgres.run()
    # logger.warning('End - Job 1')

    # step 2. Read: ItemList from Postgres, Do: Scrapy ProductDetails, Write: to Postgres
    logger.warning('Start - Job 2')
    ItemDetails_2_Postgres = Job(
        reader=ItemDetailsRead(
            step1__table='scraped_data.product_item_list',
            step1__where=None,
            step1_utls_attribute='product_url'
        ),
        processor=ItemDetailsDo(),
        writer=PostgresDataFrameWrite(
            schema_name='scraped_data',
            table_name='item_details_washing_machine'),
    )
    ItemDetails_2_Postgres.run()
    logger.warning('End - Job 2')
