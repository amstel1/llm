from typing import Any, List, Dict
from base import Do, Read, Write, StepNum
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from postgres.utils import PostgresDataFrameRead
from web_scraping.utils import EcomProductRead
from loguru import logger
import attribute_mappings

class ItemListDo(Do):
    '''
    product_url	product_name	product_position	product_type_url	product_type_name
    '''
    def process(self, data: Dict[StepNum, Any]) -> Dict[StepNum, pd.DataFrame]:
        '''turns list of list of dict into pd.Dataframe'''
        # if data is list of lists
        assert isinstance(data, dict)
        data = data.get("step_0")
        if isinstance(data, list) and isinstance(data[0], list):
            data = [item for sublist in data for item in sublist]
        df = pd.DataFrame(data)
        if 'product_price' in df.columns: df['product_price'] = df['product_price'].astype(str).replace(' ', '').replace(',', '.').astype(float)
        df['scraped_datetime'] = datetime.now()
        return {"step_0": df}


class ItemDetailsDo(Do):
    def __init__(self, product_type_name='Холодильник'):
        self.product_type_name = product_type_name
        self.handler_mapping = {
            'Стиральная машина': attribute_mappings.washing_machine_mapping, # ++
            'Холодильник': attribute_mappings.fridge_mapping,  # ++
            'Телевизор': attribute_mappings.tv_mapping,  # ++
            'Мобильный телефон': attribute_mappings.mobile_mapping,  # ++
            'Чайник': attribute_mappings.kettle_mapping,  # ++

            'Пылесос': attribute_mappings.vacuumcleaner_mapping,  # +
            'Наушники': attribute_mappings.headphones_mapping,  # +
            'Умные часы': attribute_mappings.smartwatch_mapping,  # +
            'Посудомойка': attribute_mappings.dishwasher_mapping, # +
            'Варочная панель': attribute_mappings.hob_cooker_mapping, # +
            'Духовой шкаф': attribute_mappings.oven_cooker_mapping, # ++
            'Утюг': attribute_mappings.iron_mapping, # +
            'Кондиционер': attribute_mappings.conditioner_mapping,  # +
            'Водонагреватель': attribute_mappings.waterheater_mapping, # ++
            'Микроволновка': attribute_mappings.microwave_mapping, # +
        }
        assert self.product_type_name in self.handler_mapping

    @staticmethod
    def remove_from_bracket(s):
        if isinstance(s, str):
            position = s.find('(')
            if position != -1:
                return s[:position].strip(' ')
        return s

    def handler(self, df: pd.DataFrame):
        shop_mapping = self.handler_mapping[self.product_type_name]
        cols = [x for x in shop_mapping.keys() if x in df.columns]
        df = df[cols]
        df.rename(columns=shop_mapping, inplace=True)
        for col in df.columns:
            df[col] = df[col].map(ItemDetailsDo.remove_from_bracket)
            try:
                df[col] = df[col].astype(float)
            except Exception as e:
                df[col] = df[col].replace({'Есть': 'Да'})
        return df


    def process(self, data: Dict[StepNum, Any]) -> Dict[StepNum, Any]:
        # logger.critical(data)
        assert isinstance(data, dict)
        data = data.get("step_0")
        if isinstance(data, list) and not (isinstance(data[0], list) or isinstance(data[0], tuple)):
            df = pd.DataFrame(data)
            df = self.handler(df)
            return {"step_0": df}
        else:
            return {"step_0": None}


class PickleDataRead(Read):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def read(self) -> Any:
        # RETURNS ANY - CORRECT
        with open(self.filepath, 'rb') as f:
            data = pickle.load(f)
        return data


class PickleDataWrite(Write):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def write(self, data: Dict[StepNum, Any],) -> None:
        # WRITES AS IS - CORRECT
        assert isinstance(data, dict)
        for i, key in enumerate(data):
            # .replace('.pkl', f'step_{i}.pkl')
            with open(self.filepath, 'wb') as f:
                pickle.dump(data, f)


class ItemDetailsRead(Read):
    # helper class that composes two readers into one
    def __init__(self,
                 step1__table: str,
                 step1__where: str = None,
                 step1_urls_attribute: str = 'product_url'):
        self.step1_reader = PostgresDataFrameRead(table=step1__table, where=step1__where)
        self.step1_urls_attribute = step1_urls_attribute
        self.step2_reader = EcomProductRead()

    def read(self) -> Dict[StepNum, Any]:
        data_dict = self.step1_reader.read()
        df = data_dict.get("step_0")
        logger.debug(df.columns)
        urls = df[self.step1_urls_attribute].values.tolist()  # hardcoded
        product_details = self.step2_reader.read(urls=urls)
        return {"step_0": product_details}

class PopulateQueueDo(Do):
    # fields needed
    # search_query, product_yandex_name, processed(int), product_details_yandex_link, product_reviews_yandex_link
    def process(self, data: Dict[StepNum, Any]) -> Dict[StepNum, Any]:
        names_set = data.get('step_0')  # on previous step we read from postgres, source = details, e.g. fridge.item_details_fridge
        names = sorted(list(names_set))
        df = pd.DataFrame()
        df['search_query'] = names
        df['product_yandex_name'] = None
        df['searched'] = 0
        df['product_details_yandex_link'] = None
        df['product_reviews_yandex_link'] = None
        df['scraped'] = 0
        df['product_yandex_name'] = df['product_yandex_name'].astype(str)
        df['product_details_yandex_link'] = df['product_details_yandex_link'].astype(str)
        df['product_reviews_yandex_link'] = df['product_reviews_yandex_link'].astype(str)
        return {'step_0': df}

class SetSearchQueueProcessedDo(Do):
    def process(self, data: Dict[StepNum, Any]) -> Dict[StepNum, Any]:
        # data contains several steps.
        # i need step 0 - tasks for search

        # i need step_1 with triplets. - results of search
        # triplets' keys are user queries

        # the output must be df for fridge.search_queue with scraped = 1
        search_queue_df = data.get('step_0').get('step_0')  # dataframe

        triplets = data.get('step_1')  # dict
        triplets_search_queries = [key for key, value in triplets.items() if value[1]]   # not empty triplet

        product_details_yandex_link = search_queue_df[search_queue_df['search_query'].isin(triplets_search_queries)]['product_details_yandex_link']

        search_queue_df.loc[search_queue_df['product_details_yandex_link'].isin(product_details_yandex_link), 'scraped'] = 1
        return {'step_0': search_queue_df}

