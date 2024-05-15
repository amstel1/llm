from typing import Any, List, Dict
from base import Do, Read, Write
import pandas as pd
import pickle
from datetime import datetime
from postgres.utils import PostgresDataFrameRead
from web_scraping.utils import EcomProductRead
from loguru import logger


class ItemListDo(Do):
    '''
    product_url	product_name	product_position	product_type_url	product_type_name
    '''
    def process(self, data: Any) -> Any:
        '''turns list of list of dict into pd.Dataframe'''
        # if data is list of lists
        if isinstance(data, list) and isinstance(data[0], list):
            flattened = [item for sublist in data for item in sublist]
            df = pd.DataFrame(flattened)
            df['scraped_datetime'] = datetime.now()
            return df


class ItemDetailsDo(Do):
    @staticmethod
    def remove_from_bracket(s):
        if isinstance(s, str):
            position = s.find('(')
            if position != -1:
                return s[:position].strip(' ')
        return s

    def washing_mashine_preprocess(self, df: pd.DataFrame):
        # logger.debug(df.head())
        shop_washing_machine_inverse_mapping = {
            'name': 'name',
            'product_url': 'product_url',
            'offer_count': 'offer_count',
            'min_price': 'min_price',
            "brand": "Производитель",
            "load_type": "Тип загрузки",
            "max_load": "Максимальная загрузка белья, кг",
            "depth": "Глубина, см",
            "width": "Ширина, см",
            "height": "Высота, см",
            "weight": "Вес, кг",
            "programs_number": "Количество программ",
            "has_display": "Дисплей",
            "installation_type": "Установка",
            "color": "Цвет",
            "washing_class": "Класс стирки",
            "additional_rinsing": "Дополнительное полоскание",
            "max_rpm": "Максимальное количество оборотов отжима, об/мин",
            "spinning_class": "Класс отжима",
            "spinning_speed_selection": "Выбор скорости отжима",
            "drying": "Сушка",
            "energy_class": "Класс энергопотребления",
            "addition_features": "Дополнительные функции",
            "can_add_clothes": "Возможность дозагрузки белья",
            "cancel_spinning": "Отмена отжима",
            "light_ironing": "Программа «легкая глажка»",
            "direct_drive": "Прямой привод (direct drive)",
            "inverter_motor": "Инверторный двигатель",
            "safety_features": "Безопасность",
            "water_consumption": "Расход воды за стирку, л",
        }
        shop_washing_machine_mapping = {v: k for k, v in shop_washing_machine_inverse_mapping.items()}
        cols = shop_washing_machine_mapping.keys()
        # logger.debug(cols)

        df = df[cols]
        df.rename(columns=shop_washing_machine_mapping, inplace=True)

        for col in ['min_price', 'max_load', 'depth', 'width', 'height', 'weight', 'max_rpm', 'water_consumption',
                    'programs_number']:
            df[col] = df[col].map(ItemDetailsDo.remove_from_bracket)
            df[col] = df[col].astype(float)
        for col in ['has_display', 'additional_rinsing',
                    'spinning_speed_selection', 'drying',
                    'can_add_clothes', 'cancel_spinning',
                    'light_ironing', 'direct_drive', 'inverter_motor']:
            df[col] = df[col].map(ItemDetailsDo.remove_from_bracket)
            df[col] = df[col].replace({'Есть': 'Да'})
        return df

    def process(self, data: Any) -> Any:
        # logger.critical(data)
        if isinstance(data, list) and not (isinstance(data[0], list) or isinstance(data[0], tuple)):
            df = pd.DataFrame(data)
            # logger.warning(df)
            df = self.washing_mashine_preprocess(df)
            return df
        else:
            return None


class PickleDataRead(Read):
    def __init__(self, filepath: str):
        self.filepath = filepath
    def read(self,) -> Any:
        with open(self.filepath, 'rb') as f:
            data = pickle.load(f)
        return data


class PickleDataWrite(Write):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def write(self, data: Any,) -> None:
        with open(self.filepath, 'wb') as f:
            pickle.dump(data, f)


class ItemDetailsRead(Read):
    # helper class that composes two readers into one
    def __init__(self,
                 step1__table: str,
                 step1__where: str = None,
                 step1_utls_attribute: str = 'product_url'):
        self.step1_reader = PostgresDataFrameRead(table=step1__table, where=step1__where)
        self.step1_urls_attribute = step1_utls_attribute
        self.step2_reader = EcomProductRead()

    def read(self) -> List[Dict]:
        df = self.step1_reader.read()
        urls = df[self.step1_urls_attribute].values.tolist()  # hardcoded
        product_details = self.step2_reader.read(urls=urls)
        return product_details

