from typing import Any, List, Dict
from base import Do, Read, Write, StepNum
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from postgres.utils import PostgresDataFrameRead
from web_scraping.utils import EcomProductRead
from loguru import logger


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
        df['scraped_datetime'] = datetime.now()
        return {"step_0": df}


class ItemDetailsDo(Do):
    def __init__(self, product_type_name='Холодильник'):
        self.product_type_name = product_type_name
        self.handler_mappring = {
            'Стиральная машина': self.washing_mashine_handler,
            'Холодильник': self.fridge_handler,
            'Телевизор': self.tv_handler,
            'Мобильный телефон': self.mobile_handler,
        }
        assert self.product_type_name in self.handler_mappring

    @staticmethod
    def remove_from_bracket(s):
        if isinstance(s, str):
            position = s.find('(')
            if position != -1:
                return s[:position].strip(' ')
        return s

    def mobile_handler(self, df: pd.DataFrame):
        shop_mapping = {
            'name': 'name',
            'product_url': 'product_url',
            'offer_count': 'offer_count',
            'min_price': 'min_price',
            "Производитель": "manufacturer",
            "Дата выхода": "release_date",
            "Тип устройства": "device_type",
            "Вид устройства": "device_category",
            "Платформа": "platform",
            "Версия ОС": "os_version",
            "Серия": "series",
            "Стандарт связи": "network_standard",
            "Количество SIM-карт": "num_sim_cards",
            "Тип SIM-карты": "sim_type",
            "Диагональ экрана, \"": "screen_diagonal_inch",
            "Разрешение экрана, точек": "screen_resolution_px",
            "Технология экрана": "screen_technology",
            "Частота обновления экрана": "refresh_rate_hz",
            "Оперативная память": "ram_gb",
            "Встроенная память": "internal_storage_gb",
            "Поддержка карт памяти": "memory_card_support",
            "Камера": "camera",
            "Количество основных камер": "num_main_cameras",
            "Разрешение основной камеры, Мп": "main_camera_resolution_mp",
            "Максимальное разрешение видео": "max_video_resolution",
            "Количество кадров в секунду": "fps",
            "Беспроводная зарядка": "wireless_charging",
            "Быстрая зарядка": "fast_charging",
            "Обратная беспроводная зарядка": "reverse_wireless_charging",
            "Модули камеры": "camera_modules",
            "Конструкция корпуса": "body_construction",
            "Материал корпуса": "body_material",
            "Цвет корпуса": "body_color",
            "Ударопрочный корпус": "shockproof_body",
            "Пыле- и влагозащита": "dust_water_resistance",
            "Степень защиты (IP)": "ip_rating",
            "Высота, см": "height_cm",
            "Ширина, см": "width_cm",
            "Толщина, см": "thickness_cm",
            "Вес, г": "weight_g",
            "Число пикселей на дюйм, ppi": "ppi",
            "Соотношение сторон": "aspect_ratio",
            "Сенсорный экран": "touch_screen",
            "Защита от царапин": "scratch_protection",
            "Постоянная работа экрана": "always_on_display",
            "Производитель процессора": "processor_manufacturer",
            "Процессор": "processor",
            "Количество ядер процессора": "num_processor_cores",
            "Техпроцесс": "process_technology",
            "Диафрагма": "aperture",
            "Вспышка": "flash",
            "Автофокус": "autofocus",
            "Оптическая стабилизация": "optical_stabilization",
            "Оптический зум": "optical_zoom",
            "Макс. количество кадров в секунду": "max_fps",
            "Количество фронтальных камер": "num_front_cameras",
            "Разрешение фронтальной камеры, Мп": "front_camera_resolution_mp",
            "Диафрагма фронтальной камеры": "front_camera_aperture",
            "Вспышка фронтальной камеры": "front_camera_flash",
            "Автофокус фронтальной камеры": "front_camera_autofocus",
            "Разрешение видео фронтальной камеры": "front_camera_video_resolution",
            "Стереодинамики": "stereo_speakers",
            "Безопасность": "security_features",
            "Навигация": "navigation",
            "Беспроводные интерфейсы": "wireless_interfaces",
            "Разъемы": "ports",
            "Емкость аккумулятора, мАч": "battery_capacity_mah",
            "Аккумулятор": "battery",
            "Время воспроизведения видео, ч": "video_playback_time_h",
            "Время воспроизведения аудио, ч": "audio_playback_time_h",
            "Комплект поставки": "package_contents"
        }

        shop_tv_inverse_mapping = {v: k for k, v in shop_mapping.items()}
        cols = shop_mapping.keys()

        df = df[cols]
        df.rename(columns=shop_mapping, inplace=True)

        for col in df.columns:
            df[col] = df[col].map(ItemDetailsDo.remove_from_bracket)
            try:
                df[col] = df[col].astype(float)
            except Exception as e:
                df[col] = df[col].replace({'Есть': 'Да'})

        # for col in ['offer_count', 'min_price',
        #             ]:
        #     df[col] = df[col].map(ItemDetailsDo.remove_from_bracket)
        #     df[col] = df[col].astype(float)
        # for col in []:
        #     df[col] = df[col].map(ItemDetailsDo.remove_from_bracket)
        #     df[col] = df[col].replace({'Есть': 'Да'})
        return df

    def tv_handler(self, df: pd.DataFrame):
        shop_tv_mapping = {
            'name': 'name',
            'product_url': 'product_url',
            'offer_count': 'offer_count',
            'min_price': 'min_price',
            "Производитель": "manufacturer",
            "Дата выхода": "release_date",
            "Тип": "type",
            "Диагональ экрана, \"": "screen_diagonal_inch",
            "Разрешение экрана": "screen_resolution",
            "Частота матрицы, Гц": "refresh_rate_hz",
            "Изогнутый экран": "curved_screen",
            "Поддержка 3D": "support_3d",
            "Smart TV": "smart_tv",
            "Платформа Smart TV": "smart_tv_platform",
            "Версия системы": "system_version",
            "Безрамочный дизайн": "frameless_design",
            "Цвет корпуса": "body_color",
            "Цвет рамки": "frame_color",
            "Цвет подставки": "stand_color",
            "Подставка": "stand",
            "Крепление на стену": "wall_mount",
            "Глубина цвета": "color_depth",
            "Расширенный динамический диапазон (HDR)": "hdr",
            "Форматы HDR": "hdr_formats",
            "Игровые функции": "gaming_features",
            "Процессор": "processor",
            "Фоновая подсветка": "backlight",
            "Голосовое управление": "voice_control",
            "ТВ-тюнер": "tv_tuner",
            "Сабвуфер": "subwoofer",
            "Звуковая панель (саундбар)": "soundbar",
            "Мощность звуковой системы, Вт": "sound_system_power_w",
            "Количество динамиков": "num_speakers",
            "Поддержка аудиокодеков объемного звука": "surround_sound_codec_support",
            "Поддержка HDMI eARC": "hdmi_earc_support",
            "Беспроводные интерфейсы": "wireless_interfaces",
            "Версия Bluetooth": "bluetooth_version",
            "Стандарт Wi-Fi": "wifi_standard",
            "Разъемы": "ports",
            "Кол-во разъемов HDMI": "num_hdmi_ports",
            "Версия HDMI": "hdmi_version",
            "Кол-во разъемов USB": "num_usb_ports",
            "Smart-пульт": "smart_remote",
            "Настенное крепление": "wall_mounting",
            "Крепление VESA": "vesa_mount",
            "Ширина, см": "width_cm",
            "Высота (с подставкой), см": "height_with_stand_cm",
            "Глубина (с подставкой), см": "depth_with_stand_cm",
            "Высота (без подставки), см": "height_without_stand_cm",
            "Толщина панели, см": "panel_thickness_cm",
            "Вес (с подставкой), кг": "weight_with_stand_kg",
            "Вес (без подставки), кг": "weight_without_stand_kg"
        }

        shop_tv_inverse_mapping = {v: k for k, v in shop_tv_mapping.items()}
        cols = shop_tv_mapping.keys()

        df = df[cols]
        df.rename(columns=shop_tv_mapping, inplace=True)

        for col in ['offer_count','min_price', 'release_date', 'screen_diagonal_inch', 'refresh_rate_hz', 'sound_system_power_w',
                    'num_speakers', 'bluetooth_version', 'num_hdmi_ports', 'num_usb_ports', 'width_cm', 'height_with_stand_cm',
                    'depth_with_stand_cm', 'height_without_stand_cm', 'panel_thickness_cm', 'weight_with_stand_kg', 'weight_without_stand_kg'
                    ]:
            df[col] = df[col].map(ItemDetailsDo.remove_from_bracket)
            df[col] = df[col].astype(float)
        for col in ['curved_screen', 'support_3d', 'smart_tv', 'frameless_design', 'wall_mount', 'hdr', 'gaming_features',
                    'backlight', 'voice_control', 'subwoofer', 'soundbar', 'hdmi_earc_support', 'smart_remote', 'wall_mounting']:
            df[col] = df[col].map(ItemDetailsDo.remove_from_bracket)
            df[col] = df[col].replace({'Есть': 'Да'})
        return df

    def fridge_handler(self, df: pd.DataFrame):
        shop_fridge_mapping = {
            'name': 'name',
            'product_url': 'product_url',
            'offer_count': 'offer_count',
            'min_price': 'min_price',
            "Производитель": "manufacturer",
            "Количество камер": "number_of_chambers",
            "Тип": "type",
            "Компоновка": "layout",
            "No Frost": "no_frost",
            "Зона свежести (BioFresh)": "biofresh_zone",
            "Цвет корпуса": "body_color",
            "Климатический класс": "climate_class",
            "Возможность перенавешивания дверей": "door_reversibility",
            "Дополнительная комплектация холодильного отделения": "additional_fridge_features",
            "Ручки": "handles",
            "Класс энергопотребления": "energy_class",
            "Инверторный компрессор": "inverter_compressor",
            "Тип управления": "control_type",
            "Расположение блока управления": "control_panel_location",
            "Дисплей": "display",
            # numeric
            "Количество компрессоров": "number_of_compressors",
            "Объем холодильной камеры, л": "fridge_volume_l",
            "Объем морозильной камеры, л": "freezer_volume_l",
            "Количество отделений морозильной камеры": "number_of_freezer_compartments",
            "Мощность замораживания, кг/сутки": "freezing_capacity_kg_per_day",
            "Высота, см": "height_cm",
            "Ширина, см": "width_cm",
            "Глубина, см": "depth_cm"
        }

        shop_fridge_inverse_mapping = {v: k for k, v in shop_fridge_mapping.items()}
        cols = shop_fridge_mapping.keys()
        # logger.debug(cols)

        df = df[cols]
        df.rename(columns=shop_fridge_mapping, inplace=True)

        for col in ['offer_count','min_price', 'number_of_compressors', 'fridge_volume_l', 'freezer_volume_l',
                    'number_of_freezer_compartments', 'freezing_capacity_kg_per_day', 'height_cm', 'width_cm',
                    'depth_cm']:
            df[col] = df[col].map(ItemDetailsDo.remove_from_bracket)
            df[col] = df[col].astype(float)
        for col in ['display', 'inverter_compressor', 'door_reversibility', 'biofresh_zone', 'no_frost']:
            df[col] = df[col].map(ItemDetailsDo.remove_from_bracket)
            df[col] = df[col].replace({'Есть': 'Да'})
        return df

    def washing_mashine_handler(self, df: pd.DataFrame):
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

    def process(self, data: Dict[StepNum, Any]) -> Dict[StepNum, Any]:
        # logger.critical(data)
        assert isinstance(data, dict)
        data = data.get("step_0")
        if isinstance(data, list) and not (isinstance(data[0], list) or isinstance(data[0], tuple)):
            df = pd.DataFrame(data)
            df = self.handler_mappring[self.product_type_name](df)
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

