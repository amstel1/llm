import sys
sys.path.append('/home/amstel/llm/src')
import pandas as pd
import pickle
from postgres.postgres_utils import insert_data
from loguru import logger



def remove_from_bracket(s):
    if isinstance(s, str):
        position = s.find('(')
        if position != -1:
            return s[:position].strip(' ')
    return s


def washing_mashine_preprocess(df: pd.DataFrame):
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
    df = df[cols]
    df.rename(columns=shop_washing_machine_mapping, inplace=True)

    for col in ['min_price', 'max_load', 'depth', 'width', 'height', 'weight', 'max_rpm', 'water_consumption',
                'programs_number']:
        df[col] = df[col].map(remove_from_bracket)
        df[col] = df[col].astype(float)
    for col in ['has_display', 'additional_rinsing',
                'spinning_speed_selection', 'drying',
                'can_add_clothes', 'cancel_spinning',
                'light_ironing', 'direct_drive', 'inverter_motor']:
        df[col] = df[col].map(remove_from_bracket)
        df[col] = df[col].replace({'Есть': 'Да'})
    return df


if __name__ == '__main__':
    """Shop.by product details (washing machines)-> postgres"""
    with open('/home/amstel/llm/out/shop_output_products.pkl', 'rb') as f:
        products = pickle.load(f)
    df = pd.DataFrame(products)
    df = washing_mashine_preprocess(df)
    # print(df.head())
    insert_data(
        df,
        schema_name='scraped_data',
        table_name='washing_machine_details'
    )
    logger.info('success')
