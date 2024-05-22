import streamlit as st
import pandas as pd
import sys
sys.path.append('/home/amstel/llm')
sys.path.append('/home/amstel/llm/src')
from src.mongodb.utils import MongoConnector
from abc import ABC
from loguru import logger


def get_summarization(prod_name: str) -> dict:
    mc = MongoConnector('read', 'scraped_data', 'product_review_summarizations')
    resp = mc.read_one({"product_name": prod_name})
    return resp

class Renderer(ABC):
    column_2_human_name = {}
    column_2_measurement_unit = {}

class WashingMachineRenderer(Renderer):
    product_name = "washing_machine"
    column_2_human_name = {
        'name': None,
        'price': None,
        'rating_value': 'рейтинг',
        'depth': 'глубина',
        'max_load': 'загрузка',
    }
    column_2_measurement_unit = {
        'name': None,
        'price': 'руб.',
        'rating_value': None,
        'depth': 'см.',
        'max_load': 'кг.',
    }

    @classmethod
    def render(cls, input: dict) -> list[str]:
        # input = {
        # 'name': 'Стиральная машина ATLANT СМА 60У1010-00',
        #  'price': 763.0,
        #  'rating_value': 4.5,
        #  'depth': 40.7,
        #  'max_load': 6.0 }
        results = []
        for k,v in input.items():
            s = ''
            human_name = cls.column_2_human_name.get(k)
            if human_name:
                s += human_name
            if isinstance(v, float):
                if int(v) == v:
                    s += f' {int(v)} '
                else:
                    s += f' {v} '
            else:
                s += f' {v} '
            measurement_unit = cls.column_2_measurement_unit.get(k)
            if measurement_unit:
                s += measurement_unit
            results.append(s)
        return results


if __name__ == '__main__':

    df = pd.DataFrame.from_dict({'name':
                           {0: 'Стиральная машина ATLANT СМА 60У1010-00',
                            1: 'Стиральная машина ATLANT СМА 60У1214-01',
                            2: 'Стирально-сушильная машина Candy CSWS4 3642DB/2-07',
                            3: 'Стиральная машина BEKO WRE 6512 BWW (BY)',
                            4: 'Стиральная машина Electrolux SensiCare 600 EW6SN406WI',
                            5: 'Стиральная машина BEKO WSRE6512ZAA',
                            6: 'Стиральная машина CENTEK CT-1901',
                            7: 'Стиральная машина Candy CSS4 1162D1/2-07'},
                       'price': {0: 763.0, 1: 650.0, 2: 1106.86, 3: 641.0, 4: 1900.0, 5: 712.22, 6: 807.26, 7: None},
                       'rating_value': {0: 4.5, 1: 4.5, 2: 4.5, 3: 4.5, 4: 4.5, 5: 4.77, 6: 5.0, 7: 4.71},
                       'depth': {0: 40.7, 1: 40.6, 2: 40.0, 3: 41.5, 4: 40.0, 5: 41.5, 6: 40.0, 7: 40.0},
                       'max_load': {0: 6.0, 1: 6.0, 2: 6.0, 3: 6.0, 4: 6.0, 5: 6.0, 6: 6.0, 7: 6.0}
                       })

    df = df[df['price'] > 0]
    df = df.sort_values(['price', 'rating_value', 'name'], ascending=[True, False, True])

    results = {}
    for i, (_, row) in enumerate(df.head(4).iterrows()):
        results[i+1] = row.to_dict()


    # Create a function to generate feature descriptions
    def generate_features(features):
        features_html = ""
        for i, feature in enumerate(features, start=1):
            features_html += f"<p style='margin-bottom: 10px; font-size: 0.8rem;'>" \
                             f"{feature}" \
                             f"</p>"
        return f"""
        <div style="display: flex; flex-direction: column; align-items: flex-start;">
            {features_html}
        </div>
        """

    # Create a Streamlit app
    st.title("Chat Frontend")
    upper_row = st.container()
    upper_row_columns = st.columns([1.5, 2])
    lower_row = st.container()
    # todo: dynamically define values below based on the number of input items
    lower_row_columns = st.columns([1.5, 2, 1.5, 2, 1.5, 2])

    with upper_row:
        input_data = results.get(1)
        if input_data:

            features = WashingMachineRenderer.render(input_data)
            reviews_summary = get_summarization(input_data.get('name'))
            if reviews_summary:
                advantages = reviews_summary.get('advantages')
                disadvantages = reviews_summary.get('disadvantages')
            with upper_row_columns[0]:
                st.image('image1.jpg', use_column_width=True)
            with upper_row_columns[1]:
                st.markdown(generate_features(features), unsafe_allow_html=True)
                if advantages:
                    st.write('Достоинства:')
                    st.markdown(generate_features(advantages), unsafe_allow_html=True)
                    st.markdown('#')
                if disadvantages:
                    st.write('Недостатки:')
                    st.markdown(generate_features(disadvantages), unsafe_allow_html=True)
                    st.markdown('#')

    with lower_row:
        input_data = results.get(2)
        if input_data:
            features = WashingMachineRenderer.render(input_data)

            with lower_row_columns[0]:
                st.image('image2.jpg', use_column_width=True)
            with lower_row_columns[1]:
                st.markdown(generate_features(features), unsafe_allow_html=True)

        input_data = results.get(3)
        if input_data:
            features = WashingMachineRenderer.render(input_data)

            with lower_row_columns[2]:
                st.image('image3.jpg', use_column_width=True)
            with lower_row_columns[3]:
                st.markdown(generate_features(features), unsafe_allow_html=True)

        input_data = results.get(4)
        if input_data:
            features = WashingMachineRenderer.render(input_data)
            with lower_row_columns[4]:
                st.image('image4.jpg', use_column_width=True)
            with lower_row_columns[5]:
                st.markdown(generate_features(features), unsafe_allow_html=True)