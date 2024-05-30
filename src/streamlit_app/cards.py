import numpy as np
import streamlit as st
import pandas as pd
import sys
sys.path.append('/home/amstel/llm')
sys.path.append('/home/amstel/llm/src')
from src.mongodb.utils import MongoConnector
from abc import ABC
from loguru import logger
from backend_ops import DataServer


class ItemDisplay:
    def __init__(self, items):
        self.items = items

    def display_item(self, item):
        logger.debug(item)
        st.image(item['product_image_url'], use_column_width=True)
        st.markdown(f"[**{item['name']}**]({item['product_url']})")
        st.write(f"price: {item['price']}")
        if item.get('rating_value'): st.write(f"rating_value: {item['rating_value']}")
        if item.get('rating_count'): st.write(f"rating_count: {item['rating_count']}")
        if item.get('depth'): st.write(f"depth: {item['depth']}")
        if item.get('max_load'): st.write(f"max_load: {item['max_load']}")
        if item.get('drying'): st.write(f"drying: {item['drying']}")

    def display_upper_row(self):
        """ Display the upper row with a big item. """
        if self.items:
            st.write("### Upper Row")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.empty()
            with col2:
                self.display_item(self.items[0])
            with col3:
                st.empty()

    def display_lower_row(self, lower_grid_cols):
        """ Display the lower row with configurable number of small items. """
        if lower_grid_cols > 0 and len(self.items) > 1:
            st.write("### Lower Row")
            lower_grid_cols_2_empty_cols = {
                1: 3,
                2: 2,
                3: 1
            }
            # Calculate empty columns for centering
            empty_cols = lower_grid_cols_2_empty_cols.get(lower_grid_cols, 0)
            cols = [st.empty()] * empty_cols + st.columns(lower_grid_cols) + [st.empty()] * empty_cols
            for idx, col in enumerate(cols):
                if idx >= empty_cols and idx < empty_cols + lower_grid_cols and idx - empty_cols + 1 < len(self.items):
                    with col:
                        self.display_item(self.items[idx - empty_cols + 1])

    def display_grid(self, lower_grid_cols):
        """ Display the grid with upper and lower rows. """
        if not (0 <= lower_grid_cols <= 3):
            st.error("Number of columns in the lower grid must be between 0 and 3.")
            return

        self.display_upper_row()
        self.display_lower_row(lower_grid_cols)



def create_preview_card(
        url="https://shop.by/stiralnye_mashiny/lg_f2j3ws2w/",
        title="Стиральная машина LG F2J3WS2W",
        image_url="https://shop.by/images/lg_f2j3ws2w_1.webp",
        description="Custom description"
):
    """Function to create a website preview card in Streamlit."""
    card_html = f"""
    <div style="display: flex; flex-direction: row; align-items: flex-start; gap: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 8px; box-shadow: 0 4px 6px 0 rgba(0,0,0,0.1);">
        <a href="{url}" target="_blank" style="text-decoration: none; color: #000;">
            <img src="{image_url}" alt="{title}" style="width: 120px; height: 120px; border-radius: 8px; object-fit: cover;">
        </a>
        <div style="flex-grow: 1;">
            <h4><a href="{url}" target="_blank" style="text-decoration: none; color: #000;">{title}</a></h4>
            <p>{description}</p>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)



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
                if np.isnan(v): continue
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


if __name__ == '__main__':

    df = pd.read_csv('data.csv')
    df = df[df['price'] > 0]
    df = df.sort_values(['price', 'rating_value', 'name'], ascending=[True, False, True])
    df.drop_duplicates(subset=['name'], inplace=True, keep='last')
    df = df.head(4)
    n = df.shape[0] - 1
    data_server = DataServer()
    items = data_server.collect_data(df['name'])
    logger.info(items)

    item_display = ItemDisplay(items)
    item_display.display_grid(lower_grid_cols=n)