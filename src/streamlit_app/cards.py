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
from api.credit_interset_calculator import InterestCalculator

radiobutton_options = {
                    "12 месяцев": 12,
                    "24 месяца": 24,
                    "36 месяцев": 36
                }



class ItemDisplay:

    def __init__(self, items: list[dict], duration_2_terms: dict[int, str], sql_result_ix: int):
        self.items = items
        self.duration_2_terms = duration_2_terms
        self.loan_terms_defined = False
        self.sql_result_ix = sql_result_ix

    def display_item(self, item, upper=False):
        if upper:
            description = ''
            for k,v in item.items():
                if k in ['price', 'rating_value', 'rating_count', 'depth', 'max_load', 'drying']:
                    if v:
                        description += f'{k}: {v}<br>'
            create_preview_card(
                url=item.get('product_url'),
                title=item.get('name'),
                image_url=item.get('product_image_url'),
                description=description,
            )
        else:
            st.markdown(f"""<img src="{item['product_image_url']}" alt="{item.get('name')}" style="border-radius: 4px; width: 100%; max-width: 110px; height: auto; object-fit: cover;">""", unsafe_allow_html=True)
            st.markdown(
                f"<div style='text-align: left;'><a href='{item['product_url']}' style='text-decoration: none; color: black; font-size: 14px;'><strong>{item['name']}</strong></a></div>",
                unsafe_allow_html=True)
            if item.get('price'): st.write(f"<div style='text-align: left; font-size: 14px;'>price: {item['price']}</div>",
                                           unsafe_allow_html=True)
            if item.get('rating_value'): st.write(
                f"<div style='text-align: left; font-size: 14px;'>rating_value: {item['rating_value']}</div>", unsafe_allow_html=True)
            if item.get('rating_count'): st.write(
                f"<div style='text-align: left; font-size: 14px;'>rating_count: {item['rating_count']}</div>", unsafe_allow_html=True)
            if item.get('depth'): st.write(f"<div style='text-align: left; font-size: 14px;'>depth: {item['depth']}</div>",
                                           unsafe_allow_html=True)
            if item.get('max_load'): st.write(f"<div style='text-align: left; font-size: 14px;'>max_load: {item['max_load']}</div>",
                                              unsafe_allow_html=True)
            if item.get('drying'): st.write(f"<div style='text-align: left; font-size: 14px;'>drying: {item['drying']}</div>",
                                            unsafe_allow_html=True)


    def display_upper_row(self):
        """ Display the upper row with a big item. """
        if self.items:
            st.empty()
            st.markdown("<h2 style='text-align: center; font-weight: bold; font-size: 18px;'>Лучший выбор</h2>", unsafe_allow_html=True)
            # st.write("### Лучший выбор")
            col1, col2, col3 = st.columns([1, 10, 1])
            with col1:
                st.empty()
            with col2:
                self.display_item(self.items[0], upper=True)
                st.empty()

                # Tailwind CSS for styling

                tailwind_css = """
                <html> 
                <head> 
                  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet"> 
                </head> 
                </html>
                """
                st.markdown(tailwind_css, unsafe_allow_html=True)

                # Custom CSS to make radio buttons horizontal
                # st.markdown(
                #     """
                #     <style>
                #     div[data-baseweb="radio"] > div {
                #         display: flex;
                #         flex-direction: row;
                #     }
                #     </style>
                #     """,
                #     unsafe_allow_html=True
                # )
                _ = st.radio(
                    "Выберите срок кредита:",
                    list(radiobutton_options.keys()),
                    key=f'selected_option_{self.sql_result_ix}',
                    format_func=lambda x: x,
                    horizontal=True,
                    # on_change= self.on_change_callback,
                )
                # if not self.loan_terms_defined:
                #
                #     self.on_change_callback()
                try:
                    self.result_placeholder
                except:
                    self.result_placeholder = st.empty()
                self.on_change_callback()
                st.link_button(label="Оформить",
                               url="https://www.sber-bank.by/credit-potreb/online-credit/conditions",
                               type="primary",
                               use_container_width=True)

            with col3:
                st.empty()

    def on_change_callback(self):
        selected_str = st.session_state[f'selected_option_{self.sql_result_ix}']
        logger.critical(f'selected_option_{self.sql_result_ix} --- {selected_str}')
        calculator = InterestCalculator()
        selected_int = radiobutton_options[selected_str]
        loan_terms = calculator.gpt4o(self.items[0].get('price'), selected_int)
        self.result_placeholder.markdown(loan_terms)
        # self.loan_terms_defined = True

    def display_lower_row(self, lower_grid_cols):
        """ Display the lower row with configurable number of small items. """
        if lower_grid_cols > 0 and len(self.items) > 1:
            st.empty()
            st.markdown("<h4 style='text-align: center; font-weight: bold;'>Хорошие альтернативы</h4>", unsafe_allow_html=True)
            # Start the styled container
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
            # Close the styled container


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
        <img src="{image_url}" alt="{title}" style="border-radius: 8px; width: 110%; max-width: 150px; height: auto; object-fit: cover;">
    </a>
    <div style="flex-grow: 1;">
        <h4><a href="{url}" target="_blank" style="text-decoration: none; color: #000; font-weight: bold;">{title}</a></h4>
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







# def generate_features(features):
#     features_html = ""
#     for i, feature in enumerate(features, start=1):
#         features_html += f"<p style='margin-bottom: 10px; font-size: 0.8rem;'>" \
#                          f"{feature}" \
#                          f"</p>"
#     return f"""
#     <div style="display: flex; flex-direction: column; align-items: flex-start;">
#         {features_html}
#     </div>
#     """


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

    st.empty()