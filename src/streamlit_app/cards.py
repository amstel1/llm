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
from etl_jobs.attribute_mappings import name_2_attribute
from text2sql.prod_sql_to_text import extract_where_attributes

radiobutton_options = {
                    "12 месяцев": 12,
                    "24 месяца": 24,
                    "36 месяцев": 36
                }


class ItemDisplay:

    def __init__(self, items: list[dict], duration_2_terms: dict[int, str], sql_result_ix: int, necessary_product_attributes_list:list[str]):
        self.items = items
        self.duration_2_terms = duration_2_terms
        self.loan_terms_defined = False
        self.sql_result_ix = sql_result_ix
        self.necessary_product_attributes_list = necessary_product_attributes_list
        if not self.necessary_product_attributes_list:
            self.necessary_product_attributes_list = []
        for col in ['price', 'rating_value', 'rating_count', 'name', 'product_image_url', 'product_url', 'offer_count']:
            if col not in self.necessary_product_attributes_list:
                self.necessary_product_attributes_list.append(col)

    def display_item(self, old_item: dict, upper=False):
        inverse_product_attributes_dict = name_2_attribute.get(
            st.session_state.context['sql_schema'])  # dict {rus property name: eng property name}
        all_product_attributes_dict = {v: k for k, v in inverse_product_attributes_dict.items()}
        all_product_attributes_dict.update({
            'name': 'Название',
            'price':'Цена',
            'rating_value':'Рейтинг',
            'rating_count':'Количество оценок',
        })
        assert all_product_attributes_dict
        # logger.info(f'collect_data: {items}')

        item = {}
        for k,v in old_item.items():
            if k in self.necessary_product_attributes_list and k in all_product_attributes_dict and v and k not in item:
                # logger.info(f'0611: remap {k}, {all_product_attributes_dict[k]}, {v}')
                item[all_product_attributes_dict[k]] = v
            elif k in self.necessary_product_attributes_list and v and k not in item:
                item[k] = v
                # logger.info(f'0611: no remap {k}, {v}')

        logger.debug(f'0611 item: {item}')
        # logger.debug(f'0611 necessary_product_attributes_list: {self.necessary_product_attributes_list}')


        if upper:
            description = ''
            for k,v in item.items():
                if k in all_product_attributes_dict.keys() and k not in ('product_url', 'product_image_url', 'Название', 'offer_count'):
                    logger.debug(f'k: 0711 -- {k}, {v}')
                    if v:
                        description += f'{all_product_attributes_dict[k]}: {v}<br>'
                elif k not in ('product_url', 'product_image_url', 'Название', 'offer_count'):
                    description += f'{k}: {v}<br>'
            logger.critical(f'0711-2 {description}')
            create_preview_card(
                url=item.get('product_url'),
                title=item.get('Название'),
                image_url=item.get('product_image_url'),
                description=description,
            )
        else:
            item.pop('offer_count')
            st.markdown(f"""<img src="{item.get('product_image_url')}" alt="{item.get('Название')}" style="border-radius: 4px; width: 100%; max-width: 110px; height: auto; object-fit: cover;">""", unsafe_allow_html=True)
            st.markdown(
                f"<div style='text-align: left;'><a href='{item.get('product_url')}' style='text-decoration: none; color: black; font-size: 14px;'><strong>{item.get('Название')}</strong></a></div>",
                unsafe_allow_html=True)
            if item.get('Цена'): st.write(f"<div style='text-align: left; font-size: 14px;'>Цена: {item.get('Цена')}</div>",
                                           unsafe_allow_html=True)
            if item.get('Рейтинг'): st.write(
                f"<div style='text-align: left; font-size: 14px;'>Рейтинг: {item.get('Рейтинг')}</div>", unsafe_allow_html=True)
            if item.get('Количество оценок'): st.write(
                f"<div style='text-align: left; font-size: 14px;'>Количество оценок: {item.get('Количество оценок')}</div>", unsafe_allow_html=True)
            for k, v in item.items():
                if k not in ('product_url', 'product_image_url', 'Название', 'Цена', 'Рейтинг', 'Количество оценок', ):
                    st.write(f"<div style='text-align: left; font-size: 14px;'>{k}: {v}</div>", unsafe_allow_html=True)
            # if item.get('depth'): st.write(f"<div style='text-align: left; font-size: 14px;'>лубина, см: {item.get('depth')}</div>",
            #                                unsafe_allow_html=True)
            # if item.get('max_load'): st.write(f"<div style='text-align: left; font-size: 14px;'>Загрузка, кг: {item.get('max_load')}</div>",
            #                                   unsafe_allow_html=True)
            # if item.get('drying'): st.write(f"<div style='text-align: left; font-size: 14px;'>Есть сушка: {item.get('drying')}</div>",
            #                                 unsafe_allow_html=True)


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
    logger.error(f'1211 - create_preview_card, description: {description}')
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