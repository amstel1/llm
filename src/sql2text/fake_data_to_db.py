import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from faker import Faker
import sqlite3

fields = ['local_amount', 'currency_code', 'category_id', 'subcategory_id', 'mcc_id', 'transaction_datetime', 'event_id']

class DataGenerator():
    def __init__(self,  fields: list[str]):
        self.date_range = pd.date_range('2023-03-01 00:00:00', '2024-02-29 23:59:59', freq='min').strftime('%Y-%m-%d %H:%M')

        self.fields = fields
        self.type_mapping = {
            # 'prty_id':'bounds', 
            'local_amount':'bounds', 
            'currency_code':'options', 
            'category_id':'options', 
            'subcategory_id':'options', 
            'mcc_id':'options', 
            # 'trans_dttm':'bounds', 
            'event_id':'bounds',
        }
        self.bounds = {
            # 'prty_id': (10000, 9607176),
            'local_amount': (0.01, 500),
            # 'trans_dttm': (pd.Timestamp('2023-03-01 00:00:00'), pd.Timestamp('2023-02-29 23:59:59')),
            'event_id': (66_000_000, 67_000_000-1),

            
        }
        self.options = {
            'currency_code':['usd', 'eur', 'rub', 'byn'], 
            'category_id':list(range(1001, 1010)),
            'subcategory_id':list(range(1, 52)),
            'mcc_id': list(range(4000, 4999)),
        }
        
    def generate(self, prty_id:int, size:int):
        # self.prty_id = prty_id
        # self.size = size

        data_dict = {}
        for col in self.fields:
            type_mapping = self.type_mapping.get(col)
            if type_mapping == 'bounds':
                lb, ub = self.bounds.get(col)
                generated_data = np.random.uniform(lb, ub, size)
            elif type_mapping == 'options':
                generated_data = random.choices(self.options.get(col), k=size)
            data_dict[col] = generated_data
        data_dict['client_id'] = itertools.repeat(prty_id, size)
        data_dict['transaction_datetime'] = random.choices(self.date_range, k=size)
        
        return data_dict
    
if __name__ == '__main__':
    dgp = DataGenerator(fields=fields,)
    
    results = []
    
    for j in range(1):
        size = 527 #np.random.randint(1, 1000)
        prty_id = 2607176 #random.sample(list(range(10000, 9607176)), k=1)[0]
        ret_val = pd.DataFrame(dgp.generate(prty_id=prty_id, size=size,))
        results.append(ret_val)
    
    df = pd.concat(results, axis=0)
    df['local_amount'] = df['local_amount'].round(2)
    df['event_id'] = df['event_id'].astype(int)

    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])

    conn = sqlite3.connect('fake_sbol.db')
    df.to_sql('pfm_flat_table', conn, if_exists='fail',)

    print('success')