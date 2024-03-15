import pandas as pd
data = pd.read_excel('data.xlsx')
data['lic_acc'] = data['lic_acc'].astype(str)
data['service_name_anon'] = data.apply(lambda x: x['service_name'].replace(x['lic_acc'], ''), axis=1)
data['service_name_anon'] = data['service_name_anon'].str.strip()
data.to_excel('data_anon.xlsx')