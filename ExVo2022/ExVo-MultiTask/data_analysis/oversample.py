import pandas as pd
# from cxpy.io import write_jsonl_into_file
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE

def df_count(df, key):
    values = list(df[key].unique())
    values.sort()
    for i, value in enumerate(values):
        print('{} {} has #samples: {}'.format(key, value, len(df[df[key] == value])))

def classifier(row):
    age_list = [20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5]
    country_list = [0, 1, 2, 3]
    bin_size = 2.5
    for country in country_list:
        for i, age in enumerate(age_list):
            if row['country'] == country and row['age'] >= age and row['age'] < age + bin_size:
                return country * 10  + i

def classifier2(row):
    co_list = ['China', 'South Africa', 'USA', 'Venezuela']
    return co_list[int(row['country'])]

df = pd.read_csv('data_info.csv')
df.dropna(axis=0, how='any', inplace=True)
# df.fillna(-1, axis=1, inplace=True)
df = df.astype(dtype={'Country': 'int', 'Age': 'float',
                      'Amusement': 'float', 'Awe': 'float',
                      'Awkwardness': 'float', 'Distress': 'float',
                      'Excitement': 'float', 'Fear': 'float',
                      'Horror': 'float', 'Sadness': 'float',
                      'Surprise': 'float', 'Triumph': 'float'})

df = df.rename(columns={'File_ID': 'id', 'Subject_ID': 'speaker_id',
                        'Age': 'age', 'Country': 'country',
                        'Country_string': 'country_str'})

df['id'] = df['id'].apply(lambda x: int(x.replace("[", "").replace("]", "")))
# df['audio'] = df['id']
# df['audio_path'] = df['id'].apply(lambda x: '/data2/atom/datasets/exvo/wav/' + x)
df['speaker_id'] = df['speaker_id'].apply(lambda x: int(x.split('Speaker_')[-1]) if isinstance(x, str) else x)
df['country_str'] = df['country_str'].apply(lambda x: x.strip().lower() if isinstance(x, str) else str(x))

# Avoid the non-numeric values for SMOTE
country_str = df.pop('country_str')
df['Split'] = df['Split'].apply(lambda x: 1 if x == 'Train' else 0)

# Oversampling
df['age_country'] = df.apply(classifier, axis=1)
# df_count(df, 'age_country')
smo = SMOTE(random_state=42)
df_smo, _ = smo.fit_resample(df, df['age_country'])
# df_count(df_smo, 'age_country')

# Modify/add some columns back
df = df_smo
df['country_str'] = df.apply(classifier2, axis=1)
df['Split'] = df['Split'].apply(lambda x: 'Train' if x == 1 else 'Val')
df['id'] = df['id'].apply(lambda x: str(x).zfill(5) + '.wav')
df['audio'] = df['id']
df['audio_path'] = df['id'].apply(lambda x: '/data2/atom/datasets/exvo/wav/' + x)
df.drop(['age_country'], axis=1, inplace=True)
print('#### Afer Oversampling ####')
print(df.shape)
print(df.loc[0,:])
df.to_csv('os_data_info.csv', sep='\t', index=False)
