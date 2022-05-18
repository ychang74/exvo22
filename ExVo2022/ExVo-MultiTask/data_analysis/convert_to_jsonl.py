import pandas as pd
from cxpy.io import write_jsonl_into_file


df = pd.read_csv('data_info.csv')
df.fillna(-1, axis=1, inplace=True)
df = df.astype(dtype={'Country': 'int', 'Age': 'float',
                      'Amusement': 'float', 'Awe': 'float',
                      'Awkwardness': 'float', 'Distress': 'float',
                      'Excitement': 'float', 'Fear': 'float',
                      'Horror': 'float', 'Sadness': 'float',
                      'Surprise': 'float', 'Triumph': 'float'})

df = df.rename(columns={'File_ID': 'id', 'Subject_ID': 'speaker_id',
                        'Age': 'age', 'Country': 'country',
                        'Country_string': 'country_str'})

df['id'] = df['id'].apply(lambda x: x.replace("[", "").replace("]", "") + '.wav')
df['audio'] = df['id']
df['audio_path'] = df['id'].apply(lambda x: '/data2/atom/datasets/exvo/wav/' + x)
df['speaker_id'] = df['speaker_id'].apply(lambda x: int(x.split('Speaker_')[-1]) if isinstance(x, str) else x)
df['country_str'] = df['country_str'].apply(lambda x: x.strip().lower() if isinstance(x, str) else str(x))
df['emotion_intensity'] = df[['Amusement', 'Awe',
                              'Awkwardness', 'Distress',
                              'Excitement', 'Fear',
                              'Horror', 'Sadness',
                              'Surprise', 'Triumph']].values.tolist()

df = df.drop(['Amusement', 'Awe',
              'Awkwardness', 'Distress',
              'Excitement', 'Fear',
              'Horror', 'Sadness',
              'Surprise', 'Triumph'], axis=1)


print(df['age'].value_counts())


train_df = df[df['Split'] == 'Train']
train_df = train_df.drop('Split', axis=1)
valid_df = df[df['Split'] == 'Val']
valid_df = valid_df.drop('Split', axis=1)
test_df = df[df['Split'] == 'Test']
test_df = test_df.drop('Split', axis=1)

df = df.drop('Split', axis=1)

col = df.columns.tolist()

col.sort()

write_jsonl_into_file(df.to_dict(orient='records'), 'data.jsonl')
write_jsonl_into_file(train_df.to_dict(orient='records'), 'train.jsonl')
write_jsonl_into_file(valid_df.to_dict(orient='records'), 'dev.jsonl')
write_jsonl_into_file(test_df.to_dict(orient='records'), 'test.jsonl')

