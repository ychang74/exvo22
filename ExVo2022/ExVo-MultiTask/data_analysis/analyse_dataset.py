import pandas as pd
from cxpy.io import write_jsonl_into_file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set()

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

train_df = df[df['Split'] == 'Train']
train_df = train_df.drop('Split', axis=1)
valid_df = df[df['Split'] == 'Val']
valid_df = valid_df.drop('Split', axis=1)


"""

Age vs Country

"""

df = df[df['Split'] != 'Test']

age_list = list(df['age'].unique())
age_list.sort()

chi_count_list = []
sou_count_list = []
usa_count_list = []
ven_count_list = []

bin_size = 2.5

age_bin_list = np.arange(int(min(age_list)), int(max(age_list)), bin_size)

for i, age in enumerate(age_bin_list):

    chi_count_list.append(len(df[(df['age'] >= age) & (df['age'] < age + bin_size) & (df['country'] == 0)]))
    sou_count_list.append(len(df[(df['age'] >= age) & (df['age'] < age + bin_size) & (df['country'] == 1)]))
    usa_count_list.append(len(df[(df['age'] >= age) & (df['age'] < age + bin_size) & (df['country'] == 2)]))
    ven_count_list.append(len(df[(df['age'] >= age) & (df['age'] < age + bin_size) & (df['country'] == 3)]))

chi_count_list = np.array(chi_count_list)
sou_count_list = np.array(sou_count_list)
usa_count_list = np.array(usa_count_list)
ven_count_list = np.array(ven_count_list)

age_bin_list = age_bin_list + 0.5 * bin_size

plt.figure()
plt.bar(age_bin_list, chi_count_list, color='r')
plt.bar(age_bin_list, sou_count_list, bottom=chi_count_list, color='b')
plt.bar(age_bin_list, usa_count_list, bottom=chi_count_list + sou_count_list, color='y')
plt.bar(age_bin_list, ven_count_list, bottom=chi_count_list + sou_count_list + usa_count_list, color='g')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age vs Country Count')
plt.legend(['China', 'South Africa', 'USA', 'Venezuela'])
plt.savefig('figures/age_vs_country_count.png')

sum_count_list = chi_count_list + sou_count_list + usa_count_list + ven_count_list
chi_ratio_list = chi_count_list / sum_count_list
sou_ratio_list = sou_count_list / sum_count_list
usa_ratio_list = usa_count_list / sum_count_list
ven_ratio_list = ven_count_list / sum_count_list

plt.figure()
plt.bar(age_bin_list, chi_ratio_list, color='r')
plt.bar(age_bin_list, sou_ratio_list, bottom=chi_ratio_list, color='b')
plt.bar(age_bin_list, usa_ratio_list, bottom=chi_ratio_list + sou_ratio_list, color='y')
plt.bar(age_bin_list, ven_ratio_list, bottom=chi_ratio_list + sou_ratio_list + usa_ratio_list, color='g')
plt.xlabel('Age')
plt.ylabel('Ratio')
plt.title('Age vs Country Ratio')
plt.legend(['China', 'South Africa', 'USA', 'Venezuela'])
plt.savefig('figures/age_vs_country_ratio.png')


"""

Country vs Emotion

"""

emotion_list = ['Amusement', 'Awe',
              'Awkwardness', 'Distress',
              'Excitement', 'Fear',
              'Horror', 'Sadness',
              'Surprise', 'Triumph']

fig, axes = plt.subplots(2, 5, figsize=(16,8))
fig.suptitle('Country vs Emotion Count')

n_col = 5

for i, emotion in enumerate(emotion_list):
    x_loc = int(i / n_col)
    y_loc = int(i % n_col)

    sns.histplot(ax=axes[x_loc, y_loc], data=df[[emotion, 'country']], x=emotion, hue='country', element="step", legend=False)
    plt.legend(['Venezuela', 'USA', 'South Africa', 'China'])

plt.savefig('figures/country_vs_emotion_count.png')


"""

Age vs Emotion

"""

fig, axes = plt.subplots(2, 5, figsize=(16,8))
fig.suptitle('Age vs Emotion Count')

df['age_bin'] = df['age'].apply(lambda x: int((x-20)/5) * 5 + 22.5)

print(df['age_bin'].value_counts())

bin_size = 5
n_col = 5

for i, emotion in enumerate(emotion_list):
    x_loc = int(i / n_col)
    y_loc = int(i % n_col)

    sns.histplot(ax=axes[x_loc, y_loc], data=df[[emotion, 'age_bin']], x=emotion, hue='age_bin', element="step", legend=False)
    plt.legend(['35-40', '30-35', '25-30', '20-25'])

plt.savefig('figures/age_vs_emotion_count.png')
# plt.show()
