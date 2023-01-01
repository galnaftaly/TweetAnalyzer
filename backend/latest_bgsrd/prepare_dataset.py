import pandas as pd
import os

dataset = 'ectf'
datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets', dataset))
labels = {'true': 1, 'fake': 0}

real_data = pd.read_csv(os.path.join(datasets_dir,'true.csv'))
fake_data = pd.read_csv(os.path.join(datasets_dir, 'fake.csv'))
real_data = real_data.loc[:, ~real_data.columns.str.contains('^Unnamed')]
real_data['label'] = labels['true']

fake_data = fake_data.loc[:, ~fake_data.columns.str.contains('^Unnamed')]
fake_data['label'] = labels['fake']

df = pd.concat([real_data.head(200), fake_data.head(200)], join = 'inner')
df = df.sample(frac=1).reset_index(drop = True)


with open(os.path.join(datasets_dir, '{}.csv'.format(dataset)), mode = 'w', encoding = 'utf-8', newline = '') as f:
    f.write(df[['text', 'label']].to_csv(index = False))

