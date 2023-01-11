import pandas as pd
import os
from sklearn.model_selection import train_test_split

'''
dataset format:
text, class and maybe additional non-relevant columns
'''

dataset = 'twitter'
datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets', dataset))

df = pd.read_csv(os.path.join(datasets_dir, 'twitter_not_clean.csv'), index_col = False)
df = df.drop(['id', 'keyword', 'location'], axis = 1)
df = df.rename(columns = {'target': 'label'})

real_data = df[df.label == 0]
fake_data = df[df.label == 1]

df_new = pd.concat([real_data, fake_data], join = 'inner')
df_new = df_new.sample(frac=1).reset_index(drop = True)

X = df_new.iloc[:, 0:-1]
y = df_new.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = False)
df_train = pd.concat([X_train, y_train], axis = 1)
df_train['type'] = 'train'
df_test = pd.concat([X_test, y_test], axis = 1)
df_test['type'] = 'test'
df_text_splitted = pd.concat([df_train, df_test]).reset_index(drop = True)
df_text_splitted.to_csv(os.path.join(datasets_dir, '{}.csv'.format(dataset)), index = False)

