import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils.utils import *

dataset = 'ectf'
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets'))
data = os.path.join(datasets_dir, dataset, '{}.csv'.format(dataset))

def main():
    df_text = pd.read_csv(data)
    df_text.dropna(inplace = True)
    X = df_text.iloc[:, 0:-1]
    y = df_text.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42, shuffle = False) # 0.1 x 0.8 = 0.08
    df_train = pd.concat([X_train, y_train], axis = 1)
    df_train['type'] = 'train'
    df_val = pd.concat([X_val, y_val], axis = 1)
    df_val['type'] = 'val'
    df_test = pd.concat([X_test, y_test], axis = 1)
    df_test['type'] = 'test'
    df_text_splitted = pd.concat([df_train, df_val, df_test]).reset_index(drop = True)
    save_as_pickle(os.path.join(data_dir, 'df_data.pkl'), df_text_splitted)

if __name__ == "__main__":
    main()