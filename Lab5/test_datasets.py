
import pytest
import os
import pandas as pd
import pickle


datasets = {}
files = os.listdir('./')
for df_file in files:
    if df_file.endswith('.csv'):
        print(df_file)
        datasets[df_file] = pd.read_csv(os.path.join('./', df_file))

pkl_filename = 'model.pkl'
with open(pkl_filename, 'rb') as file_:
    model = pickle.load(file_)



def test_metric_crash():
    '''Протестируем метрику'''
    for dataset in datasets:
        test_y = datasets[dataset]['D']
        test_X = datasets[dataset].drop('D', axis=1)
        assert model.score(test_X, test_y) > 0.98


def test_std():
    '''Протестируем дисперсию'''
    for dataset in datasets:
        assert datasets[dataset]['B'].std() < 50


def test_median():
    '''Протестируем медиану'''
    for dataset in datasets:
        assert 150 < datasets[dataset]['B'].median() < 300
