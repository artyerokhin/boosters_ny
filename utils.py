# Различные утилиты для соревания
import pandas as pd
import numpy as np

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial
import requests
import os
import time
import geopandas as gpd

import datetime
import pytz

from math import sin, cos, sqrt, atan2, radians
from tqdm import tqdm

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from zipfile import ZipFile
from tzwhere import tzwhere


def unzip_to_path(archive):
    """ Извлечение из zip формата """
    path_name = '-'.join(archive.split('.')[:-1])
    zf = ZipFile(archive, 'r')
    zf.extractall(path_name)
    zf.close()


def rmse(y_true, y_pred):
    "calculate rmse"
    return sqrt(mean_squared_error(y_true, y_pred))


def load_geo_dataframe(filename, path='data'):
    """ load geo dataframe with point geometries from files """
    df = None
    for f in tqdm(os.listdir(path)):
        if os.path.isdir(os.path.join(path,f)):
            if df is None:
                df = gpd.read_file(os.path.join(path, f, filename))
            else:
                df = df.append(gpd.read_file(os.path.join(path, f, filename)))

    df['lat'] = df.geometry.y
    df['long'] = df.geometry.x

    return df


def distance(x, y):
    """
    Параметры
    ----------
    x : tuple, широта и долгота первой геокоординаты
    y : tuple, широта и долгота второй геокоординаты

    Результат
    ----------
    result : дистанция в километрах между двумя геокоординатами
    """
    R = 6373.0 # радиус земли в километрах

    lat_a, long_a, lat_b, long_b = map(radians, [*x,*y])
    dlon = long_b - long_a
    dlat = lat_b - lat_a
    a = sin(dlat/2)**2 + cos(lat_a) * cos(lat_b) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def nearest_distances(data, initial_data=None, metric=distance, n_neighbors=10, return_more=None):
    """
    Получение ближайших к точкам дистанций:
    Аргументы:
        data (pd.DataFrame) - набор данных, для которого и от которого
            считаются расстояния
        initial_data (pd.DataFrame) - точки, для которых нужно отсчитывать
            расстояния до ближайших
        metric (python fucntion) - метрика расстояния
        n_neighbors (int) - число соседей
        return_more (str) - дополнительный столбец, который возвращается вместе
            с расстоянием (например, население ближайших населенных пунктов)
    Возвращает:
        distances (np.array) - расстояния до ближайших точек
        more (np.array) - дополнительный набор данных для ближайших городов
    """
    start_time = time.time()
    knc = KNeighborsClassifier(metric=metric)
    dots = data[['lat','long']].dropna()
    knc.fit(X=dots , y=np.ones(dots.shape[0]))

    if initial_data is None:
        distances, indexes = knc.kneighbors(X=dots, n_neighbors=n_neighbors)
    else:
        distances, indexes = knc.kneighbors(X=initial_data, n_neighbors=n_neighbors)

    print('Finded neares distances in {}'.format(time.time() - start_time))

    if return_more:
        more = data[[return_more]].values[indexes]
        return distances, more
    else:
        return distances


def add_distance_features(df, distances, feature_prefix):
    """
        Добавляем признаки дистанции:
        Аргументы:
            df (pd.DataFrame) - исходный датафейм
            distances (np.array) - расстояния
            feature_prefix (str) - префикс признака (для наименования столбцов)
        Возвращает:
            df_copy (pd.DataFrame) - копия исодного датафрейма с изменениями
    """
    df_copy = df.copy() # копия на тот случай, чтобы исходный датафрейм не трогать

    for n in range(distances.shape[1]):
        df_copy['{}_{}'.format(feature_prefix, n)] = distances[:, n]

    return df_copy


def add_stat_distance_features(df, feature_prefix):
    """
        Добавляем признаки распредения дистанций (среднее, медиана и т.п.)
        Аргументы:
            df (pd.DataFrame) - исходный датафейм
            feature_prefix (str) - префикс признака (для наименования столбцов)
        Возвращает:
            df_copy (pd.DataFrame) - копия исодного датафрейма с изменениями
    """
    df_copy = df.copy()

    df_copy['{}_mean'.format(feature_prefix)] = df_copy.iloc[:, df_copy.columns.str.contains(feature_prefix)].mean(axis=1)
    df_copy['{}_median'.format(feature_prefix)] = df_copy.iloc[:, df_copy.columns.str.contains(feature_prefix)].median(axis=1)
    df_copy['{}_std'.format(feature_prefix)] = df_copy.iloc[:, df_copy.columns.str.contains(feature_prefix)].std(axis=1)
    df_copy['{}_interquartile'.format(feature_prefix)] = df_copy.iloc[:, df_copy.columns.str.contains(feature_prefix)].quantile(0.75, axis=1) - \
                               df_copy.iloc[:, df_copy.columns.str.contains(feature_prefix)].quantile(0.25, axis=1)

    df_copy['{}_min_max'.format(feature_prefix)] = df_copy.iloc[:, df_copy.columns.str.contains(feature_prefix)].max(axis=1) - \
                         df_copy.iloc[:, df_copy.columns.str.contains(feature_prefix)].min(axis=1)

    return df_copy


def hyperopt_xgb(X, y, N):
    """ Поиск гиперпараметров посредством гиперопт """

    print('hyperopt..')

    start_time = time.time()

    # search space to pass to hyperopt
    fspace = {'learning_rate':hp.choice('learning_rate', [0.03,0.05,0.07,0.1,0.2]),
                'n_estimators': hp.choice('n_estimators', [100,200,300,400,500,600,700,800,900,1000]),
                'eta': hp.quniform('eta', 0.025, 0.25, 0.025),
                'max_depth':  hp.choice('max_depth', np.arange(5, 14, dtype=int)),
                'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
                'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                'gamma': hp.quniform('gamma', 0, 1, 0.05),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                'alpha' :  hp.quniform('alpha', 0, 10, 1),
                'lambda': hp.quniform('lambda', 0, 2, 0.1)}

    # objective function to pass to hyperopt
    def objective(params):

        iteration_start = time.time()

        params.update({'random_state': 1, 'n_jobs':-1})

        losses = []
        print(params)
        model = xgb.XGBRegressor(**params)

        kf = KFold(n_splits=5, random_state=1, shuffle=True)

        for train_index, test_index in kf.split(X_):

            X_train, X_valid = X_.loc[train_index, :], X_.loc[test_index, :]
            Y_train, Y_valid = Y_[train_index], Y_[test_index]
            model.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)], early_stopping_rounds=20, eval_metric='rmse',
                                            verbose=False)

            pred = model.predict(X_valid)
            losses.append(rmse(Y_valid, pred))

        iteration_time = time.time()-iteration_start
        loss = np.mean(losses)
        print('iteration time %.1f, loss %.5f' % (iteration_time, loss))

        return {'loss': loss, 'status': STATUS_OK,
                'runtime': iteration_time,
                'params': params}

    # object with history of iterations to pass to hyperopt
    trials = Trials()

    # loop over iterations of hyperopt
    for t in range(N):
        # run hyperopt, n_startup_jobs - number of first iterations with random search
        best = fmin(fn=objective, space=fspace, algo=partial(tpe.suggest, n_startup_jobs=10),
                    max_evals=50, trials=trials)

    print('best parameters', trials.best_trial['result']['params'])

    return trials.best_trial['result']['params']

def timezone_from_coordinates(lat, long, tzw):
    """ Получение таймзоны из координат (требует tzwhere) """
    return tzw.tzNameAt(lat, long)


# YANDEX GEOCODING
def yandex_geocode(adresses, api_key, address_column='fixed_address'):
    """ Геокодирование Яндекса """
    adresses = []
    coords = []

    for adr in address[address_column].values:
        api_adr = ' '.join(adr.split()).replace(' ', '+') # строка адреса в формате запроса
        r = requests.get('https://geocode-maps.yandex.ru/1.x/?apikey={}&format=json&geocode={}'.format(
                            API_KEY, api_adr))
        j = r.json()
        try:
            adresses.append(j['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['metaDataProperty'][
            'GeocoderMetaData']['Address'])
        except:
            adresses.append('NO ADRESS')

        try:
            coords.append(r.json()['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point'
                                  ]['pos'].split(' '))
        except:
            coords.append(['0','0'])

    return addresses, coords

def formatting_adresses(adresses, coords, filename='fixed.csv'):
    """ Форматирование ответов Яндекса """
    formatted_addresses = []
    lat = []
    lon = []
    locality = []

    for n in range(len(adresses)):
        if adresses[n] == 'NO ADRESS':
            lat.append(0)
            lon.append(0)
            formatted_addresses.append(',,')
            locality.append(',,')
        else:
            lat.append(float(coords[n][1]))
            lon.append(float(coords[n][0]))
            loc = ',,'
            try:
                for i in adresses[n]['Components']:
                    if i['kind'] == 'locality':
                         loc = ',,' + i['name']
                locality.append(loc)
            except:
                locality.apppend(loc)
            try:
                addr_form = adresses[n]['formatted'].split(', ')
                if len(addr_form)==1:
                    new_addr = ', '.join(['', ''] + addr_form)
                elif len(addr_form)==2:
                    new_addr = ', '.join([''] + [addr_form[1]] + [addr_form[0]])
                else:
                    new_addr = ', '.join(addr_form[1:] + [addr_form[0]])
                formatted_addresses.append(new_addr)
            except:
                formatted_addresses.append('')

    coordinates = pd.DataFrame([x.address.values, lat, lon, locality]).T
    coordinates.columns = ['address', 'lat', 'long', 'address_rus']
    coordinates.to_csv(filename, index=False)
