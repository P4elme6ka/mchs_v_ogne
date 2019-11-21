import os
import sys
import pandas
import xarray
import requests
from datetime import datetime, timedelta
import pickle
import numpy as np
from tqdm import tqdm as tqdm

DATASETS_PATH = os.environ.get('DATASETS_PATH', '../data/')

ncep_data = []
year = 2019
for var in ('air', 'uwnd', 'rhum'):
    dataset_filename = '{}/ncep/{}.{}.nc'.format(DATASETS_PATH, var, year)
    ncep_data.append(xarray.open_dataset(dataset_filename))
ncep_data = xarray.merge(ncep_data)


def extract_features(row):
    point = ncep_data.sel(
        lon=row['longitude'],
        lat=row['latitude'],
        level=1000,
        method='nearest',
    )

    p1w = point.rolling(time=1).mean()
    p2w = point.rolling(time=3).mean()
    p3w = point.rolling(time=5).mean()

    date = datetime.strptime(row['date'], '%Y-%m-%d').date()

    date1 = date + timedelta(days=1)
    date2 = date + timedelta(days=2)
    date3 = date + timedelta(days=3)

    date4 = date - timedelta(days=1)
    date5 = date - timedelta(days=2)
    date6 = date - timedelta(days=3)

    v = point.sel(time=datetime.strftime(date, '%Y-%m-%d'))
    v1w = p1w.sel(time=datetime.strftime(date, '%Y-%m-%d'))
    v2w = p2w.sel(time=datetime.strftime(date, '%Y-%m-%d'))
    v3w = p3w.sel(time=datetime.strftime(date, '%Y-%m-%d'))

    v1w1 = p1w.sel(time=datetime.strftime(date1, '%Y-%m-%d'))
    v2w1 = p2w.sel(time=datetime.strftime(date1, '%Y-%m-%d'))
    v3w1 = p3w.sel(time=datetime.strftime(date1, '%Y-%m-%d'))

    v1w2 = p1w.sel(time=datetime.strftime(date2, '%Y-%m-%d'))
    v2w2 = p2w.sel(time=datetime.strftime(date2, '%Y-%m-%d'))
    v3w2 = p3w.sel(time=datetime.strftime(date2, '%Y-%m-%d'))

    v1w3 = p1w.sel(time=datetime.strftime(date3, '%Y-%m-%d'))
    v2w3 = p2w.sel(time=datetime.strftime(date3, '%Y-%m-%d'))
    v3w3 = p3w.sel(time=datetime.strftime(date3, '%Y-%m-%d'))

    # ------------------------------------------------------------ -

    v1w4 = p1w.sel(time=datetime.strftime(date4, '%Y-%m-%d'))
    v2w4 = p2w.sel(time=datetime.strftime(date4, '%Y-%m-%d'))
    v3w4 = p3w.sel(time=datetime.strftime(date4, '%Y-%m-%d'))

    v1w5 = p1w.sel(time=datetime.strftime(date5, '%Y-%m-%d'))
    v2w5 = p2w.sel(time=datetime.strftime(date5, '%Y-%m-%d'))
    v3w5 = p3w.sel(time=datetime.strftime(date5, '%Y-%m-%d'))

    v1w6 = p1w.sel(time=datetime.strftime(date6, '%Y-%m-%d'))
    v2w6 = p2w.sel(time=datetime.strftime(date6, '%Y-%m-%d'))
    v3w6 = p3w.sel(time=datetime.strftime(date6, '%Y-%m-%d'))

    return {
        'fire_id': row['fire_id'],
        'fire_type': '',
        'fire_type_name': '',
        'date': row['date'],
        'temperature': v.air.values.item(0),
        'humidity': v.rhum.values.item(0),
        'uwind': v.uwnd.values.item(0),
        't1w': v1w.air.values.item(0),
        't2w': v2w.air.values.item(0),
        't3w': v3w.air.values.item(0),
        't1w1': v1w1.air.values.item(0),
        't2w1': v2w1.air.values.item(0),
        't3w1': v3w1.air.values.item(0),
        't1w2': v1w2.air.values.item(0),
        't2w2': v2w2.air.values.item(0),
        't3w2': v3w2.air.values.item(0),
        'h1w': v1w.rhum.values.item(0),
        'h2w': v2w.rhum.values.item(0),
        'h3w': v3w.rhum.values.item(0),
        'h1w1': v1w.rhum.values.item(0),
        'h2w1': v2w.rhum.values.item(0),
        'h3w1': v3w.rhum.values.item(0),
        'h1w2': v1w.rhum.values.item(0),
        'h2w2': v2w.rhum.values.item(0),
        'h3w2': v3w.rhum.values.item(0),
        'h1w3': v1w.rhum.values.item(0),
        'h2w3': v2w.rhum.values.item(0),
        'h3w3': v3w.rhum.values.item(0),
        'h1w4': v1w.rhum.values.item(0),
        'h2w4': v2w.rhum.values.item(0),
        'h3w4': v3w.rhum.values.item(0),
        'h1w5': v1w.rhum.values.item(0),
        'h2w5': v2w.rhum.values.item(0),
        'h3w5': v3w.rhum.values.item(0),
        'h1w6': v1w.rhum.values.item(0),
        'h2w6': v2w.rhum.values.item(0),
        'h3w6': v3w.rhum.values.item(0),
    }


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    df_points = pandas.read_csv(input_csv)

    df_features = []
    for i, row in tqdm(df_points.iterrows(), total=df_points.shape[0]):
        features = extract_features(row)
        df_features.append(features)
    df_features = pandas.DataFrame(df_features)
    df_features.set_index('fire_id', inplace=True)

    with open('model.pickle', 'rb') as fin:
        fire_classifier = pickle.load(fin)

    X = df_features.iloc[:, 3:].fillna(0)

    df_predictions = pandas.DataFrame(
        fire_classifier.predict_proba(X),
        index=df_features.index,
        columns=[
            'fire_{}_prob'.format(class_id)
            for class_id in fire_classifier.classes_
        ],
    )

    df_predictions.to_csv(output_csv)
