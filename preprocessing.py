import pandas as pd
import numpy as np

def dummies_from_file(df, file_name):
    _df = df.copy()
    with open(file_name) as f:
         dummies = json.load(f)


    for column, values in dummies.items():
        for value in values:
            _df[f'{column}_{value}'] = (_df[column].astype(str) == value).astype(int)
        _df = _df.drop(column, axis=1)
    return _df
    

def norm_from_file(df, file_name):
    _df = df.copy()
    with open(file_name) as f:
         transformations = json.load(f)


    for column, t in transformations.items():
        if column == 'n_bike':
            continue
        elif t['function'] == 'standardization':
            _df[column] = (_df[column] - t['mean']) / t['std']
        elif t['function'] == 'normalization':
            _df[column] = _df[column] / t['maxi']

    return _df

def preprocess(values, return_df=False):
    _df = pd.DataFrame(values, columns = [
     'date',    # Date (dd/mm/yyyy)
     'hour',    # Hour (int)
     'temp',    # Temperature (°C)
     'hum',     # Humidity (%)
     'wind',    # Wind speed (m/s)
     'visb',    # Visibility (10m)
     'dew',     # Dew point temperature (°C)
     'solar',   # Solar Radiation (MJ/m2)
     'rain',    # Rainfall(mm)
     'snow',    # Snowfall (cm)
     'season',  # Seasons ({"Winter", "Autumn", "Spring", "Summer"})
     'holiday', # Holiday ({"Holiday", "No Holiday"})
    ])


    # set boolean string to boolean
    _df['holiday'] = (_df['holiday'] == "Holiday").astype(int)

    # get datetime and its arguments
    _df['date'] = pd.to_datetime(_df['date'], format="%d/%m/%Y")
    _df['year'] = _df['date'].dt.year
    _df['month'] = _df['date'].dt.month_name()
    _df['day'] = _df['date'].dt.day
    _df['week_day'] = _df['date'].dt.day_name()
    _df['working_day'] = (_df['date'].dt.dayofweek < 5).astype(np.int)
    _df = _df.drop('date', axis=1)
    # meteorological arguments
    _df['wind'] = np.log(_df['wind'].astype(float) + 1)
    _df['solar'] = np.sqrt(_df['solar'].astype(float))
    _df['rain'] = _df.rolling(2, min_periods=1)['rain'].mean()
    _df['dryness'] = 1 / (_df.rain + 1)
    _df = _df.drop('rain', axis=1)
    _df['snow'] = _df.rolling(8, min_periods=1)['snow'].mean()
    _df['snowing'] = (_df.snow > 0).astype(np.int)
    _df = _df.drop('snow', axis=1)
    _df['invisb'] = _df.visb.max() - _df.visb
    _df = _df.drop('visb', axis=1)

    # complete preprocessing
    _df = dummies_from_file(_df, 'dummies.json')
    _df = norm_from_file(_df, 'transformations.json')
    if not return_df:
        _df = _df.values
    return _df
