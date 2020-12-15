# Python_For_Data_Analysis_Seoul_Bike

## How to use the API

### Feature matrix creation

On the client side, it will be ask to create a matrix of features to predict. Each row is a list of ordered features as following :

|**Feature**| **Format**|
|:-:|:-:|
| Date | dd/mm/yyyy|
| Hour | int|
| Temperature | °C|
| Humidity | %|
| Wind speed | m/s|
| Visibility | 10m|
| Dew point temperature | °C|
| Solar Radiation | MJ/m2|
| Rainfal | mm|
| Snowfall | mm|
| Seasons | {"Winter", "Autumn", "Spring", "Summer"}|
| Holiday | {"Holiday", "No Holiday"}|

### Run the API

We can run this server from terminal running the `python deployed_model.py` or `python3 deployed_model.py` according to your OS and python version.


### Request the API

From the **preprocessing.py** file I recreate the preprocessing function :
- set holiday feautre as boolean 
- extract time features
- trasnform meteorological arguments
- complete preprocessing with one hot encoding on categorical feautures and norm/standardization on numerical ones``

```python
import requests
from preprocessing import preprocess

X = None  # Matrix of raw features
url = 'http://localhost:5000/predict'  # API request url

def serialize(df):
    return [[value for value in row] for row in df.values]
    
# preprocess de feature matrix
X = serialize(preprocess(X))
# request the API
r = requests.post(url, json={'inputs': serialize(X)})

# get the predictions
prediction = r.json()
```
## Feautures preprocessing



## Selected model

### Model comparison

### RandomForestRegressor
