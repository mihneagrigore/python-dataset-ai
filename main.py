import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

cities = ['London', 'New York', 'Tokyo', 'Paris', 'Berlin', 'Madrid', 'Rome', 'Beijing', 'Moscow', 'Sydney']

weather_conditions = ['Clear', 'Cloudy', 'Rainy', 'Stormy', 'Snowy']
wind_speeds = [1, 5, 10, 15, 20, 25, 30]
temperature_range = [-10, 40]

def generate_weather_data(num_samples=500, missing_prob=0.1, outlier_prob=0.05):
    data = []
    for _ in range(num_samples):
        city = random.choice(cities)
        condition = random.choice(weather_conditions)
        temperature = random.randint(temperature_range[0], temperature_range[1])

        if random.random() < outlier_prob:
            temperature = random.choice([-100, 100, 200])

        wind = random.choice(wind_speeds)
        if random.random() < outlier_prob:
            wind = random.choice([50, 100])

        humidity = random.randint(20, 100)
        if random.random() < outlier_prob:
            humidity = random.choice([5, 110])

        pressure = random.randint(980, 1025)
        if random.random() < outlier_prob:
            pressure = random.choice([950, 1050])

        visibility = random.choice(['Good', 'Moderate', 'Poor'])
        date_time = random.choice(pd.date_range(start='2000-01-01', end='2024-12-31'))

        if random.random() < missing_prob:
            temperature = np.nan
        if random.random() < missing_prob:
            wind = np.nan
        if random.random() < missing_prob:
            humidity = np.nan
        if random.random() < missing_prob:
            pressure = np.nan

        data.append({
            'City': city,
            'Condition': condition,
            'Temperature': temperature,
            'Wind': wind,
            'Humidity': humidity,
            'Pressure': pressure,
            'Visibility': visibility,
            'DateTime': date_time
        })

    return pd.DataFrame(data)

train_data = generate_weather_data(500, missing_prob=0.1)
test_data = generate_weather_data(200, missing_prob=0.1)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)