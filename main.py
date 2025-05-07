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

def fill_missing_values(data):
    """Impută valorile lipsă pentru coloanele numerice și categorice."""
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].fillna(data[column].mode()[0])
        else:
            data[column] = data[column].fillna(data[column].mean())
    return data

train_data = fill_missing_values(train_data)

test_data = fill_missing_values(test_data)

train_data.to_csv('train_data_filled.csv', index=False)
test_data.to_csv('test_data_filled.csv', index=False)

print("Datele de antrenament au fost generate cu succes!")
print("Au fost salvate fisierele csv")

def plot_distributions(data):
    numerical_columns = ['Temperature', 'Wind', 'Humidity', 'Pressure']
    for col in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], kde=True, bins=15)
        plt.title(f'Histogram for {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'histogram_{col}.png')
        plt.close()

    categorical_columns = ['City', 'Condition', 'Visibility']
    for col in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=data[col])
        plt.title(f'Countplot for {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.savefig(f'countplot_{col}.png')
        plt.close()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=data['City'], y=data['Temperature'], estimator=np.mean)
    plt.title('Barplot for City vs Temperature')
    plt.xlabel('City')
    plt.ylabel('Average Temperature')
    plt.savefig('barplot_City_vs_Temperature.png')
    plt.close()

plot_distributions(train_data)
print("Graficele au fost generate cu succes!")

numerical_summary = train_data.describe()

categorical_summary = train_data.describe(include=['object'])

with open('describe.txt', 'w') as f:
    f.write("Statistici descriptive pentru variabile numerice:\n")
    f.write(str(numerical_summary))
    f.write("\n\n")

    f.write("Statistici descriptive pentru variabile categorice:\n")
    f.write(str(categorical_summary))

print("Fisierul describe.txt a fost creat cu succes!")


def detect_outliers(data):
    numerical_columns = ['Temperature', 'Wind', 'Humidity', 'Pressure']

    for col in numerical_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"\n{col} - IQR Method\n")
        print(f"Q1 (25th percentile): {Q1}\n")
        print(f"Q3 (75th percentile): {Q3}\n")
        print(f"IQR: {IQR}\n")
        print(f"Lower Bound: {lower_bound}\n")
        print(f"Upper Bound: {upper_bound}\n")

    for col in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[col])
        plt.title(f'Boxplot for {col} with Outliers')
        plt.savefig(f'outliers_{col}.png')
        plt.close()

detect_outliers(train_data)
print("Graficele outliers au fost generate cu succes!")

def correlation_analysis(data):

    numerical_columns = ['Temperature', 'Wind', 'Humidity', 'Pressure']
    correlation_matrix = data[numerical_columns].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    with open('correlation_matrix.txt', 'w') as f:
        f.write("Correlation Matrix:\n")
        f.write(str(correlation_matrix))

    print("Matricea de corelație și heatmap-ul au fost salvate cu succes!")

correlation_analysis(train_data)

def plot_target_relationships(data, target_variable='Temperature'):
    numerical_columns = ['Wind', 'Humidity', 'Pressure']

    if data[target_variable].dtype in ['int64', 'float64']:
        for col in numerical_columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=data[col], y=data[target_variable])
            plt.title(f'Scatter plot of {col} vs {target_variable}')
            plt.xlabel(col)
            plt.ylabel(target_variable)
            plt.savefig(f'scatter_{col}_vs_{target_variable}.png')
            plt.close()

    else:
        for col in numerical_columns:
            plt.figure(figsize=(8, 6))
            sns.violinplot(x=data[target_variable], y=data[col])
            plt.title(f'Violin plot of {col} vs {target_variable}')
            plt.xlabel(target_variable)
            plt.ylabel(col)
            plt.savefig(f'violin_{col}_vs_{target_variable}.png')
            plt.close()

    print(f"Scatter/Violin plots for {target_variable} relationships have been saved.")

plot_target_relationships(train_data, target_variable='Temperature')



def train_and_evaluate_regression(train_file, test_file, target_variable):

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train = train_data.drop(columns=[target_variable])
    y_train = train_data[target_variable]

    X_test = test_data.drop(columns=[target_variable])
    y_test = test_data[target_variable]

    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2: {r2}")


    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title(f'Predictions vs Actual for {target_variable}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(f'predictions_vs_actual_{target_variable}.png')
    plt.close()

    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='green')
    plt.title(f'Residuals for {target_variable}')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig(f'residuals_{target_variable}.png')
    plt.close()

train_file = 'train_data_filled.csv'
test_file = 'test_data_filled.csv'
target_variable = 'Temperature'

train_and_evaluate_regression(train_file, test_file, target_variable)