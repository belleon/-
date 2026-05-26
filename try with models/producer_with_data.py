from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import padasip as pa
import time
import random
import json
from confluent_kafka import Producer
import numpy as np

# Конфигурация Kafka
conf = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'linear-regression-producer'
}

producer = Producer(conf)

def generate_linear_data(i):

    data =pd.read_csv('train_energy_data.csv')
    building_type_map = {"Residential": 1, "Commercial": 2, "Industrial": 3}
    day_of_week_map = {"Weekday": 1, "Weekend": 0}

    data["Building Type"] = data["Building Type"].map(building_type_map)
    data["Day of Week"] = data["Day of Week"].map(day_of_week_map)


    X = data[["Square Footage", "Number of Occupants", "Appliances Used", "Average Temperature", "Building Type", "Day of Week"]]
    y = data["Energy Consumption"]
    print(np.array(X)[i].tolist(),float(np.array(y)[i]))

    return {'x': np.array(X)[i].tolist(), 'y': float(np.array(y)[i]), 'timestamp': time.time()}

def delivery_report(err, msg):
    if err:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

# Основной цикл отправки данных
topic = 'linear-regression-data'
try:
    for i in range(10000):
        data = generate_linear_data(i)
        producer.produce(
            topic,
            key=str(i),
            value=json.dumps(data),
            callback=delivery_report
        )
        producer.poll(0)
        time.sleep(0.1)
finally:
    producer.flush()
