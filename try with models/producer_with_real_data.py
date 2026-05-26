from sklearn.datasets import fetch_california_housing

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

data = fetch_california_housing()
x_all = data.data
y_all = data.target

# Перемешиваем индексы
indices = np.random.permutation(len(x_all))
x_all_shuffled = x_all[indices]
y_all_shuffled = y_all[indices]

def generate_linear_data(i):
    return {'x': x_all_shuffled[i].tolist(), 'y': float(y_all_shuffled[i]), 'timestamp': time.time()}


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


