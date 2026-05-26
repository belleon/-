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

def generate_linear_data(w = [2,4,7],noise = 0.5):
    

    x =np.random.randint(1,100,(3))

    y = x@w + random.uniform(-noise, noise)
    print(x,y)
    return {'x': x.tolist(), 'y': y, 'timestamp': time.time()}

def delivery_report(err, msg):
    if err:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

# Основной цикл отправки данных
topic = 'linear-regression-data'
try:
    for i in range(1000):
        data = generate_linear_data()
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


