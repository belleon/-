# producer.py
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

def generate_linear_data(slope=2.0, intercept=1.0, noise=0.5):
    """Генерация данных для линейной регрессии"""
    x = random.uniform(-10, 10)
    # y = mx + b + noise
    y = slope * x + intercept + random.uniform(-noise, noise)
    return {'x': x, 'y': y, 'timestamp': time.time()}

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