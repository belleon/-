# producer_aapl.py
import yfinance as yf
import numpy as np
import json
import time
from confluent_kafka import Producer

# Конфигурация Kafka
conf = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'aapl-producer'
}
producer = Producer(conf)
topic = 'linear-regression-data'

def fetch_aapl_data(period='3y'):
    """Загружает исторические данные AAPL за указанный период."""
    ticker = yf.Ticker('AAPL')
    df = ticker.history(period=period)
    # Используем цены закрытия
    prices = df['Close'].values
    print(f"Загружено {len(prices)} дней данных по AAPL")
    return prices

def create_sequences(data, window_size=8):
    """Создаёт последовательности для онлайн-обучения.
    Каждый вход x — список из window_size предыдущих цен,
    y — следующая цена."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size].tolist())
        y.append(float(data[i+window_size]))
    return X, y

def delivery_report(err, msg):
    if err:
        print(f'Ошибка доставки сообщения: {err}')
    else:
        print(f'Сообщение доставлено в {msg.topic()} [{msg.partition()}]')

# Загружаем цены AAPL
prices = fetch_aapl_data(period='7y')  # 2 года дневных данных
window_size = 8
X_data, y_data = create_sequences(prices, window_size)

print(f"Сформировано {len(X_data)} окон для обучения")

# Отправляем данные в Kafka
try:
    for i, (x, y) in enumerate(zip(X_data, y_data)):
        message = {
            'x': x,          # список из 8 цен закрытия
            'y': y,          # следующая цена закрытия
            'timestamp': time.time()
        }
        producer.produce(
            topic,
            key=str(i),
            value=json.dumps(message),
            callback=delivery_report
        )
        producer.poll(0)
        time.sleep(0.01)  # небольшая задержка для эмуляции потока
except KeyboardInterrupt:
    print("Producer остановлен")
finally:
    producer.flush()
    print("Все сообщения отправлены")