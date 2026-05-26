# producer_household_power.py
import pandas as pd
import numpy as np
import json
import time
from confluent_kafka import Producer

# ---------------------- Конфигурация Kafka ----------------------
conf = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'household-power-producer'
}
producer = Producer(conf)
topic = 'linear-regression-data'

# ---------------------- Чтение и предобработка данных ----------------------
DATA_FILE = 'household_power_consumption.txt'  # путь к вашему файлу

def load_and_preprocess(file_path):
    """
    Загружает данные, выполняет предобработку, возвращает список наблюдений.
    Каждое наблюдение = (x, y), где x - список признаков, y - целевая переменная.
    """
    # Чтение CSV с разделителем ';', пропуски '?' -> NaN
    df = pd.read_csv(file_path, sep=';', na_values=['?'], low_memory=False)[-200000:]
    
    dtypes = {
        'Global_active_power': 'float32',
        'Voltage': 'float32',
        'Global_intensity': 'float32',
        'Sub_metering_1': 'float32',
        'Sub_metering_2': 'float32',
        'Sub_metering_3': 'float32'
    }
    usecols = ['Date', 'Time', 'Global_active_power', 'Voltage', 'Global_intensity',
               'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    df = pd.read_csv(file_path, sep=';', na_values=['?'],
                     dtype=dtypes, usecols=usecols, engine='c')
    
    # Быстрое преобразование даты+времени
    datetime_str = df['Date'] + ' ' + df['Time']
    dt = pd.to_datetime(datetime_str, format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df['hour'] = dt.dt.hour.astype('int8')
    df['day_of_week'] = dt.dt.dayofweek.astype('int8')
    df['month'] = dt.dt.month.astype('int8')
    
    # Целевая переменная и признаки
    X = df[['hour', 'day_of_week', 'month', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].values
    y = df['Global_active_power'].values
    
    # Удаление строк с NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    
    # Формирование списка наблюдений (быстрый list comprehension)
    observations = [(x.tolist(), float(yi)) for x, yi in zip(X, y)]
    print(f"Загружено {len(observations)} наблюдений.")
    return observations

def delivery_report(err, msg):
    if err:
        print(f'Ошибка доставки: {err}')
    else:
        print(f'Сообщение доставлено в {msg.topic()} [{msg.partition()}]')

def main():
    # Загрузка данных
    observations = load_and_preprocess(DATA_FILE)
    
    # Отправка в Kafka
    try:
        for i, (x, y) in enumerate(observations):
            message = {
                'x': x,
                'y': y,
                'timestamp': time.time()
            }
            producer.produce(
                topic,
                key=str(i),
                value=json.dumps(message),
                callback=delivery_report
            )
            producer.poll(0)
            # Задержка для имитации потока (можно убрать или уменьшить)
            time.sleep(0.05)
            if i % 100 == 0:
                print(f"Отправлено {i} наблюдений")
    except KeyboardInterrupt:
        print("Producer остановлен")
    finally:
        producer.flush()
        print("Все сообщения отправлены")

if __name__ == "__main__":
    main()