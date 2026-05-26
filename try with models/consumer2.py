# consumer_lr_with_mse.py
import json
import numpy as np
from confluent_kafka import Consumer, KafkaException
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
import psutil
import os
import matplotlib.pyplot as plt
import gc

conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'linear-regression-group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(conf)
topic = 'linear-regression-data'
consumer.subscribe([topic])

model = LinearRegression(fit_intercept=False)
X_buffer = []
y_buffer = []

# Статистика
training_times = []
memory_usages = []
mse_scores = []          # MSE на всех накопленных данных после обучения
training_count = 0
total_training_time = 0.0

process = psutil.Process(os.getpid())

def compute_mse(model, X, y):
    """Вычисляет MSE модели на данных X, y"""
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred)

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            raise KafkaException(msg.error())

        data = json.loads(msg.value().decode('utf-8'))
        x = data['x']
        y = data['y']

        # Замер памяти до (после GC для чистоты)
        gc.collect()
        mem_before = process.memory_info().rss / (1024 * 1024)

        # Добавляем данные
        X_buffer.append(x)
        y_buffer.append(y)

        # Обучение
        train_start = time.time()
        model.fit(X_buffer, y_buffer)
        train_end = time.time()
        train_time = train_end - train_start

        # Вычисляем MSE на всех данных
        X_arr = np.array(X_buffer)
        y_arr = np.array(y_buffer)
        mse = compute_mse(model, X_arr, y_arr)

        # Память после обучения (и после возможного удаления временных объектов)
        gc.collect()
        mem_after = process.memory_info().rss / (1024 * 1024)

        # Сохраняем статистику
        training_times.append(train_time)
        memory_usages.append(mem_after)
        mse_scores.append(mse)
        total_training_time += train_time
        training_count += 1

        print(f"Batch #{training_count} | MSE={mse:.6f} | Time={train_time:.4f}s | Mem={mem_after:.2f}MB")

except KeyboardInterrupt:
    print("\nStopping...")
    if training_count == 0:
        consumer.close()
        exit()

    # Финальная статистика
    avg_time = total_training_time / training_count
    avg_mem = np.mean(memory_usages)
    print(f"\nTotal iterations: {training_count}")
    print(f"Avg training time: {avg_time:.6f}s")
    print(f"Avg memory: {avg_mem:.2f} MB")
    print(f"Final MSE: {mse_scores[-1]:.6f}")

    # Графики
    iterations = range(1, training_count + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('LinearRegression (growing buffer) – Performance')

    axes[0].plot(iterations, memory_usages, 'b-', linewidth=1)
    axes[0].set_ylabel('RSS, MB')
    axes[0].set_title('Memory usage')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(iterations, training_times, 'r-', linewidth=1)
    axes[1].set_ylabel('Time, sec')
    axes[1].set_title('Training time per batch')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    axes[2].plot(iterations, mse_scores, 'g-', linewidth=1)
    axes[2].set_xlabel('Iteration (batch)')
    axes[2].set_ylabel('MSE')
    axes[2].set_title('Mean Squared Error on accumulated data')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('lr_mse_mem_time.png', dpi=150)
    plt.show()

    consumer.close()