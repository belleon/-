# consumer_adaptive_lambda.py
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from confluent_kafka import Consumer, KafkaException
import psutil
import os
import gc
from collections import deque

# ------------------------------------------------------------
# 1. Собственная реализация RLS с возможностью менять lambda на лету
# ------------------------------------------------------------
class AdaptiveRLS:
    def __init__(self, n_features, lambda_init=0.99, delta=1000.0):
        self.lambda_ = lambda_init
        self.theta = np.zeros(n_features)
        self.P = np.eye(n_features) * delta

    def predict(self, x):
        return np.dot(self.theta, x)

    def update(self, x, y):
        x = np.asarray(x)
        Px = self.P @ x
        denom = self.lambda_ + np.dot(x, Px)
        K = Px / denom
        y_pred = np.dot(self.theta, x)
        e = y - y_pred
        self.theta = self.theta + K * e
        self.P = (self.P - np.outer(K, Px)) / self.lambda_
        return e

    def set_lambda(self, new_lambda):
        self.lambda_ = new_lambda

# ------------------------------------------------------------
# 2. Конфигурация Kafka
# ------------------------------------------------------------
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'adaptive-lambda-group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(conf)
topic = 'linear-regression-data'
consumer.subscribe([topic])

# ------------------------------------------------------------
# 3. Параметры адаптации lambda
# ------------------------------------------------------------
LAMBDA_INIT = 0.99
LAMBDA_MIN = 0.80
LAMBDA_MAX = 0.99
ADAPT_SPEED = 0.01          # шаг изменения lambda
ERROR_THRESHOLD = 1.2        # порог отношения raw_mse / sliding_mse
SLIDING_WINDOW = 100         # окно для скользящей MSE
START_ADAPT_AFTER = 50       # после скольких итераций начинаем адаптацию
RETURN_SPEED = 0.002         # скорость возврата к LAMBDA_INIT (если ошибка мала)

model = AdaptiveRLS(n_features=8, lambda_init=LAMBDA_INIT)

# Скользящая MSE
mse_window = deque(maxlen=SLIDING_WINDOW)
iteration = 0

# Статистика
training_times = []
memory_usages = []
mse_raw_history = []
mse_sliding_history = []
lambda_history = []

process = psutil.Process(os.getpid())

def adapt_lambda(raw_mse, sliding_mse, current_lambda):
    """Возвращает новое значение lambda на основе отношения ошибок."""
    if sliding_mse < 1e-12:
        return current_lambda
    ratio = raw_mse / sliding_mse

    if ratio > ERROR_THRESHOLD:
        # Ошибка выросла – уменьшаем lambda (ускоряем забывание)
        new_lambda = max(LAMBDA_MIN, current_lambda - ADAPT_SPEED)
        if new_lambda != current_lambda:
            print(f"  lambda уменьшено: {current_lambda:.4f} -> {new_lambda:.4f} (ratio={ratio:.2f})")
        return new_lambda
    elif current_lambda < LAMBDA_INIT:
        # Медленно возвращаемся к базовому значению
        new_lambda = min(LAMBDA_INIT, current_lambda + RETURN_SPEED)
        if new_lambda != current_lambda:
            print(f"  lambda увеличено: {current_lambda:.4f} -> {new_lambda:.4f}")
        return new_lambda
    else:
        return current_lambda

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            raise KafkaException(msg.error())

        data = json.loads(msg.value().decode('utf-8'))
        x = np.array(data['x'])
        y = data['y']

        gc.collect()
        mem_before = process.memory_info().rss / (1024 * 1024)
        start_time = time.perf_counter()

        # Шаг RLS
        e = model.update(x, y)
        squared_error = e * e

        # Обновление скользящего MSE
        mse_window.append(squared_error)
        sliding_mse = np.mean(mse_window) if mse_window else squared_error

        # Адаптация lambda (после накопления начальных данных)
        if iteration >= START_ADAPT_AFTER:
            new_lambda = adapt_lambda(squared_error, sliding_mse, model.lambda_)
            model.set_lambda(new_lambda)

        training_time = time.perf_counter() - start_time
        gc.collect()
        mem_after = process.memory_info().rss / (1024 * 1024)

        # Сохраняем метрики
        iteration += 1
        training_times.append(training_time)
        memory_usages.append(mem_after)
        mse_raw_history.append(squared_error)
        mse_sliding_history.append(sliding_mse)
        lambda_history.append(model.lambda_)

        if iteration % 50 == 0:
            print(f"Iter {iteration:5d}: lambda={model.lambda_:.4f}, "
                  f"sliding MSE={sliding_mse:.6f}, time={training_time:.6f}s, mem={mem_after:.2f}MB")

except KeyboardInterrupt:
    print("\nОстановка consumer...")
    if iteration == 0:
        consumer.close()
        exit()

    # --------------------------------------------------------
    # 4. Построение графиков
    # --------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("RLS с динамическим λ (адаптация по ошибке)", fontsize=14)

    axes[0].plot(range(1, iteration+1), memory_usages, 'b-', linewidth=1)
    axes[0].set_ylabel("RSS, МБ")
    axes[0].set_title("Память")
    axes[0].grid(True)

    axes[1].plot(range(1, iteration+1), training_times, 'r-', linewidth=1)
    axes[1].set_ylabel("Время, сек")
    axes[1].set_title("Время на одно наблюдение")
    axes[1].grid(True)

    axes[2].plot(range(1, iteration+1), mse_sliding_history, 'g-', linewidth=0.8, label='скользящее MSE')
    axes[2].set_ylabel("MSE")
    axes[2].set_title("Динамика ошибки")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(range(1, iteration+1), lambda_history, 'm-', linewidth=1)
    axes[3].set_xlabel("Итерация")
    axes[3].set_ylabel("λ")
    axes[3].set_title("Фактор забывания (адаптивный)")
    axes[3].set_ylim([LAMBDA_MIN - 0.05, LAMBDA_MAX + 0.05])
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig("adaptive_lambda_analysis.png", dpi=150)
    plt.show()

    # Финальная статистика
    print("\n" + "="*60)
    print("ИТОГОВАЯ СТАТИСТИКА (RLS с динамическим λ)")
    print("="*60)
    print(f"Всего итераций: {iteration}")
    print(f"Среднее время: {np.mean(training_times):.6f} сек")
    print(f"Средняя память: {np.mean(memory_usages):.2f} МБ")
    print(f"Начальный λ: {LAMBDA_INIT}")
    print(f"Средний λ (после {START_ADAPT_AFTER} итераций): {np.mean(lambda_history[START_ADAPT_AFTER:]):.4f}")
    print(f"Минимальный λ: {min(lambda_history):.4f}")
    print(f"Максимальный λ: {max(lambda_history):.4f}")
    print(f"Среднее MSE (последние 200): {np.mean(mse_sliding_history[-200:]):.6f}")

finally:
    consumer.close()