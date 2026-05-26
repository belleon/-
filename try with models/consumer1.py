import padasip as pa
import json
import numpy as np
from confluent_kafka import Consumer, KafkaException
import time
import psutil
import os
import matplotlib.pyplot as plt
from collections import deque

# Конфигурация Kafka consumer
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'linear-regression-group',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(conf)
topic = 'linear-regression-data'
consumer.subscribe([topic])

# Инициализация модели RLS
rls = pa.filters.FilterRLS(n=8, mu=0.99, w='random')

# Статистика времени
total_training_time = 0.0
training_count = 0
training_times = []

# Статистика памяти
process = psutil.Process(os.getpid())
memory_usages = []  # RSS в МБ

# Статистика ошибок (MSE)
mse_values = []          # квадрат ошибки на каждом шаге
sliding_window_size = 100
mse_sliding = deque(maxlen=sliding_window_size)  # для скользящего среднего
sliding_mse_history = [] # сохраняем скользящее MSE для графика

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            raise KafkaException(msg.error())

        data = json.loads(msg.value().decode('utf-8'))
        x = np.array(data['x'])
        y = data['y']

        # ----- ПРЕДСКАЗАНИЕ ДО ОБУЧЕНИЯ (prequential) -----
        y_pred = rls.predict(x)  # предсказание на основе текущих весов
        error = y - y_pred
        squared_error = error ** 2
        mse_values.append(squared_error)
        mse_sliding.append(squared_error)
        current_sliding_mse = np.mean(mse_sliding) if len(mse_sliding) > 0 else squared_error
        sliding_mse_history.append(current_sliding_mse)
        # Замер памяти до обучения
        mem_before = process.memory_info().rss / (1024 * 1024)

        # Обучение (адаптация) на истинном значении
        train_start = time.time()
        rls.adapt(np.array([y]), x)   # внимание: x уже numpy array
        train_end = time.time()
        training_time = train_end - train_start

        # Замер памяти после
        mem_after = process.memory_info().rss / (1024 * 1024)
        current_memory = mem_after

        total_training_time += training_time
        training_count += 1
        training_times.append(training_time)
        memory_usages.append(current_memory)

        # Вывод в консоль (каждые 50 итераций для сокращения вывода)
        if training_count % 50 == 0:
            print(f"Итерация #{training_count}: MSE={squared_error:.6f}, скользящее MSE={current_sliding_mse:.6f}, время={training_time:.6f}с, память={current_memory:.2f}МБ")

except KeyboardInterrupt:
    print("\nStopping consumer...")
    if training_count > 0:
        avg_memory = np.mean(memory_usages)
        min_memory = min(memory_usages)
        max_memory = max(memory_usages)
        avg_time = total_training_time / training_count
        final_mse = mse_values[-1] if mse_values else 0.0
        avg_mse = np.mean(mse_values[-1000:]) if len(mse_values) > 1000 else np.mean(mse_values)

        print(f"\n{'=' * 60}")
        print(f"ФИНАЛЬНАЯ СТАТИСТИКА")
        print(f"{'=' * 60}")
        print(f"Всего итераций: {training_count}")
        print(f"Общее время: {total_training_time:.4f} сек")
        print(f"Среднее время: {avg_time:.6f} сек")
        print(f"Память (сред/мин/макс): {avg_memory:.2f} / {min_memory:.2f} / {max_memory:.2f} МБ")
        print(f"Среднее MSE (последние 1000): {avg_mse:.6f}")
        print(f"Финальное MSE: {final_mse:.6f}")
        print(f"Финальные веса: {rls.w}")

        # ---------- ПОСТРОЕНИЕ ГРАФИКОВ ----------
        iterations = range(1, training_count + 1)
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle('Анализ производительности RLS-адаптации с MSE')

        # График памяти
        axes[0].plot(iterations, memory_usages, 'b-', linewidth=1.5)
        axes[0].set_xlabel('Итерация')
        axes[0].set_ylabel('RSS, МБ')
        axes[0].set_title('Изменение использования оперативной памяти')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # График времени обучения
        axes[1].plot(iterations, training_times, 'r-', linewidth=1.5)
        axes[1].set_xlabel('Итерация')
        axes[1].set_ylabel('Время, секунды')
        axes[1].set_title('Время обучения на каждой итерации')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # График MSE (скользящее окно)
        axes[2].plot(iterations, sliding_mse_history, 'g-', linewidth=1.0, alpha=0.8)
        axes[2].set_xlabel('Итерация')
        axes[2].set_ylabel('MSE (скользящее окно {})'.format(sliding_window_size))
        axes[2].set_title('Динамика среднеквадратичной ошибки (prequential)')
        axes[2].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('rls_mse_analysis.png', dpi=150)
        print("\nГрафик сохранён как 'rls_mse_analysis.png'")
        plt.show()

    else:
        print("Нет обработанных сообщений.")

finally:
    consumer.close()