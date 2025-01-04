import time
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from tabulate import tabulate

# Размер блока потоков для GPU
THREADS_PER_BLOCK = 16

# Количество повторений для усреднения времени
NUM_ITERATIONS = 70

# Маленькая константа для предотвращения деления на ноль
SMALL_EPSILON = 1e-7

# CUDA-функция для суммирования элементов массива на GPU
@cuda.jit
def gpu_sum_elements(vector, result):
    thread_idx = cuda.threadIdx.x  # Индекс потока в блоке
    block_idx = cuda.blockIdx.x   # Индекс блока
    global_idx = thread_idx + block_idx * THREADS_PER_BLOCK  # Глобальный индекс элемента

    if global_idx < vector.shape[0]:
        cuda.atomic.add(result, 0, vector[global_idx])

# Функция для проведения вычислений на CPU и GPU
def perform_benchmark():
    results = []
    min_vector_size = 50000
    max_vector_size = 1000000
    step_size = 50000

    for current_size in range(min_vector_size, max_vector_size + 1, step_size):
        total_cpu_time = 0.0
        total_gpu_time = 0.0

        for _ in range(NUM_ITERATIONS):
            vector = np.ones(current_size, dtype=np.float32)
            result_gpu = np.zeros(1, dtype=np.float32)

            # Копируем данные на GPU
            device_vector = cuda.to_device(vector)
            device_result = cuda.to_device(result_gpu)

            # Замер времени для GPU
            start_gpu = time.time()
            gpu_sum_elements[int((current_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), THREADS_PER_BLOCK](device_vector, device_result)
            total_gpu_time += time.time() - start_gpu

            # Копируем результат с GPU обратно на CPU
            result_gpu = device_result.copy_to_host()

            # Замер времени для CPU
            start_cpu = time.time()
            cpu_result = np.sum(vector)
            total_cpu_time += time.time() - start_cpu

        # Сохраняем результаты для текущего размера вектора
        results.append([current_size, total_cpu_time / NUM_ITERATIONS, total_gpu_time / NUM_ITERATIONS])

    # Выводим результаты в таблице
    print(tabulate(results, headers=["Vector Size", "CPU Time (s)", "GPU Time (s)"]))
    return results

# Функция для построения графиков
def generate_charts(vector_sizes, cpu_times, gpu_times, speedups):
    # График для времени выполнения на CPU
    plt.figure()
    plt.title("CPU")
    plt.plot(vector_sizes, cpu_times)
    plt.xlabel("размер вектора")
    plt.ylabel("время, мс")
    plt.grid()
    plt.legend()

    # График для времени выполнения на GPU
    plt.figure()
    plt.title("GPU")
    plt.plot(vector_sizes, gpu_times)
    plt.xlabel("размер вектора")
    plt.ylabel("время, мс")
    plt.grid()
    plt.legend()

    # График для ускорения
    plt.figure()
    plt.title("Ускорение")
    plt.plot(vector_sizes, speedups)
    plt.xlabel("размер вектора")
    plt.grid()
    plt.legend()

    # Показываем все графики
    plt.show()

# Основная программа
if __name__ == "__main__":
    # Выполнение измерений
    benchmark_results = perform_benchmark()

    # Извлечение данных для графиков
    sizes = [row[0] for row in benchmark_results]
    cpu_times = [row[1] for row in benchmark_results]
    gpu_times = [row[2] for row in benchmark_results]
    speedups = [cpu_times[i] / (gpu_times[i] if gpu_times[i] > SMALL_EPSILON else SMALL_EPSILON) for i in range(len(cpu_times))]

    # Построение графиков
    generate_charts(sizes, cpu_times, gpu_times, speedups)

