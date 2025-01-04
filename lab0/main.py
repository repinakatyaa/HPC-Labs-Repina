import numpy as np
import time
from numba import cuda
import math
from matplotlib import pyplot as plt

size_of_matrix = 100

# Инициализация матриц для CPU
matrix_A_cpu = np.random.randint(0, 10, (size_of_matrix, size_of_matrix))
matrix_B_cpu = np.random.randint(0, 10, (size_of_matrix, size_of_matrix))
result_cpu = np.zeros((size_of_matrix, size_of_matrix), dtype=int)

# Инициализация матриц для GPU
matrix_A_gpu = cuda.to_device(matrix_A_cpu)
matrix_B_gpu = cuda.to_device(matrix_B_cpu)
result_gpu = cuda.device_array((len(matrix_A_cpu), len(matrix_B_cpu)))

# Функция для умножения матриц на CPU
def matrix_multiplication_cpu(A, B, C):
    for row in range(size_of_matrix):
        for col in range(size_of_matrix):
            total = 0
            for k in range(size_of_matrix):
                total += A[row, k] * B[k, col]
            C[row, col] = total

def cpu_computation():
    print("Запуск на CPU...")
    start_time = time.time()
    matrix_multiplication_cpu(matrix_A_cpu, matrix_B_cpu, result_cpu)
    cpu_duration = time.time() - start_time
    print(f"Время выполнения на CPU: {cpu_duration:.6f} секунд")
    return cpu_duration

@cuda.jit
def matrix_multiplication_gpu(A, B, C):
    for i in range(size_of_matrix):
        for j in range(size_of_matrix):
            total = 0
            for k in range(size_of_matrix):
                total += A[i, k] * B[k, j]
            C[i, j] = total

def gpu_computation():
    # Параметры для сетки и блоков
    block_size = (32, 32)
    grid_size_x = int(math.ceil(matrix_A_cpu.shape[0] / block_size[0]))
    grid_size_y = int(math.ceil(matrix_B_cpu.shape[1] / block_size[1]))
    grid_size = (grid_size_x, grid_size_y)
    
    print(f"Размер сетки = {grid_size}, размер блока = {block_size}")
    print("Работа CPU завершена.\n")

    print("Запуск на GPU...")
    start_time = time.time()
    matrix_multiplication_gpu[grid_size, block_size](matrix_A_gpu, matrix_B_gpu, result_gpu)
    gpu_duration = time.time() - start_time
    print(f"Время выполнения на GPU: {gpu_duration:.6f} секунд")
    print("Работа GPU завершена.\n")
    return gpu_duration

if __name__ == "__main__":
    cpu_time = cpu_computation()
    gpu_time = gpu_computation()

    # Данные для графиков 
    cpu_times = [0.200, 1.620, 13.970, 105.120, 956.780, 1989.240]
    gpu_times = [0.155, 0.163, 0.188, 0.215, 0.257, 0.271]
    matrix_sizes = [100, 200, 400, 800, 1600, 2000]

    # Построение графиков
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # График времени на CPU
    axs[0].plot(matrix_sizes, cpu_times)
    axs[0].set_title("Время выполнения на CPU")
    axs[0].set_xlabel("Размер матрицы")
    axs[0].set_ylabel("Время в секундах")
    axs[0].grid()

    # График времени на GPU
    axs[1].plot(matrix_sizes, gpu_times)
    axs[1].set_title("Время выполнения на GPU")
    axs[1].set_xlabel("Размер матрицы")
    axs[1].set_ylabel("Время в секундах")
    axs[1].grid()

    # График ускорения
    speedup = [cpu_times[i] / gpu_times[i] for i in range(len(cpu_times))]
    axs[2].plot(matrix_sizes, speedup)
    axs[2].set_title("Ускорение")
    axs[2].set_xlabel("Размер матрицы")
    axs[2].grid()

    plt.tight_layout()
    plt.savefig("performance_comparison.png")  # Сохранение графиков
