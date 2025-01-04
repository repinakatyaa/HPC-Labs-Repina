from numba import cuda
import numpy as np
import cv2
import math
import time

# Добавляем шум "соль и перец"
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """Добавляет шум 'соль и перец' к изображению."""
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Добавляем белые пиксели (соль)
    num_salt = np.ceil(salt_prob * total_pixels).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # Добавляем черные пиксели (перец)
    num_pepper = np.ceil(pepper_prob * total_pixels).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

# CUDA-ядро для медианного фильтра
@cuda.jit
def median_filter_gpu(input_image, output_image, width, height):
    """CUDA-ядро для применения медианного фильтра 3x3."""
    x, y = cuda.grid(2)

    if x < 1 or y < 1 or x >= width - 1 or y >= height - 1:
        return  # Пропускаем границы изображения

    # Создаем массив для хранения 9 пикселей окна 3x3
    window = cuda.local.array(9, dtype=np.float32)
    idx = 0
    for j in range(-1, 2):
        for i in range(-1, 2):
            window[idx] = input_image[y + j, x + i]
            idx += 1

    # Применяем сортировку пузырьком
    for i in range(9):
        for j in range(i + 1, 9):
            if window[i] > window[j]:
                temp = window[i]
                window[i] = window[j]
                window[j] = temp

    # Устанавливаем медианное значение
    output_image[y, x] = window[4]

# Основная функция для фильтрации
def apply_median_filter(input_image):
    """Применяет медианный фильтр с использованием CUDA."""
    height, width = input_image.shape

    # Выделяем память для выходного изображения
    output_image = np.zeros((height, width), dtype=np.float32)

    # Копируем данные на устройство
    d_input = cuda.to_device(input_image)
    d_output = cuda.to_device(output_image)

    # Определяем параметры блоков и сетки
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Запускаем CUDA-ядро
    median_filter_gpu[blocks_per_grid, threads_per_block](d_input, d_output, width, height)

    # Копируем результат на хост
    output_image = d_output.copy_to_host()

    return output_image

# Сохраняем изображение
def save_image_as_bmp(image, filename):
    """Сохраняет изображение в формате BMP."""
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, image)


if __name__ == "__main__":
    # Загружаем изображение от пользователя
    input_jpg_path = input("Введите путь к изображению в формате JPG: ")
    output_bmp_path = "filtered_image.bmp"

    src_image = cv2.imread(input_jpg_path, cv2.IMREAD_GRAYSCALE)
    if src_image is None:
        print("Ошибка: невозможно загрузить изображение. Проверьте путь.")
    else:
        # Добавляем шум "соль и перец"
        noisy_image = add_salt_and_pepper_noise(src_image, salt_prob=0.05, pepper_prob=0.05)
        save_image_as_bmp(noisy_image, "noisy_image.bmp")
        print("Зашумленное изображение сохранено как 'noisy_image.bmp'.")

        # Применяем медианный фильтр
        start_time = time.time()
        filtered_image = apply_median_filter(noisy_image)
        elapsed_time = time.time() - start_time

        # Сохраняем результат
        save_image_as_bmp(filtered_image, output_bmp_path)
        print(f"Обработанное изображение сохранено как '{output_bmp_path}'.")
        print(f"Время обработки на GPU: {elapsed_time:.4f} секунд.")
