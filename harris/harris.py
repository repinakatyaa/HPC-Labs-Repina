import cv2
import numpy as np
import cupy as cp
import time

# Функция для вычисления углов методом Харриса
def compute_harris_corners(input_img, corner_thresh):
    # Преобразуем изображение в тип float32
    grayscale_img = np.float32(input_img)

    # Градиенты по направлениям X и Y
    gradient_x = cv2.Sobel(grayscale_img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grayscale_img, cv2.CV_64F, 0, 1, ksize=3)

    # Вычисление элементов матрицы
    grad_xx = gradient_x ** 2
    grad_yy = gradient_y ** 2
    grad_xy = gradient_x * gradient_y

    # Применение размытия Гаусса
    grad_xx = cv2.GaussianBlur(grad_xx, (5, 5), sigmaX=1)
    grad_yy = cv2.GaussianBlur(grad_yy, (5, 5), sigmaX=1)
    grad_xy = cv2.GaussianBlur(grad_xy, (5, 5), sigmaX=1)

    # Вычисление детерминанта и следа матрицы
    determinant = grad_xx * grad_yy - grad_xy ** 2
    trace = grad_xx + grad_yy
    response = determinant - 0.04 * (trace ** 2)

    # Определение углов
    corner_points = np.zeros_like(response)
    corner_points[response > corner_thresh] = 255

    return corner_points.astype(np.uint8)

def process_image(file_path, threshold_value, output_path):
    # Загружаем исходное изображение
    original_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if original_img is None:
        print("Ошибка: не удалось загрузить изображение. Проверьте путь.")
        return

    # Запуск обработки на CPU
    start_cpu = time.time()
    corners_cpu_result = compute_harris_corners(original_img, threshold_value)
    cpu_duration = time.time() - start_cpu

    # Рисуем углы на изображении (CPU)
    highlighted_cpu_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    highlighted_cpu_img[corners_cpu_result > 0] = [0, 0, 255]  # Углы отмечаем красным

    # Сохраняем результат CPU
    cpu_output_file = output_path + "_cpu.png"
    cv2.imwrite(cpu_output_file, highlighted_cpu_img)

    # Запуск обработки на GPU
    input_gpu = cp.asarray(original_img)
    start_gpu = time.time()
    gpu_corners_result = compute_harris_corners(cp.asnumpy(input_gpu), threshold_value)
    gpu_duration = time.time() - start_gpu

    # Рисуем углы на изображении (GPU)
    highlighted_gpu_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    highlighted_gpu_img[gpu_corners_result > 0] = [0, 0, 255]  # Углы отмечаем красным

    # Сохраняем результат GPU
    gpu_output_file = output_path + "_gpu.png"
    cv2.imwrite(gpu_output_file, highlighted_gpu_img)

    # Сравниваем результаты
    are_results_equal = np.array_equal(corners_cpu_result, gpu_corners_result)

    # Выводим результаты
    print(f"Время выполнения на CPU: {cpu_duration:.4f} секунд")
    print(f"Время выполнения на GPU: {gpu_duration:.4f} секунд")
    print(f"Результаты совпадают: {'Да' if are_results_equal else 'Нет'}")

if __name__ == "__main__":
    # Считываем параметры от пользователя
    input_image_path = input("Введите путь к исходному изображению: ")
    corner_threshold = float(input("Введите пороговое значение для детекции углов: "))
    result_image_path = input("Введите путь для сохранения результатов: ")

    process_image(input_image_path, corner_threshold, result_image_path)
