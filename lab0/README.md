# Лабораторная работа 0 "MatMul"
## Задача: 
- Реализовать алгоритм перемножения матриц Язык: C++ или Python.<br />
    - Входные данные: 2 матрицы размером от 100х100 до 2000х2000 каждая.<br />
    - УВыходные данные: проверка корректности перемножения + время вычисления.
    - Характеристика системы:

Процессор	AMD Ryzen 5 5500U with Radeon Graphics 2.10 GHz

Оперативная память	16,0 ГБ

Тип системы	64-разрядная операционная система, процессор x64

Видеокарта: NVIDIA GeForce GTX 1070 Ti


Реализация выполнена на языке **Python**.
Реализация алгоритма на GPU написана с использованием CUDA с помощью библиотеки numba.
Использовался **@cuda.jit** - декоратор для ускорения. 

В данной работе распараллелено вычисление элементов результирующей матрицы для получения ускорения.

Из полученных результатов можно сделать вывод о том, что при увелечении размеров блоков **растёт ускорение GPU относительно CPU**.
