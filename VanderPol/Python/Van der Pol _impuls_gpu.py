from PIL import Image
import numpy as np
from numba import cuda, float64
import time
import math

# Параметры системы
mu = 0.5
L_min, L_max = -1, 4
gamma_min, gamma_max = -0.5, 4
eps = 1e-12
overflow_limit = 1e4
k = 10
w = 4
h = 2 * np.pi / (w * k)
n = 3000
n1 = 26

# Параметры изображения
img_width = 1000
img_height = 1000

# Функции для расчета периодов на GPU
@cuda.jit
def compute_period(L_vals_gpu, gamma_vals_gpu, periods_gpu, mu, h, n, n1, eps, overflow_limit):
    idx = cuda.grid(1)
    if idx < img_width * img_height:
        i = idx // img_width
        j = idx % img_width
        
        # Используем обратное индексирование для gamma_vals_gpu чтобы учесть reversed
        i_reversed = img_height - 1 - i
        L = L_vals_gpu[j]
        gamma = gamma_vals_gpu[i_reversed]
        
        x = 0.1
        y = 0.1
        
        # Прогонка метода Рунге-Кутты и импульса
        for _ in range(n):
            for _ in range(10):  # Вызов функции Imp
                x, y = runge_kutta_step(x, y, h, gamma, mu)
            #y += L * (1 - x**2 / 2 + x**4 / 24)  # Добавление слагаемого из функции Imp
            y += L * math.cos(x)
            if (abs(x) > overflow_limit) or (abs(y) > overflow_limit) or (x != x) or (y != y):
                periods_gpu[idx] = 0
                return
        
        atx, aty = x, y
        p = 0
        for _ in range(n1):
            for _ in range(10):  # Вызов функции Imp
                x, y = runge_kutta_step(x, y, h, gamma, mu)
            #y += L * (1 - x**2 / 2 + x**4 / 24)  # Добавление слагаемого из функции Imp
            y += L * math.cos(x)
            p += 1
            if abs(atx - x) < eps and abs(aty - y) < eps:
                break
            if (abs(x) > overflow_limit) or (abs(y) > overflow_limit) or (x != x) or (y != y):
                periods_gpu[idx] = 0
                return
        
        periods_gpu[idx] = p

@cuda.jit(device=True)
def F(y):
    return y

@cuda.jit(device=True)
def G(x, y, gamma, mu):
    result = (gamma - mu * x**2) * y - x
    if abs(result) > overflow_limit:
        return np.nan
    return result

@cuda.jit(device=True)
def runge_kutta_step(x, y, h, gamma, mu):
    k1 = F(y)
    l1 = G(x, y, gamma, mu)
    k2 = F(y + h * l1 / 2)
    l2 = G(x + h * k1 / 2, y + h * l1 / 2, gamma, mu)
    k3 = F(y + h * l2 / 2)
    l3 = G(x + h * k2 / 2, y + h * l2 / 2, gamma, mu)
    k4 = F(y + h * l3)
    l4 = G(x + h * k3, y + h * l3, gamma, mu)
    return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6, y + h * (l1 + 2 * l2 + 2 * l3 + l4) / 6

def get_color_for_period(period):
    color_map = {
        1: (0, 0, 128), 2: (0, 128, 0), 3: (0, 0, 255), 4: (128, 0, 0), 5: (128, 128, 0),
        6: (0, 255, 0), 7: (0, 128, 128), 8: (128, 0, 128), 9: (255, 255, 0), 10: (0, 255, 255),
        11: (192, 192, 192), 12: (128, 128, 128), 13: (255, 0, 255), 14: (255, 0, 0), 15: (135, 206, 235)
    }
    if period < 16:
        return color_map.get(period, (255, 255, 255))
    elif 16 <= period < 25:
        return (128, 128, 128)
    return (0, 0, 0)

def generate_drm_image():
    L_vals = np.linspace(L_min, L_max, img_width)
    gamma_vals = np.linspace(gamma_min, gamma_max, img_height)
    
    # Создаем массив для хранения периодов
    periods = np.zeros(img_width * img_height, dtype=np.float64)
    
    # Перенос данных на GPU
    L_vals_gpu = cuda.to_device(L_vals)
    gamma_vals_gpu = cuda.to_device(gamma_vals)
    periods_gpu = cuda.to_device(periods)
    
    # Запуск CUDA-ядра
    threads_per_block = 256  # Увеличиваем количество потоков на блок
    blocks_per_grid = (img_width * img_height + (threads_per_block - 1)) // threads_per_block
    compute_period[blocks_per_grid, threads_per_block](L_vals_gpu, gamma_vals_gpu, periods_gpu, mu, h, n, n1, eps, overflow_limit)
    
    # Перенос данных обратно на CPU
    periods = periods_gpu.copy_to_host()
    
    # Преобразование списка в изображение
    pixels = [get_color_for_period(int(p)) for p in periods]
    
    # Исправляем порядок пикселей для правильного отображения
    img_array = np.array(pixels).reshape((img_height, img_width, 3))
    #img_array = np.flipud(img_array)  # Переворачиваем изображение по вертикали
    #img_array = np.fliplr(img_array)  # Отражаем изображение по горизонтали
     
    img = Image.fromarray(np.uint8(img_array))

    # Сохраняем изображение в файл
    output_file = "dynamic_mode_map.png"
    img.save(output_file)
    print(f"Изображение сохранено в файл: {output_file}")
    
    return img

def main():
    start_time = time.perf_counter()
    img = generate_drm_image()
    end_time = time.perf_counter()
    print(f"Время выполнения генерации карты: {end_time - start_time:.2f} секунд")
    
    img.show()  # Открыть изображение в стандартном просмотрщике

if __name__ == "__main__":
    main()
