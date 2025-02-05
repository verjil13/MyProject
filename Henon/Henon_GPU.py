
from PIL import Image
import numpy as np
from numba import cuda, float64
import time

# Параметры системы
delta = 0.01
maxregim = 13  # максимально разрешимый режим
Ntrans = 3000

# Разрешение изображения
image_width = 5000
image_height = 5000

# Функция для карты режима Хенона
@cuda.jit(device=True)
def Henon(x, alpha, beta):
    dx0 = 1 - alpha * x[0]**2 - beta * x[1]
    dx1 = x[0]
    return dx0, dx1

# Параметры сетки
a_min, a_max = 0, 2
b_min, b_max = -0.5, 0.5
a_n = np.linspace(a_min, a_max, image_height)
b_n = np.linspace(b_min, b_max, image_width)

# Функция для вычисления режима при заданных alpha и beta
@cuda.jit
def compute_regime_kernel(a_vals_gpu, b_vals_gpu, map_Henon_gpu, maxregim, Ntrans, delta):
    idx = cuda.grid(1)
    if idx < len(a_n) * len(b_n):
        i = idx // len(b_n)
        j = idx % len(b_n)
        a = a_vals_gpu[i]
        b = b_vals_gpu[j]
        
        x = cuda.local.array(2, dtype=float64)
        x[0] = 0.1
        x[1] = 0.1
        
        flag = False
        for n in range(Ntrans):
            x[0], x[1] = Henon(x, a, b)
            if abs(x[0]) > 10:
                flag = True
                break
        
        if flag:
            map_Henon_gpu[idx] = 0
            return
        
        y_sec = cuda.local.array(15, dtype=float64)
        y_sec[0] = x[0]
        
        for k in range(maxregim + 1):
            x[0], x[1] = Henon(x, a, b)
            y_sec[k + 1] = x[0]
        
        i_y = 1
        while i_y < 15 and abs(y_sec[i_y] - y_sec[0]) > delta:
            i_y += 1
        
        map_Henon_gpu[idx] = min(i_y, maxregim)

def generate_Henon_map():
    a_vals_gpu = cuda.to_device(np.array(a_n))
    b_vals_gpu = cuda.to_device(np.array(b_n))
    
    map_Henon = np.zeros(len(a_n) * len(b_n), dtype=np.int32)
    map_Henon_gpu = cuda.to_device(map_Henon)
    
    threads_per_block = 256
    blocks_per_grid = (len(a_n) * len(b_n) + (threads_per_block - 1)) // threads_per_block
    compute_regime_kernel[blocks_per_grid, threads_per_block](a_vals_gpu, b_vals_gpu, map_Henon_gpu, maxregim, Ntrans, delta)
    
    map_Henon = map_Henon_gpu.copy_to_host()
    map_Henon_2d = map_Henon.reshape((len(a_n), len(b_n)))
    map_Henon_2d = np.flipud(np.fliplr(map_Henon_2d))  # Исправленное отражение
    
    output_file = f"map_Henon_gpu_{image_width}.png"
    save_image(map_Henon_2d, output_file)
    print(f"Изображение сохранено в файл: {output_file}")
    
    img = Image.open(output_file)
    img.show()
    
    return map_Henon_2d

# Функция для сохранения изображения с цветовой картой
def save_image(data, filename):
    data_normalized = (data - data.min()) / (data.max() - data.min())
    cmap = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    cmap[..., 0] = np.clip(255 * data_normalized, 0, 255)  # Красный
    cmap[..., 1] = np.clip(255 * (1 - np.abs(data_normalized - 0.5) * 2), 0, 255)  # Зелёный
    cmap[..., 2] = np.clip(255 * (1 - data_normalized), 0, 255)  # Синий
    
    img = Image.fromarray(cmap, mode='RGB')
    img = img.resize((image_width, image_height), Image.LANCZOS)
    img.save(filename)

def main():
    start_time = time.perf_counter()
    generate_Henon_map()
    end_time = time.perf_counter()
    print(f"Время выполнения: {end_time - start_time:.2f} секунд")

if __name__ == "__main__":
    main()
