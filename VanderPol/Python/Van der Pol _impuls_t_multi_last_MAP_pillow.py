from PIL import Image, ImageTk
import numpy as np
from joblib import Parallel, delayed
import time

# Параметры системы
mu = 0.5
L_min, L_max = -1, 4
gamma_min, gamma_max = -0.5, 4
eps = 1e-9
overflow_limit = 1e2
k = 10
w = 4
h = 2 * np.pi / (w * k)
n = 3000
n1 = 26

# Параметры изображения
img_width = 50
img_height = 50


def F(y):
    return y

def G(x, y, gamma, mu):
    try:
        result = (gamma - mu * x**2) * y - x
        if abs(result) > overflow_limit:
            return np.nan
    except OverflowError:
        return np.nan
    return result

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

def Imp(x, y, L, gamma, mu, h):
    for _ in range(10):
        x, y = runge_kutta_step(x, y, h, gamma, mu)
    y += L * (1 - x**2 / 2 + x**4 / 24)
    return x, y

def find_period(x, y, gamma, mu, L):
    for _ in range(n):
        x, y = Imp(x, y, L, gamma, mu, h)
        if (x > overflow_limit) or (y > overflow_limit) or np.isnan(x) or np.isnan(y):
            return 0
    atx, aty = x, y
    p = 0
    for _ in range(n1):
        x, y = Imp(x, y, L, gamma, mu, h)
        p += 1
        if np.sqrt((atx - x)**2 + (aty - y)**2) < eps:
            break
        if (x > overflow_limit) or (y > overflow_limit) or np.isnan(x) or np.isnan(y):
            return 0
    return p

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

    def process_pixel(L, gamma):
        period = find_period(0.1, 0.1, gamma, mu, L)
        return get_color_for_period(period)

    # Параллельная обработка пикселей
    pixels = Parallel(n_jobs=-1)(
        delayed(process_pixel)(L, gamma)
        for gamma in reversed(gamma_vals)
        for L in L_vals
    )

    # Преобразование списка в изображение
    img = Image.new("RGB", (img_width, img_height))
    img.putdata(pixels)
    return img

def main():
    start_time = time.perf_counter()
    img = generate_drm_image()
    end_time = time.perf_counter()
    print(f"Время выполнения генерации карты: {end_time - start_time:.2f} секунд")
    
    img.show()  # Открыть изображение в стандартном просмотрщике

if __name__ == "__main__":
    main()
