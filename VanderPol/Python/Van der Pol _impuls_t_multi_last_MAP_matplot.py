import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # Импортируем для многопоточных вычислений

# Начало замера времени
start_time = time.perf_counter()

# Параметры системы
mu = 0.5      # Параметр нелинейности
L_min, L_max = -1, 4  # Диапазон L для построения КДР
gamma_min, gamma_max = -0.5, 4  # Диапазон gamma для построения КДР
eps = 1e-9  # Порог для определения возвращения в точку (для периода)
overflow_limit = 1e2  # Лимит значений для предотвращения переполнения
k = 10  # Параметр импульсного воздействия
w = 4  # Частота
h = 2 * np.pi / (w * k)  # Шаг Рунге-Кутты
n = 3000  # Количество итераций для выхода на режим
n1 = 26  # Количество итераций для определения периода

# Параметры графика
figsize = (5, 5)  # Размер графика в дюймах
dpi = 15  # Разрешение графика
s = 1  # Размер точек в квадратных пикселях

# Расчет шага step для равномерного покрытия графика точками
width_px = figsize[0] * dpi
height_px = figsize[1] * dpi
point_size = np.sqrt(s)
n_x = width_px // point_size
n_y = height_px // point_size
step_L = (L_max - L_min) / n_x
step_gamma = (gamma_max - gamma_min) / n_y

# Функции системы и метода Рунге-Кутты
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
    
    x_next = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    y_next = y + h * (l1 + 2 * l2 + 2 * l3 + l4) / 6
    return x_next, y_next

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
        1: 'navy', 2: 'green', 3: 'blue', 4: 'maroon', 5: 'olive',
        6: 'lime', 7: 'teal', 8: 'purple', 9: 'yellow', 10: 'aqua',
        11: 'silver', 12: 'gray', 13: 'fuchsia', 14: 'red', 15: 'skyblue'
    }
    if period < 16:
        return color_map.get(period, 'white')
    elif 16 <= period < 25:
        return 'gray'
    else:
        return 'black'

def generate_drm():
    L_vals = np.arange(L_min, L_max, step_L)
    gamma_vals = np.arange(gamma_min, gamma_max, step_gamma)
    grid = [(L, gamma) for L in L_vals for gamma in gamma_vals]
    
    results = Parallel(n_jobs=-1)(delayed(find_period)(0.1, 0.1, gamma, mu, L) for L, gamma in grid)

    end_time = time.perf_counter()
    print(f"Время выполнения: {end_time - start_time:.2f} секунд")
    fig, ax = plt.subplots(figsize=figsize)
    
    for (L, gamma), period in zip(grid, results):
        if period is not None:
            color = get_color_for_period(period)
            ax.scatter(L, gamma, color=color, s=s, marker='s')
    
    ax.set_title("Карта динамических режимов для системы Ван дер Поля")
    ax.set_xlabel("L")
    ax.set_ylabel("gamma")
    ax.set_facecolor("white")
    plt.show()

generate_drm()
