import tkinter as tk
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
n = 500
n1 = 26

# Параметры холста
canvas_width = 30
canvas_height = 30

# Функции системы
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
        1: '#000080', 2: '#008000', 3: '#0000FF', 4: '#800000', 5: '#808000',
        6: '#00FF00', 7: '#008080', 8: '#800080', 9: '#FFFF00', 10: '#00FFFF',
        11: '#C0C0C0', 12: '#808080', 13: '#FF00FF', 14: '#FF0000', 15: '#87CEEB'
    }
    if period < 16:
        return color_map.get(period, '#FFFFFF')
    elif 16 <= period < 25:
        return '#808080'
    return '#000000'

def generate_drm(canvas):
    start_time = time.perf_counter()
    L_vals = np.linspace(L_min, L_max, canvas_width)
    gamma_vals = np.linspace(gamma_min, gamma_max, canvas_height)

    def process_point(i, j, L, gamma):
        period = find_period(0.1, 0.1, gamma, mu, L)
        color = get_color_for_period(period)
        return i, j, color

    grid = [(i, j, L, gamma) for i, L in enumerate(L_vals) for j, gamma in enumerate(reversed(gamma_vals))]
    results = Parallel(n_jobs=-1)(delayed(process_point)(i, j, L, gamma) for i, j, L, gamma in grid)

    for i, j, color in results:
        canvas.create_rectangle(i, j, i + 1, j + 1, outline=color, fill=color)

    end_time = time.perf_counter()
    print(f"Время выполнения генерации карты: {end_time - start_time:.2f} секунд")

def main():
    root = tk.Tk()
    root.title("Карта динамических режимов")
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack()
    generate_drm(canvas)
    root.mainloop()

if __name__ == "__main__":
    main()
