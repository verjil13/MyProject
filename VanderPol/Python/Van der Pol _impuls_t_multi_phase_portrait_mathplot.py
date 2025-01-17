import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Замер времени начала
start_time = time.perf_counter()

# Система уравнений Ван дер Поля
def VanDerPol(XY, t, mu, gamma):
    x, y = XY
    x_dt = y
    y_dt = (gamma - mu * x**2) * y - x
    return x_dt, y_dt

# Параметры системы
mu = 0.001      # Параметр нелинейности
gamma = 0.001   # Параметр системы
L = 0.3         # Величина внешнего воздействия
k = 10          # Параметр для шага и внешнего воздействия

# Параметры интегрирования
tmax = 20                  # Максимальное время
n = 24000                  # Количество шагов
h = 2 * np.pi / (4 * k)    # Шаг интегрирования
overflow_limit = 1e1       # Лимит значений для предотвращения переполнения

# Диапазоны начальных условий
xmin, xmax = -10, 10
ymin, ymax = -10, 10
d = 1

####################################
###  ФУНКЦИИ ДЛЯ РК-4 И ИМПУЛЬСА ###
####################################

# Функции для уравнений
def F(y):
    return y

def G(x, y, mu, gamma):
    try:
        result = (gamma - mu * x**2) * y - x
        if abs(result) > overflow_limit:
            return np.nan  # Прекращаем вычисления при достижении предела
    except OverflowError:
        return np.nan  # При переполнении возвращаем NaN, чтобы остановить траекторию
    return result

# Метод Рунге-Кутта 4-го порядка
def runge_kutta_step(x, y, h, mu, gamma):
    k1 = F(y)
    l1 = G(x, y, mu, gamma)
    if np.isnan(l1): return np.nan, np.nan  # Прерывание при переполнении
    
    k2 = F(y + h * l1 / 2)
    l2 = G(x + h * k1 / 2, y + h * l1 / 2, mu, gamma)
    if np.isnan(l2): return np.nan, np.nan
    
    k3 = F(y + h * l2 / 2)
    l3 = G(x + h * k2 / 2, y + h * l2 / 2, mu, gamma)
    if np.isnan(l3): return np.nan, np.nan
    
    k4 = F(y + h * l3)
    l4 = G(x + h * k3, y + h * l3, mu, gamma)
    if np.isnan(l4): return np.nan, np.nan
    
    x_next = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    y_next = y + h * (l1 + 2 * l2 + 2 * l3 + l4) / 6
    
    # Проверяем, не превышают ли значения предел
    if abs(x_next) > overflow_limit or abs(y_next) > overflow_limit:
        return np.nan, np.nan
    
    return x_next, y_next

# Функция для добавления внешнего воздействия
def Imp(x, y, L):
    # Добавляем воздействие к y
    y += L * (1 - x**2 / 2 + x**4 / 24)
    return x, y

# Функция для вычисления траектории для заданного начального условия
def compute_trajectory(initial_condition):
    x_0, y_0 = initial_condition
    X, Y = [], []
    
    x, y = x_0, y_0
    for i in range(1, n):
        x, y = runge_kutta_step(x, y, h, mu, gamma)
        if np.isnan(x) or np.isnan(y):
            break  # Прерываем интегрирование при переполнении
        
        # Добавляем внешнее воздействие каждые k шагов
        if i % k == 0:
            x, y = Imp(x, y, L)

        # Прерываем траекторию, если значения выходят за пределы overflow_limit
        if abs(x) > overflow_limit or abs(y) > overflow_limit:
            break
        
        # Начиная с 100-й итерации, добавляем точки в массив
        r = k * 2  # Разрежаем точки
        if i > 100 and i % r == 0:
            X.append(x)
            Y.append(y)
    
    return X, Y

####################################
###  ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ  ###
####################################

# Создаем начальные условия по сетке
initial_conditions = [(x_0, y_0) for x_0 in np.arange(xmin, xmax, d) for y_0 in np.arange(ymin, ymax, d)]

# Параллельное вычисление траекторий для каждого начального условия с использованием joblib
results = Parallel(n_jobs=-1)(delayed(compute_trajectory)(ic) for ic in initial_conditions)

# Подготовка графика
fig, ax = plt.subplots(figsize=(10, 10))

# Добавляем траектории на график
for X, Y in results:
    if X and Y:  # Проверяем, что траектория не пуста
        ax.scatter(X, Y, c=np.linspace(0, 1, len(X)), cmap='viridis', s=0.5)

# Замер времени окончания
end_time = time.perf_counter()

# Вычисляем и выводим затраченное время
print(f"Время выполнения: {end_time - start_time:.2f} секунд")

#######################
###  ВИЗУАЛИЗАЦИЯ  ###
#######################

# Настройка параметров графика
ax.set_title("Фазовая траектория осциллятора Ван дер Поля для множества начальных условий")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_facecolor("black")  # Фон графика
plt.show()
