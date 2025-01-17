import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from joblib import Parallel, delayed

# Замер времени начала
start_time = time.perf_counter()

# Параметры визуализации
rcParams['font.sans-serif']=['Arial', 'Dejavu Sans']
rcParams['font.size'] = 24

# Параметры системы
delta = 0.01
maxregim = 13  # максимально разрешимый режим
Ntrans = 3000

# Функция для карты режима Хенона
def Henon(x, alpha, beta):
    dx = np.zeros(2)
    dx[0] = 1 - alpha * x[0]**2 - beta * x[1]
    dx[1] = x[0]
    return dx

# Параметры сетки
a_min, a_max, a_d = 0, 2, 0.01
b_min, b_max, b_d = -0.5, 0.5, 0.005
a_n = np.arange(a_min, a_max + delta, a_d)
b_n = np.arange(b_min, b_max + delta, b_d)

# Функция для вычисления режима при заданных alpha и beta
def compute_regime(a, b):
    # Пропускаем переходной процесс длины Ntrans
    x = [0.1, 0.1]  # начальные условия
    flag = False
    for n in range(Ntrans):
        x = Henon(x, a, b)
        if abs(x[0]) > 10:
            flag = True
            break
    if flag:
        return -1  # сигнал о выходе за пределы

    # Вычисляем режим и проверяем периодичность
    y_sec = [x[0]]
    for _ in range(maxregim + 1):
        x = Henon(x, a, b)
        y_sec.append(x[0])
    
    i_y = 1
    while i_y < len(y_sec) and abs(y_sec[i_y] - y_sec[0]) > delta:
        i_y += 1
    return min(i_y, maxregim)

# Параллельное вычисление карты
map_Henon = np.zeros((len(a_n), len(b_n)), dtype=int)
results = Parallel(n_jobs=-1)(delayed(compute_regime)(a, b) for i, a in enumerate(a_n) for j, b in enumerate(b_n))

# Заполнение карты результатами
for index, (i, j) in enumerate(np.ndindex(len(a_n), len(b_n))):
    map_Henon[i, j] = results[index] if results[index] != -1 else 0
    
# Замер времени окончания
end_time = time.perf_counter()

# Вычисляем и выводим затраченное время
print(f"Время выполнения: {end_time - start_time:.2f} секунд")  

# Построение графика
plt.figure(figsize=(9, 9))
plt.imshow(map_Henon, cmap='jet', origin='lower',
           extent=(b_min, b_max, a_min, a_max), aspect=b_d / a_d)
plt.colorbar(boundaries=np.arange(-0.5, maxregim + 1.5, 1),
             ticks=range(0, maxregim + 1), fraction=0.044)
plt.xlabel(r'$\beta$', fontsize=28)
plt.ylabel(r'$\alpha$', fontsize=28)
plt.tight_layout()
plt.savefig('map_Henon_parallel.png')
plt.show()
