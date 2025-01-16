import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Система уравнений:
def HadleyAttractor(XYZ, t, alpha, beta, xi, delta):
    x, y, z = XYZ
    x_dt = -y*y - z*z - alpha*x + alpha*xi
    y_dt = x*y - beta*x*z - y + delta
    z_dt = beta*x*y + x*z-z
    return x_dt, y_dt, z_dt

# Параметры системы и начальные условия:
alpha = 0.2
beta = 4
xi = 8
delta = 1

x_0, y_0, z_0 = 0.39, -1, 0

# Максимальное время и общее количество
# временных точек:
tmax, n = 100, 10000

# Интегрируем систему уравнений в каждой точке
# временного интервала t:
t = np.linspace(0, tmax, n)
f = odeint(HadleyAttractor, (x_0, y_0, z_0), t,
           args=(alpha, beta, xi, delta))
X, Y, Z = f.T

# Подготовка фигуры для анимации
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Траектории с динамическим масштабированием")
line, = ax.plot([], [], [], lw=0.5, color='goldenrod')
point, = ax.plot([], [], [], 'bo')

# Инициализация границ
initial_range = 10
ax.set_xlim([-initial_range, initial_range])
ax.set_ylim([-initial_range, initial_range])
ax.set_zlim([-initial_range, initial_range])

# Функция инициализации
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point
# Параметры анимации

# Функция обновления для каждого кадра

def update(frame): 
    if frame == 0:
        return line, point
      
    # Обновляем траекторию
    line.set_data(X[:frame], Y[:frame])
    line.set_3d_properties(Z[:frame])
    point.set_data_3d([X[frame]], [Y[frame]], [Z[frame]])
    
    # Масштабируем автоматически при приближении к границе
    max_range = max(np.abs(X[:frame]).max(), np.abs(Y[:frame]).max(), np.abs(Z[:frame]).max())
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    return line, point

# Анимация

ani = FuncAnimation(fig, update, frames=range(0, n, 50), init_func=init, blit=True, interval=20, repeat=False)
plt.show()
