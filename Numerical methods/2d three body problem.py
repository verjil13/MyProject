import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Гравитационная постоянная
G = 1

# Массы тел
m1, m2, m3 = 1, 1, 1

# Функция для расчета производных (системы уравнений) трех тел
def three_body(t, y):
    x1, y1, x2, y2, x3, y3 = y[:6]
    vx1, vy1, vx2, vy2, vx3, vy3 = y[6:]
    
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
   

    
    
    ax1 = G * m2 * (x2 - x1) / r12**1.5 + G * m3 * (x3 - x1) / r13**1.5
    ay1 = G * m2 * (y2 - y1) / r12**1.5 + G * m3 * (y3 - y1) / r13**1.5
    
    ax2 = G * m1 * (x1 - x2) / r12**1.5 + G * m3 * (x3 - x2) / r23**1.5
    ay2 = G * m1 * (y1 - y2) / r12**1.5 + G * m3 * (y3 - y2) / r23**1.5
    
    ax3 = G * m1 * (x1 - x3) / r13**1.5 + G * m2 * (x2 - x3) / r23**1.5
    ay3 = G * m1 * (y1 - y3) / r13**1.5 + G * m2 * (y2 - y3) / r23**1.5
    
    return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

# Начальные условия
initial_conditions = [
    1.0, 0.0,                  # x1, y1
    -0.5, np.sqrt(3) / 2,      # x2, y2
    -0.5, -np.sqrt(3) / 2,     # x3, y3
    0.0, 0.5,                  # vx1, vy1
    -0.5, -0.25,               # vx2, vy2
    0.5, -0.25                 # vx3, vy3
]

# Время интеграции
t_span = (0, 500)
t_eval = np.linspace(*t_span, 20000)

# Решение системы
sol = solve_ivp(three_body, t_span, initial_conditions, t_eval=t_eval, method='DOP853')
x1, y1 = sol.y[0], sol.y[1]
x2, y2 = sol.y[2], sol.y[3]
x3, y3 = sol.y[4], sol.y[5]

# Подготовка фигуры для анимации
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Траектории трех тел")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.grid()

# Линии траекторий
line1, = ax.plot([], [], lw=1, color='blue', label='Тело 1')
line2, = ax.plot([], [], lw=1, color='green', label='Тело 2')
line3, = ax.plot([], [], lw=1, color='red', label='Тело 3')

# Точки текущих позиций
point1, = ax.plot([], [], 'bo')
point2, = ax.plot([], [], 'go')
point3, = ax.plot([], [], 'ro')

# Текст времени
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top')

# Параметры анимации
speed_factor = 5

# Функция инициализации
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    point1.set_data([], [])
    point2.set_data([], [])
    point3.set_data([], [])
    time_text.set_text('')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    return line1, line2, line3, point1, point2, point3, time_text

# Функция обновления кадров
def update(frame):
    frame *= speed_factor
    if frame >= len(x1):
        frame = len(x1) - 1
    
    # Обновляем траектории
    line1.set_data(x1[:frame], y1[:frame])
    line2.set_data(x2[:frame], y2[:frame])
    line3.set_data(x3[:frame], y3[:frame])
    
    # Обновляем текущие позиции тел
    point1.set_data([x1[frame]], [y1[frame]])
    point2.set_data([x2[frame]], [y2[frame]])
    point3.set_data([x3[frame]], [y3[frame]])
    
    # Обновляем текст времени
    current_time = sol.t[frame]
    time_text.set_text(f'Время: {current_time:.2f}')
    
    # Проверка текущих координат и обновление лимитов области отображения
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    margin = 0.5  # Поле для расширения границ
    all_x = [x1[frame], x2[frame], x3[frame]]
    all_y = [y1[frame], y2[frame], y3[frame]]
    
    # Проверка и изменение границ осей
    if min(all_x) < x_min + margin:
        ax.set_xlim(x_min - margin, x_max)
    if max(all_x) > x_max - margin:
        ax.set_xlim(x_min, x_max + margin)
    if min(all_y) < y_min + margin:
        ax.set_ylim(y_min - margin, y_max)
    if max(all_y) > y_max - margin:
        ax.set_ylim(y_min, y_max + margin)
    
    # Перерисовка осей с обновленными значениями
    ax.figure.canvas.draw()
    
    return line1, line2, line3, point1, point2, point3, time_text

# Создание анимации
ani = FuncAnimation(fig, update, frames=len(t_eval) // speed_factor, init_func=init, blit=True, interval=20, repeat=False)
plt.legend()
plt.show()
