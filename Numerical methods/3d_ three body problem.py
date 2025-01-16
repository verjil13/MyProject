import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Гравитационная постоянная (условная для масштабирования)
G = 1.0

# Массы тел
m1, m2, m3 = 1.0, 2.0, 1.0

# Функция для расчета производных (системы уравнений) трех тел в 3D
def three_body_3d(t, y):
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = y[:9]
    vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3 = y[9:]
    
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)
    
    ax1 = G * m2 * (x2 - x1) / r12**1.5 + G * m3 * (x3 - x1) / r13**1.5
    ay1 = G * m2 * (y2 - y1) / r12**1.5 + G * m3 * (y3 - y1) / r13**1.5
    az1 = G * m2 * (z2 - z1) / r12**1.5 + G * m3 * (z3 - z1) / r13**1.5
    
    ax2 = G * m1 * (x1 - x2) / r12**1.5 + G * m3 * (x3 - x2) / r23**1.5
    ay2 = G * m1 * (y1 - y2) / r12**1.5 + G * m3 * (y3 - y2) / r23**1.5
    az2 = G * m1 * (z1 - z2) / r12**1.5 + G * m3 * (z3 - z2) / r23**1.5
    
    ax3 = G * m1 * (x1 - x3) / r13**1.5 + G * m2 * (x2 - x3) / r23**1.5
    ay3 = G * m1 * (y1 - y3) / r13**1.5 + G * m2 * (y2 - y3) / r23**1.5
    az3 = G * m1 * (z1 - z3) / r13**1.5 + G * m2 * (z2 - z3) / r23**1.5
    
    return [vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3]

# Начальные условия
initial_conditions = [
    1.0, 0.0, 0.0,               # x1, y1, z1 (позиция тела 1)
    -0.5, np.sqrt(3)/2, 0.0,     # x2, y2, z2 (позиция тела 2)
    -0.5, -np.sqrt(3)/2, 0.0,    # x3, y3, z3 (позиция тела 3)
    0.0, 0.5, 0.5,               # vx1, vy1, vz1 (скорость тела 1)
    -0.433, -0.25, -0.5,         # vx2, vy2, vz2 (скорость тела 2)
    0.433, -0.25, -0.5           # vx3, vy3, vz3 (скорость тела 3)      
]

# Время интеграции
t_span = (0, 500)
t_eval = np.linspace(*t_span, 20000)

# Решаем систему уравнений
sol = solve_ivp(three_body_3d, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Получаем траектории
x1, y1, z1 = sol.y[0], sol.y[1], sol.y[2]
x2, y2, z2 = sol.y[3], sol.y[4], sol.y[5]
x3, y3, z3 = sol.y[6], sol.y[7], sol.y[8]

# Подготовка фигуры для анимации
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Линии и точки для отображения тел
line1, = ax.plot([], [], [], lw=1, color='blue', label='Тело 1')
line2, = ax.plot([], [], [], lw=1, color='green', label='Тело 2')
line3, = ax.plot([], [], [], lw=1, color='red', label='Тело 3')
point1, = ax.plot([], [], [], 'bo')
point2, = ax.plot([], [], [], 'go')
point3, = ax.plot([], [], [], 'ro')

# Начальный масштаб
def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    line3.set_data([], [])
    line3.set_3d_properties([])
    point1.set_data([], [])
    point1.set_3d_properties([])
    point2.set_data([], [])
    point2.set_3d_properties([])
    point3.set_data([], [])
    point3.set_3d_properties([])
    return line1, line2, line3, point1, point2, point3

def update(frame):
    # Получаем минимальные и максимальные значения координат на текущем кадре
    min_x = min(x1[:frame+1].min(), x2[:frame+1].min(), x3[:frame+1].min())
    max_x = max(x1[:frame+1].max(), x2[:frame+1].max(), x3[:frame+1].max())
    min_y = min(y1[:frame+1].min(), y2[:frame+1].min(), y3[:frame+1].min())
    max_y = max(y1[:frame+1].max(), y2[:frame+1].max(), y3[:frame+1].max())
    min_z = min(z1[:frame+1].min(), z2[:frame+1].min(), z3[:frame+1].min())
    max_z = max(z1[:frame+1].max(), z2[:frame+1].max(), z3[:frame+1].max())

    # Устанавливаем границы области отображения с запасом
    margin = 0.1
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_zlim(min_z - margin, max_z + margin)
    
    # Перерисовываем холст для обновления масштаба
    fig.canvas.draw()

    # Обновляем траектории
    line1.set_data(x1[:frame], y1[:frame])
    line1.set_3d_properties(z1[:frame])
    line2.set_data(x2[:frame], y2[:frame])
    line2.set_3d_properties(z2[:frame])
    line3.set_data(x3[:frame], y3[:frame])
    line3.set_3d_properties(z3[:frame])
    
    # Обновляем текущие позиции тел
    point1.set_data([x1[frame]], [y1[frame]])
    point1.set_3d_properties([z1[frame]])
    point2.set_data([x2[frame]], [y2[frame]])
    point2.set_3d_properties([z2[frame]])
    point3.set_data([x3[frame]], [y3[frame]])
    point3.set_3d_properties([z3[frame]])
    
    return line1, line2, line3, point1, point2, point3

# Создание анимации
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20, repeat=False)
plt.legend()
plt.show()
