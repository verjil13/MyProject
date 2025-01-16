import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Гравитационная постоянная
G = 1.0

# Функция для расчета производных (система уравнений трех тел в 3D)
def three_body_3d(t, y, m1, m2, m3):
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

# Функция для запуска анимации
def run_simulation():
    # Получаем значения параметров из полей ввода
    m1 = float(entry_m1.get())
    m2 = float(entry_m2.get())
    m3 = float(entry_m3.get())
    tmax = float(entry_tmax.get())
    n = int(entry_n.get())
    skip_frames = int(entry_skip.get())
    
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
    t_span = (0, tmax)
    t_eval = np.linspace(*t_span, n)

    # Решаем систему уравнений
    sol = solve_ivp(three_body_3d, t_span, initial_conditions, t_eval=t_eval, args=(m1, m2, m3), method='RK45')
    
    # Получаем траектории
    x1, y1, z1 = sol.y[0], sol.y[1], sol.y[2]
    x2, y2, z2 = sol.y[3], sol.y[4], sol.y[5]
    x3, y3, z3 = sol.y[6], sol.y[7], sol.y[8]

    # Очистка предыдущего графика
    ax.clear()
    
    # Устанавливаем легенды и оси
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['Тело 1', 'Тело 2', 'Тело 3'])

    # Линии и точки для отображения тел
    line1, = ax.plot([], [], [], lw=1, color='blue')
    line2, = ax.plot([], [], [], lw=1, color='green')
    line3, = ax.plot([], [], [], lw=1, color='red')
    point1, = ax.plot([], [], [], 'bo')
    point2, = ax.plot([], [], [], 'go')
    point3, = ax.plot([], [], [], 'ro')

    def init():
        # Инициализация графика
        line1.set_data([], [])
        line1.set_3d_properties([])
        line2.set_data([], [])
        line2.set_3d_properties([])
        line3.set_data([], [])
        line3.set_3d_properties([])
        return line1, line2, line3, point1, point2, point3

    def update(frame):
        frame = frame * skip_frames  # Пропуск кадров для ускорения
        if frame >= len(x1):
            return line1, line2, line3, point1, point2, point3

        # Обновление данных
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

        # Автоматическое масштабирование
        if frame > 0:  # Проверка на наличие данных
            max_range = max(abs(x1[:frame]).max(), abs(y1[:frame]).max(), abs(z1[:frame]).max(),
                            abs(x2[:frame]).max(), abs(y2[:frame]).max(), abs(z2[:frame]).max(),
                            abs(x3[:frame]).max(), abs(y3[:frame]).max(), abs(z3[:frame]).max())
            margin = 0.1
            ax.set_xlim([-max_range - margin, max_range + margin])
            ax.set_ylim([-max_range - margin, max_range + margin])
            ax.set_zlim([-max_range - margin, max_range + margin])

        return line1, line2, line3, point1, point2, point3

    ani = FuncAnimation(fig, update, frames=len(t_eval)//skip_frames, init_func=init, blit=True, interval=20, repeat=False)
    canvas.draw()

# Создание графического интерфейса
root = tk.Tk()
root.title("3D Задача трёх тел - параметры")

# Метки и поля для ввода параметров
tk.Label(root, text="Масса 1:").grid(row=0, column=0)
entry_m1 = tk.Entry(root)
entry_m1.insert(0, "1.0")
entry_m1.grid(row=0, column=1)

tk.Label(root, text="Масса 2:").grid(row=1, column=0)
entry_m2 = tk.Entry(root)
entry_m2.insert(0, "1.0")
entry_m2.grid(row=1, column=1)

tk.Label(root, text="Масса 3:").grid(row=2, column=0)
entry_m3 = tk.Entry(root)
entry_m3.insert(0, "1.0")
entry_m3.grid(row=2, column=1)

tk.Label(root, text="Макс. время:").grid(row=3, column=0)
entry_tmax = tk.Entry(root)
entry_tmax.insert(0, "10")
entry_tmax.grid(row=3, column=1)

tk.Label(root, text="Количество точек:").grid(row=4, column=0)
entry_n = tk.Entry(root)
entry_n.insert(0, "1000")
entry_n.grid(row=4, column=1)

tk.Label(root, text="Пропуск кадров:").grid(row=5, column=0)
entry_skip = tk.Entry(root)
entry_skip.insert(0, "5")
entry_skip.grid(row=5, column=1)

# Кнопка запуска
tk.Button(root, text="Запустить", command=run_simulation).grid(row=6, columnspan=2)

# Настройка области графика
fig = Figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=2, rowspan=7)

root.mainloop()
