import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import tkinter as tk
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Система уравнений Dequan-Li
def DequanLi(XYZ, t, alpha, beta, delta, epsilon, rho, xi):
    x, y, z = XYZ
    x_dt = alpha*(y - x) + delta*x*z
    y_dt = rho*x + xi*y - x*z
    z_dt = beta*z + x*y - epsilon*x*x
    return x_dt, y_dt, z_dt

# Функция для запуска анимации
def run_animation():
    # Считываем параметры из интерфейса
    alpha = float(entry_alpha.get())
    beta = float(entry_beta.get())
    delta = float(entry_delta.get())
    epsilon = float(entry_epsilon.get())
    rho = float(entry_rho.get())
    xi = float(entry_xi.get())
    tmax = float(entry_tmax.get())
    n = int(entry_n.get())
    frame_step = int(entry_frame_step.get())

    # Начальные условия
    x_0, y_0, z_0 = 0.01, 0, 0
    t = np.linspace(0, tmax, n)

    # Интегрируем систему уравнений
    f = odeint(DequanLi, (x_0, y_0, z_0), t, args=(alpha, beta, delta, epsilon, rho, xi))
    X, Y, Z = f.T

    # Очищаем предыдущий график
    ax.clear()
    ax.set_title("3D Траектория Dequan-Li с динамическим масштабированием")
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

    # Функция обновления для анимации
    def update(frame):
        if frame == 0:
            return line, point

        # Обновляем траекторию
        line.set_data(X[:frame], Y[:frame])
        line.set_3d_properties(Z[:frame])
        point.set_data([X[frame]], [Y[frame]])
        point.set_3d_properties([Z[frame]])

        # Масштабируем автоматически при приближении к границе
        max_range = max(np.abs(X[:frame]).max(), np.abs(Y[:frame]).max(), np.abs(Z[:frame]).max())
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        return line, point

    # Анимация
    ani = FuncAnimation(fig, update, frames=range(0, n, frame_step), init_func=init, blit=True, interval=20, repeat=False)
    canvas.draw()

# Создаем графический интерфейс
root = tk.Tk()
root.title("Параметры системы Dequan-Li")

# Поля ввода для параметров
tk.Label(root, text="alpha:").grid(row=0, column=0)
entry_alpha = tk.Entry(root)
entry_alpha.insert(0, "40")
entry_alpha.grid(row=0, column=1)

tk.Label(root, text="beta:").grid(row=1, column=0)
entry_beta = tk.Entry(root)
entry_beta.insert(0, "1.833")
entry_beta.grid(row=1, column=1)

tk.Label(root, text="delta:").grid(row=2, column=0)
entry_delta = tk.Entry(root)
entry_delta.insert(0, "0.16")
entry_delta.grid(row=2, column=1)

tk.Label(root, text="epsilon:").grid(row=3, column=0)
entry_epsilon = tk.Entry(root)
entry_epsilon.insert(0, "0.65")
entry_epsilon.grid(row=3, column=1)

tk.Label(root, text="rho:").grid(row=4, column=0)
entry_rho = tk.Entry(root)
entry_rho.insert(0, "55")
entry_rho.grid(row=4, column=1)

tk.Label(root, text="xi:").grid(row=5, column=0)
entry_xi = tk.Entry(root)
entry_xi.insert(0, "20")
entry_xi.grid(row=5, column=1)

tk.Label(root, text="tmax:").grid(row=6, column=0)
entry_tmax = tk.Entry(root)
entry_tmax.insert(0, "50")
entry_tmax.grid(row=6, column=1)

tk.Label(root, text="n (точки времени):").grid(row=7, column=0)
entry_n = tk.Entry(root)
entry_n.insert(0, "40000")
entry_n.grid(row=7, column=1)

tk.Label(root, text="frame_step (шаг кадров):").grid(row=8, column=0)
entry_frame_step = tk.Entry(root)
entry_frame_step.insert(0, "50")
entry_frame_step.grid(row=8, column=1)

# Подготовка фигуры и осей для анимации
fig = plt.Figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=2, rowspan=10)

# Кнопка для запуска анимации
button_run = tk.Button(root, text="Запустить анимацию", command=run_animation)
button_run.grid(row=9, columnspan=2)

# Запуск интерфейса
root.mainloop()
