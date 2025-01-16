
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Формула Kranen-Nes
def kranen_nes(T, Tb):
    T_kelvin = T + 273.15  # Перевод температуры в Кельвины
    return 100 * 10 ** (
        3.2041 * (
            1 - 0.998 * ((Tb - 41) / (T_kelvin - 41)) * ((1393 - T_kelvin) / (1393 - Tb))
        )
    )

# Формула Антуана
def antoine(T, A, B, C):
    return 10 ** (A - B / (T + C)) * 0.133322

# Запрос температуры кипения у пользователя
try:
    Tb_celsius = float(input("Введите температуру кипения вещества (°C): "))
except ValueError:
    print("Ошибка: введите числовое значение.")
    exit()

Tb_kelvin = Tb_celsius + 273.15  # Перевод температуры кипения в Кельвины

# Генерация данных с помощью формулы Kranen-Nes
T_range = np.linspace(-100, 500, 100)  # Диапазон температур, °C
P_data = kranen_nes(T_range, Tb_kelvin)  # Давления, рассчитанные по Kranen-Nes

# Преобразуем данные для логарифмической формы уравнения Антуана
logP_data = np.log10(P_data / 0.133322)  # Давление в мм рт. ст. и логарифм

# Функция для curve_fit (логарифмическая форма)
def antoine_log(T, A, B, C):
    return A - B / (T + C)

# Увеличиваем количество итераций для curve_fit
maxfev = 100

# Улучшенные начальные предположения для коэффициентов Антуана
initial_guess = [7, 800, 150]

# Подгонка коэффициентов
try:
    popt, pcov = curve_fit(antoine_log, T_range, logP_data, p0=initial_guess, maxfev=maxfev)
    A_fit, B_fit, C_fit = popt
    print(f"Коэффициенты Антуана:\nA = {A_fit:.5f}, B = {B_fit:.5f}, C = {C_fit:.5f}")

    # Построение графиков
    plt.figure(figsize=(10, 6))

    # Данные Kranen-Nes
    plt.plot(T_range, P_data, 'o', label='Kranen-Nes Data', markersize=5)

    # Данные Антуана с найденными коэффициентами
    P_antoine_fit = antoine(T_range, A_fit, B_fit, C_fit)
    plt.plot(T_range, P_antoine_fit, '-', label='Antoine Fit')

    # Настройки графика
    plt.xlabel('Температура, °C')
    plt.ylabel('Давление, кПа')
    plt.title('Сравнение давления по Kranen-Nes и Антуану')
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()
except RuntimeError as e:
    print(f"Ошибка подгонки: {e}")
    print("Попробуйте изменить начальные параметры или диапазон данных.")
