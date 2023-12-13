import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
# Определение функции и её градиента
def func(x, y):
    return (2 * x ** 2 + y ** 2) * np.exp(-x ** 2 - y ** 2)


def gradient(x, y):
    df_dx = 4 * x * np.exp(-x ** 2 - y ** 2) - 2 * x * (2 * x ** 2 + y ** 2) * np.exp(-x ** 2 - y ** 2)
    df_dy = 2 * y * np.exp(-x ** 2 - y ** 2) - 2 * y * (2 * x ** 2 + y ** 2) * np.exp(-x ** 2 - y ** 2)
    return np.array([df_dx, df_dy])


# Параметры градиентного спуска
learning_rate = 0.01
max_iterations = 10000
epsilon = 1e-20  # Заданный эпсилон
delta = 1e-20 # Заданная дельта

# Начальные координаты
x_current, y_current = 0.5, 0

# Массив для хранения точек
points = []

counter_iteration = 0

# Градиентный спуск с условием остановки
for iteration in range(max_iterations):
    counter_iteration += 1
    grad = gradient(x_current, y_current)
    x_new, y_new = x_current - learning_rate * grad[0], y_current - learning_rate * grad[1]
    points.append((x_new, y_new))

    # Проверка условия остановки - модуль разности
    if np.abs(func(x_new, y_new) - func(x_current, y_current)) < epsilon:
        break

    # Проверка условия остановки - норма разности
    # if np.linalg.norm([x_current - x_new, y_current - y_new]) < delta:
    #     break

    x_current, y_current = x_new, y_new

# Преобразование в массив NumPy для удобства
points = np.array(points)

# Визуализация градиентного спуска
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Построение 3D-графика
x = np.linspace(-3, 3, 2000)
y = np.linspace(-3, 3, 2000)
x, y = np.meshgrid(x, y)
z = func(x, y)
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.5)

# Построение траектории градиентного спуска
ax.plot(points[:, 0], points[:, 1], func(points[:, 0], points[:, 1]), marker='o', color='r')

# Соединение точек прямой
ax.plot(points[:, 0], points[:, 1], np.zeros_like(points[:, 0]), color='b', linestyle='dashed')

# Добавление подписей
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(r'Градиентный спуск для $f(x, y) = (2x^2 + y^2)exp(-x^2 - y^2)$')
ax.set_zlim(0, 0.8)

end_time = time.time()
execution_time = end_time - start_time

print(f"Время выполнения программы: {execution_time} секунд")

# Отображение графика
plt.show()
print(points)
print(counter_iteration)
