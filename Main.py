import numpy as np
import matplotlib.pyplot as plt


whatever = np.array([
    [0, 0],[0.5, 0.4],[1, 0],
    [0.5, -0.4],[0, 0],[0.4, 0.5],[0, 1],[-0.4, 0.5],[0, 0]
])

fish = np.array([
    [0, 0],[0.3, 0.1],[0.6, 0.3],
    [0.8, 0.05],[0.6, -0.2],[0.3, -0.1],[0, 0]
])

x_flower = whatever[:, 0]
y_flower = whatever[:, 1]

x_fish = fish[:, 0]
y_fish = fish[:, 1]

plt.plot(x_flower, y_flower, marker='o', color='blue', label='Flower')
plt.plot(x_fish, y_fish, marker='o',color='green', label='Fish')

plt.title("Whatever and Fish")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()