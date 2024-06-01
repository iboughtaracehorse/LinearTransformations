import numpy as np
import matplotlib.pyplot as plt

## FUNCTIONS ##


def vector_rotation(vectors, angle):
    radians = np.deg2rad(angle);

    rotate = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])

    rotated = np.dot(vectors, rotate)

    return rotated


def vector_resize(vectors, coefficient):
    scaled = np.array([ # but not icy...
        [coefficient, 0],
        [0, coefficient]
    ])

    # Scale each point
    resized = np.dot(vectors, scaled)

    return resized


## FUNCTIONS END ##


flower = np.array([
    [0, 0],[0.5, 0.4],[1, 0],
    [0.5, -0.4],[0, 0],[0.4, 0.5],[0, 1],[-0.4, 0.5],[0, 0]
])

fish = np.array([
    [0, 0],[0.3, 0.1],[0.6, 0.3],
    [0.8, 0.05],[0.6, -0.2],[0.3, -0.1],[0, 0]
])

fish_rotation = vector_rotation(fish, 20)
fish_resized = vector_resize(fish, 2)

x_fish = fish[:, 0]
y_fish = fish[:, 1]

x_fish_rotated = fish_rotation[:, 0]
y_fish_rotated = fish_rotation[:, 1]

x_fish_resized = fish_resized[:, 0]
y_fish_resized = fish_resized[:, 1]

plt.plot(x_fish, y_fish, marker='o',color='darkblue', label='Fish')
plt.plot(x_fish_rotated, y_fish_rotated, marker='o',color='darkgreen', label='Fish rotation')
plt.plot(x_fish_resized, y_fish_resized, marker='o',color='black', label='Fish resized')

plt.title("Fish")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()





