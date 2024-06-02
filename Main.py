import numpy as np
import matplotlib.pyplot as plt

## FUNCTIONS ##


def vector_rotation(vectors, angle):
    radians = np.deg2rad(angle)
    rotate = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    rotated = np.dot(vectors, rotate)
    return rotated


def vector_scale(vectors, coefficient):
    scale = np.array([ # but not icy...
        [coefficient, 0],
        [0, coefficient]
    ])

    scaled = np.dot(vectors, scale)
    return scaled


def mirror(vectors):
    # [1, -1] for x, [-1, 1] for y
    mirrored = vectors * np.array([-1, 1])
    return mirrored


def transformation(vectors, transformation):
    transformed = vectors * transformation
    return transformed

## FUNCTIONS END ##


flower = np.array([
    [0, 0],[0.5, 0.4],[1, 0],
    [0.5, -0.4],[0, 0],[0.4, 0.5],[0, 1],[-0.4, 0.5],[0, 0]
])

fish = np.array([
    [0, 0],[0.3, 0.1],[0.6, 0.3],
    [0.8, 0.05],[0.6, -0.2],[0.3, -0.1],[0, 0]
])

fish_rotation = vector_rotation(fish, 180)
fish_resized = vector_resize(fish, 2)
fish_mirrored = mirror(fish)
fish_transformed = transformation(fish, [5, 2])

x_fish = fish[:, 0]
y_fish = fish[:, 1]

x_fish_rotated = fish_rotation[:, 0]
y_fish_rotated = fish_rotation[:, 1]

x_fish_resized = fish_resized[:, 0]
y_fish_resized = fish_resized[:, 1]

x_fish_mirrored = fish_mirrored[:, 0]
y_fish_mirrored = fish_mirrored[:, 1]

x_fish_transformed = fish_transformed[:, 0]
y_fish_transformed = fish_transformed[:, 1]

plt.plot(x_fish, y_fish, marker='o',color='darkblue', label='Fish')
plt.plot(x_fish_mirrored, y_fish_mirrored, marker='o',color='green', label='Fish')
plt.plot(x_fish_transformed, y_fish_transformed, marker='o',color='red', label='Fish')
#plt.plot(x_fish_rotated, y_fish_rotated, marker='o',color='darkgreen', label='Fish rotation')
#plt.plot(x_fish_resized, y_fish_resized, marker='o',color='black', label='Fish resized')




##########################################################################
plt.title("Fish transformations")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
##########################################################################





