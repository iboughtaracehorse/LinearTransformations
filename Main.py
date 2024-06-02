import numpy as np
import matplotlib.pyplot as plt

##########################################################################


def vector_rotation(vectors, angle):
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = np.dot(vectors, rotate)
    return rotated


def vector_scale(vectors, coefficient):
    scale = np.array([
        [coefficient, 0],
        [0, coefficient]
    ])

    scaled = np.dot(vectors, scale) # but not icy...
    return scaled


def mirror(vectors):
    # [1, -1] for x, [-1, 1] for y
    mirrored = vectors * np.array([-1, 1])
    return mirrored


def axis_rotation(vectors, angle):
    rotate = np.array([[1, angle], [0, 1]])
    rotated = np.dot(vectors, rotate)
    return rotated


def transformation(vectors, transformation):
    transformed = np.dot(vectors, transformation)
    return transformed


##########################################################################

flower = np.array([
    [0, 0],[0.5, 0.4],[1, 0],
    [0.5, -0.4],[0, 0],[0.4, 0.5],[0, 1],[-0.4, 0.5],[0, 0]
])

fish = np.array([
    [0, 0],[0.3, 0.1],[0.6, 0.3],
    [0.8, 0.05],[0.6, -0.2],[0.3, -0.1],[0, 0]
])

pyramid = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0.5, 0.5, 1],
    [0, 0, 0]
])

edges = [
    [pyramid[0], pyramid[1]],
    [pyramid[1], pyramid[2]],
    [pyramid[2], pyramid[3]],
    [pyramid[3], pyramid[0]],
    [pyramid[0], pyramid[4]],
    [pyramid[1], pyramid[4]],
    [pyramid[2], pyramid[4]],
    [pyramid[3], pyramid[4]]
]

##########################################################################

fish_rotation = vector_rotation(fish, 2)
fish_resized = vector_scale(fish, 2)
fish_mirrored = mirror(fish)
fish_transformed = transformation(fish, [[1, 0],[0, 2]])
vector_axis_rotation = axis_rotation(fish, 2)

##########################################################################

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

x_fish_axis = vector_axis_rotation[:, 0]
y_fish_axis = vector_axis_rotation[:, 1]

##########################################################################


plt.plot(x_fish, y_fish, marker='o',color='#007362', label='Fish') # OG FISH
plt.plot(x_fish_rotated, y_fish_rotated, marker='o',color='#3b3b3b', label='Fish rotation') # ROTATED FISH
plt.plot(x_fish_mirrored, y_fish_mirrored, marker='o',color='#89b2a0', label='Fish mirrored') # MIRRORED FISH
#plt.plot(x_fish_resized, y_fish_resized, marker='o',color='cyan', label='Fish resized') # SCALED FISH
#plt.plot(x_fish_transformed, y_fish_transformed, marker='o',color='pink', label='Fish')  # TRANSFORMED FISH
#plt.plot(x_fish_axis, y_fish_axis, marker='o',color='pink', label='Fish')

##########################################################################

plt.title("Fish transformations")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

'''fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d')
for edge in edges:
    ax1.plot(*zip(*edge), color='black')
ax1.scatter(pyramid[:, 0], pyramid[:, 1], pyramid[:, 2], c='red', marker='o')
ax1.set_title("Original Pyramid")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
plt.show()'''

##########################################################################





