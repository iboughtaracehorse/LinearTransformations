#####################################  LIBRARIES  #########################################

import numpy as np
import matplotlib.pyplot as plt
import cv2

#####################################  FUNCTIONS  #########################################


def vector_rotation(vectors, angle):
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = np.dot(vectors, rotate)
    return rotated


def vector_scale(vectors, coefficient):
    scale = np.array([[coefficient, 0], [0, coefficient]])
    scaled = np.dot(vectors, scale) # but not icy...
    return scaled


def vector_scale_3d(vectors, coefficient):
    scale = np.array([[coefficient, 0, 0], [0, coefficient, 0], [0, 0, coefficient]])
    scaled = np.dot(vectors, scale)
    return scaled


def mirror(vectors, axis):
    if axis == "x":
        mirrored = vectors * np.array([1, -1])
    elif axis == "y":
        mirrored = vectors * np.array([-1, 1])

    return mirrored


def mirror_3d(vectors, axis):
    if axis == 'x':
        mirror = np.array([-1, 1, 1])
    elif axis == 'y':
        mirror = np.array([1, -1, 1])
    elif axis == 'z':
        mirror = np.array([1, 1, -1])
    mirrored = vectors * mirror

    return mirrored


def axis_rotation(vectors, angle, axis):
    if axis == "x":
        rotate = np.array([[1, angle], [0, 1]])
    elif axis == "y":
        rotate = np.array([[1, 0], [angle, 1]])

    rotated = np.dot(vectors, rotate)
    return rotated


def transformation(vectors, transformation):
    transformed = np.dot(vectors, transformation)
    return transformed


#####################################  OBJECTS  #########################################

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

#################################  FUNCTION CALLS  #####################################

fish_rotation = vector_rotation(fish, 20)
fish_resized = vector_scale(fish, -3)
fish_mirrored = mirror(fish, "y")
fish_transformed = transformation(fish, [[1, 4],[9, 1]])
vector_axis_rotation = axis_rotation(fish, 2, "y")
#pyramid_scaled = vector_scale_3d(pyramid, 2)
#pyramid_mirror = mirror_3d(pyramid, "z")

#############################  GETTING THOSE COORDINATES  #################################

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

###################################  PLOTTING  #######################################

#plt.plot(x_fish, y_fish, marker='o', color='#007362', label='Fish') ### OG FISH
#plt.plot(x_fish_rotated, y_fish_rotated, marker='o', color='#3b3b3b', label='Fish rotation') ### ROTATED FISH
#plt.plot(x_fish_mirrored, y_fish_mirrored, marker='o', color='#89b2a0', label='Fish mirrored') ### MIRRORED FISH
#plt.plot(x_fish_resized, y_fish_resized, marker='o', color='#003F73', label='Fish resized') ### SCALED FISH
#plt.plot(x_fish_transformed, y_fish_transformed, marker='o',color='#8993B2', label='TransformedFish')  ### TRANSFORMED FISH
#plt.plot(x_fish_axis, y_fish_axis, marker='o',color='#999F4B', label='Axis rotation')

###################################  SHOWING  #######################################

'''plt.title("Fish transformations")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

###################################  PYRAMID SHOW  #######################################

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(131, projection='3d')

for edge in edges:
    ax.plot(*zip(*edge), color='#007362')
ax.scatter(pyramid[:, 0], pyramid[:, 1], pyramid[:, 2], c='black', marker='o', label='Original Pyramid')

ax3 = fig.add_subplot(132, projection='3d')
for edge in edges:
    scaled_edge = [vector_scale_3d(np.array(edge), 2) for vertex in edge]
    ax.plot(*zip(*scaled_edge[0]), color='#89b2a0')
ax.scatter(pyramid_scaled[:, 0], pyramid_scaled[:, 1], pyramid_scaled[:, 2], c='black', marker='o', label='Scaled Pyramid')

ax3 = fig.add_subplot(133, projection='3d')
for edge in edges:
    mirrored_edge = mirror_3d(np.array(edge), axis='y')
    ax3.plot(*zip(*mirrored_edge), color='#007362')
ax3.scatter(pyramid_mirror[:, 0], pyramid_mirror[:, 1], pyramid_mirror[:, 2], c='black', marker='o', label='Mirrored Pyramid')

ax.set_title("Pyramid Scaling")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()'''

###################################  IMAGE  #######################################

'''
def rotate_image(image, angle):
    h, w, c = image.shape
    center = (w // 2, h // 2)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation, (w, h))
    return rotated_image


def scale_image(image, coefficient):
    h, w, c = image.shape
    new_height = int(h * coefficient)
    new_width = int(w * coefficient)
    scaled_and_icy = cv2.resize(image, (new_width, new_height))
    return scaled_and_icy


def mirror_image(image, axis):
    if axis == "x":
        mode = 0
    elif axis == "y":
        mode = 1

    mirrored = cv2.flip(image, mode)

    return mirrored


image = cv2.imread('nikki.jpg')

h, w, c = image.shape
image_coordinate = np.array([[0, 0], [0, h], [w, 0], [w, h]])

pic_rotation = rotate_image(image, 25)
scaled_image = scale_image(image, 0.5)
mirrored_image = mirror_image(image, "x")

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(pic_rotation, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image")
plt.axis('off')

cv2.imshow('Original Image', image)
#cv2.imshow('Scaled Image', scaled_image)
cv2.imshow('Mirrored Image', mirrored_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()'''

