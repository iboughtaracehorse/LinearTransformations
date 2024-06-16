#####################################  LIBRARIES  #########################################

import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

#####################################  FUNCTIONS  #########################################


def eigenvalues_eigenvectors(matrix, limit):

    values, vectors = np.linalg.eig(matrix)
    check = []

    for i in range(len(values)):
        a = values[i]
        b = vectors[:, i]
        av = a * b
        bv = np.dot(matrix, b)

        if np.allclose(bv, av):
            check.append(True)
        else:
            check.append(False)

        values = np.round(values, limit) ### ATTENTION: THE NUMBERS ARE ROUNDED HERE!!!
        vectors = np.round(vectors, limit) ### ATTENTION: THE NUMBERS ARE ROUNDED HERE!!!

    return values, vectors, check


matrix = np.array([[4, -2], [1, 1]])
values, vectors, check = eigenvalues_eigenvectors(matrix, 2)
print(values, vectors, check)


image_raw = imread('nikki.jpg')
image_shape = image_raw.shape
print(image_shape)

plt.imshow(image_raw)
plt.axis('off')
plt.show()

