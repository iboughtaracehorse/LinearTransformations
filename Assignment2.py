#####################################  LIBRARIES  #########################################

import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

#####################################  0  #########################################


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


#####################################  1  #########################################


image_raw = imread('ew.jpg')
image_shape = image_raw.shape
print(image_shape)

plt.imshow(image_raw)
plt.axis('off')
plt.show()

image_vector = np.array(image_shape)
print(image_vector) ## what is the reason for this


#####################################  2  #########################################


image_sum = image_raw.sum(axis = 2)
image_bw = image_sum / image_sum.max()

plt.imshow(image_bw, cmap='gray')
plt.axis('off')
plt.show()

print(image_bw.max())

#####################################  3  #########################################


image_bw_centered = image_bw - np.mean(image_bw, axis=0)

cov_matrix = np.cov(image_bw_centered, rowvar=False)

values, vectors = np.linalg.eigh(cov_matrix)

sorted = np.argsort(values)[::-1]
values = values[sorted]
vectors = vectors[:, sorted]

variance = np.cumsum(values) / np.sum(values)

variance_percent = 0.95

n_components = np.where(variance >= variance)[0][0] + 1
print(f"Components for {variance_percent}", n_components)

plt.plot(variance)
plt.xlabel('Components')
plt.ylabel('Variance')
plt.title('Graph')
plt.grid(True)
plt.show()


