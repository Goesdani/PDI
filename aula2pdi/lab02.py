# importar biblioteca
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
img = cv2.imread("foto.jpg")

# Adicionar um ruído artificial à imagem:
# print(img.shape)

noise = np.random.normal(0, 25, img.shape).astype('uint8')
# print('Noise', noise.shape)
noisy_img = cv2.add(img, noise)

# Aplicar diferentes filtros:
gaussian = cv2.GaussianBlur(noisy_img, (3,3),0)
median = cv2.medianBlur(noisy_img, 3)
bilateral = cv2.bilateralFilter(noisy_img, 9, 75, 75)

# Imprimir resultados:
plt.subplot(2, 3, 1); plt.title('Original'); plt.imshow(img)
plt.subplot(2, 3, 2); plt.title('Noisy'); plt.imshow(noisy_img)
plt.subplot(2, 3, 3); plt.title('Gaussian'); plt.imshow(gaussian)
plt.subplot(2, 3, 4); plt.title('Median'); plt.imshow(median)
plt.subplot(2, 3, 5); plt.title('Bilateral'); plt.imshow(bilateral)

plt.show()