####### Programa para leer imágenes y realizar operaciones ######

#### 1. Importación de librería OpenCV ####
import cv2
import numpy as np
from matplotlib import pyplot as plt

#### 2. Ruta de las imágenes ####
ruta_imagen_1 = 'image.jpg'
ruta_imagen_2 = 'playa.jpg'

#### 3. Nombre de las imágenes ####
nombre_imagen_1 = 'Imagen 1 original'
nombre_imagen_2 = 'Imagen 2 original'

#### 4. Funciones ####

# 4.1. Función para cargar una imagen
def load_image(path, window_name):    
    image = cv2.imread(path)
    if image is None:
        print("Error: No se pudo cargar la imagen")
    else: 
        return image
    
# 4.2. Función para redimensionar una imagen 
def resize_image(image, widht, height):
    image_resized = cv2.resize(image, (widht, height))
    return image_resized

# 4.3. Función para sumar imágenes
def add_images(image1, image2):
    return cv2.add(image1, image2)

# 4.4. Función para restar imágenes
def subtract_images(image1, image2):
    return cv2.subtract(image1, image2)

# 4.5. Función para sumar con peso
def addWeighted_image(image1, image2):
    return cv2.addWeighted(image1, 0.7, image2, 0.3, 0)

# 4.6. Función para dividir imágenes
def bitwise_image(image1, image2):
    # Evitar la división por cero añadiendo un pequeño valor a image2
    image2 = np.where(image2 == 0, 1, image2)
    return cv2.bitwise_and(image1, image2, mask = None)

# 4.7. Funcion color hsv
def color_image_hsv(image1):
    hsv_image = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    return hsv_image

# 4.8. Funcion color gray
def color_image_gray(image1):
    gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    return gray_image

#### 5. Carga de imágenes ####
imagen1_original = load_image(ruta_imagen_1, nombre_imagen_1)
imagen2_original = load_image(ruta_imagen_2, nombre_imagen_2)
  
#### 6. Ejecución de operaciones ####

# 6.1. Operación 1: redimensionamiento de imágenes
imagen1 = resize_image(imagen1_original, 480, 480)
imagen2 = resize_image(imagen2_original, 480, 480)

# 6.2. Operación 2: rotación (180 grados)
# Dimensiones y cálculo del centro de la imagen
(h, w) = imagen1.shape[:2]
center = (w / 2, h / 2)
# Rotación
M = cv2.getRotationMatrix2D(center, 180, 1.0)
rotated = cv2.warpAffine(imagen1, M, (w, h))

# 6.3. Operación 3: crop (cortar)
#cropped = imagen1[70:170, 440:540]
cropped = imagen1[70:170, 325:450]

# 6.4. Operación 4: suma
added_image = add_images(imagen1, imagen2)

# 6.5. Operación 5: resta
subtracted_image = subtract_images(imagen1, imagen2)

# 6.6. Operación 6: suma con peso
addedWeighted_image = addWeighted_image(imagen1, imagen2)

# 6.5. Operación 7: operación bit
bitedwise_image = bitwise_image(imagen1, imagen2)

# 6.6. Operación 8: color hsv
colored_image_hsv = color_image_hsv (imagen1)

# 6.7. Operación 9: color gray
colored_image_gray = color_image_gray (imagen1)

# 6.8. Operación 10: convolucion 2D
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(imagen1,-1,kernel)

# 6.9. Operación 11: difuminado
blur = cv2.blur(imagen1,(5,5))

# 6.10. Operación 12: mediana
median = cv2.medianBlur(imagen1,5)

# 6.11. Operación 13: difuminacion Gaussiana
blur_g = cv2.GaussianBlur(imagen1,(5,5),0)

# 6.12. Operación 14: bilateral
blur_b = cv2.bilateralFilter(imagen1,9,75,75)

# 6.13. Operación 15: binary
ret,thr1 = cv2.threshold(colored_image_gray,127,255,cv2.THRESH_BINARY)

# 6.14. Operación 16: binary_Inv
ret,thr2 = cv2.threshold(colored_image_gray,127,255,cv2.THRESH_BINARY_INV)

# 6.15. Operación 17: trunc
ret,thr3 = cv2.threshold(colored_image_gray,127,255,cv2.THRESH_TRUNC)

# 6.16. Operación 18: tozero
ret,thr4 = cv2.threshold(colored_image_gray,127,255,cv2.THRESH_TOZERO)

# 6.17. Operación 19: tozero_Inv
ret,thr5 = cv2.threshold(colored_image_gray,127,255,cv2.THRESH_TOZERO_INV)

# 6.18. Operación 20: binary
image_blur = cv2.medianBlur(colored_image_gray,5)
ret,thr1 = cv2.threshold(image_blur,127,255,cv2.THRESH_BINARY)

# 6.19. Operación 21: adaptive thresh mean
thr2 = cv2.adaptiveThreshold(image_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

# 6.20. Operación 22: adaptive thresh gaussian
thr3 = cv2.adaptiveThreshold(image_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

# 6.21. Operación 23: otsu
# Global thresholding
ret1, thr1 = cv2.threshold(colored_image_gray,127,255,cv2.THRESH_BINARY)
# Thresholding con Binarización de Otsu
ret2, thr2 = cv2.threshold(colored_image_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Thresholding con Binarización de Otsu tras filtro Gaussiano.
blur = cv2.GaussianBlur(colored_image_gray,(5,5),0)
ret3, thr3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#### 7. Mostrar los resultados ####

### Primera parte: operaciones básicas en 1 imágen ###

# 7.1. Imagen original
cv2.imshow('Imagen 1 original', imagen1_original)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7.2. Operación 1: redimensionamiento de imágenes
cv2.imshow('Imagen 1 redimensionada', imagen1)
cv2.waitKey(0)

# 7.3. Operación 2: rotación
cv2.imshow("rotated", rotated)
cv2.waitKey(0)

# 7.4. Operación 3: crop
cv2.imshow("cropped", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Segunda parte: operaciones con 2 imágenes ###

# 7.5. Imágenes originales
cv2.imshow('Imagen 1 original', imagen1_original)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagen 2 original', imagen2_original)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7.6. Operación 1: redimensionamiento de imágenes
cv2.imshow('Imagen 1 redimensionada', imagen1)
cv2.waitKey(0)
cv2.imshow('Imagen 2 redimensionada', imagen2)
cv2.waitKey(0)

# 7.7. Operación 4: suma
cv2.imshow('Imagen Suma', added_image)
cv2.waitKey(0)

# 7.8. Operación 5: resta
cv2.imshow('Imagen Resta', subtracted_image)
cv2.waitKey(0)

# 7.9. Operación 6: suma con peso
cv2.imshow('Suma con peso', addedWeighted_image)
cv2.waitKey(0)

# 7.10. Operación 7: operación con bit
cv2.imshow('Operacion Bit', bitedwise_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Tercera parte: operaciones con 1 imágen ###

# 7.11. Operación 1: redimensionamiento de imágenes
cv2.imshow('Imagen 1 redimensionada', imagen1)
cv2.waitKey(0)

# 7.12. Operación 8: color hsv
cv2.imshow('Color HSV', colored_image_hsv)
cv2.waitKey(0)

# 7.13. Operación 9: color gray
cv2.imshow('Color GRAY', colored_image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Cuarta parte: filtros (ruido) a 1 imágen ###

# 7.14. Operación 1: redimensionamiento de imágen
cv2.imshow('Imagen 1 redimensionada', imagen1)
cv2.waitKey(0)

# 7.15. Operación 10: convolucion 2D
cv2.imshow('Convolucion 2D',dst)
cv2.waitKey(0)

# 7.16. Operación 11: difuminado
cv2.imshow('Blur',blur)
cv2.waitKey(0)

# 7.17. Operación 12: mediana
cv2.imshow('Mediana',median)
cv2.waitKey(0)

# 7.18. Operación 13: gaussiana
cv2.imshow('Gaussiana',blur_g)
cv2.waitKey(0)

# 7.19. Operación 14: bilateral
cv2.imshow('Bilateral',blur_b)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Quinta parte: umbral simple a 1 imágen ###

# 7.20. Operación 9: color gray
cv2.imshow('Color GRAY', colored_image_gray)
cv2.waitKey(0)

# 7.21. Operación 15: Binary
cv2.imshow('BINARY',thr1)
cv2.waitKey(0)

# 7.22. Operación 16: Binary_Inv
cv2.imshow('BINARY_INV',thr2)
cv2.waitKey(0)

# 7.23. Operación 17: Trunc
cv2.imshow('TRUNC',thr3)
cv2.waitKey(0)

# 7.24. Operación 18: Tozero
cv2.imshow('TOZERO',thr4)
cv2.waitKey(0)

# 7.25. Operación 19: Tozero_Inv
cv2.imshow('TOZERO_INV',thr5)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Sexta parte: umbral a 1 imágen ###

# 7.26. Operación 9: color gray
cv2.imshow('Color GRAY', colored_image_gray)
cv2.waitKey(0)

# 7.27. Operación 20: Binary
cv2.imshow('BINARY',thr1)
cv2.waitKey(0)

# 7.28. Operación 21: Adaptive Thresh Mean
cv2.imshow('ADAPTIVE_THRESH_MEAN',thr2)
cv2.waitKey(0)

# 7.29. Operación 22: Adaptive Thresh Gaussian
cv2.imshow('ADAPTIVE_THRESH_GAUSSIAN',thr3)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Séptima parte: otsu a imágen ###

# 7.30. Operación 9: color gray
cv2.imshow('Color GRAY', colored_image_gray)
cv2.waitKey(0)

# 7.31. Operación 23: otsu
images = [colored_image_gray, 0, thr1,
          colored_image_gray, 0, thr2,
          blur, 0, thr3]
titles = ['Imagen original','Histogram','Global Thresholding (v=127)',
          'Imagen con ruido','Histogram',"Otsu's Thresholding",
          'Imagen filtro Gaussiano','Histogram',"Otsu's Thresholding"]

for i in range(0, 3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()