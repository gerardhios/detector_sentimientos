import cv2
from pathlib import Path
from random import randint

# Ruta de la carpeta raiz del dataset
carpeta_dataset = Path("data")

# Ruta de la carpeta con las imágenes originales
carpeta_imagenes_originales = Path(carpeta_dataset, "original")

# Crear la carpeta de las imagenes originales si no existe
if not carpeta_imagenes_originales.exists():
    carpeta_imagenes_originales.mkdir(parents=True)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Solicitar al usuario que ingrese la emocion de la imagen
emocion = input("Ingrese la emoción de la imagen (Enojado, Feliz, Neutro, Triste, Sorprendido): ")

# Crear la carpeta de la emocion si no existe
carpeta_emocion = Path(carpeta_imagenes_originales, emocion)
if not carpeta_emocion.exists():
    carpeta_emocion.mkdir(parents=True)

# Solicitar el número de imágenes a capturar
num_imagenes = int(input("Ingrese el número de imágenes a capturar: "))

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()

    if ret:
        # Mostrar el fotograma
        cv2.imshow('Detector de emociones', frame)

        # Salir del bucle si se presiona 'c'
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

contador = 1
while True:

    # Genera dos numeros enteros aleatorios entre 0 y 20 y si son iguales se toma la foto
    aleatorio1 = randint(0, 20)
    aleatorio2 = randint(0, 20)

    if aleatorio1 == aleatorio2:
        # Capturar un fotograma de la cámara
        ret, frame = cap.read()

        if ret:
            # Mostrar el fotograma
            cv2.imshow('Captura de emociones', frame)

            # Guardar la imagen en la carpeta de la emocion
            ruta_imagen = Path(carpeta_emocion, f"{emocion}_{contador}.jpg")
            cv2.imwrite(str(ruta_imagen), frame)

            # Salir del bucle si se capturaron todas las imágenes
            if contador == num_imagenes:
                break

            # Incrementar el número de imágenes capturadas
            contador += 1

# Liberar la captura de video y cerrar la ventana
cap.release()
cv2.destroyAllWindows()