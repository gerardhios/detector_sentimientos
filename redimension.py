import os
from pathlib import Path
from PIL import Image
import dlib
import cv2
import matplotlib.pyplot as plt

detector_caras = dlib.get_frontal_face_detector()

# Ruta de la carpeta raiz del dataset
carpeta_dataset = Path("dataset")

# Ruta de la carpeta con las imágenes originales
carpeta_imagenes_originales = Path(carpeta_dataset, "original")

# Obtener la lista de los nombres de las carpetas dentro de la carpeta de imágenes originales
carpetas = [carpeta for carpeta in carpeta_imagenes_originales.iterdir() if carpeta.is_dir()]

# Lista para almacenar las imágenes redimensionadas
imagenes_redimensionadas = []

# Iterar sobre las carpetas y redimensionar las imágenes pasando 30% de las imagenes originales a una carpeta test y el resto a train
for carpeta in carpetas:
    # Obtener la lista de archivos en la carpeta con Path
    archivos = os.listdir(carpeta)
    
    # Calcular la cantidad de imágenes que se van a mover a la carpeta de test
    cantidad_test = int(len(archivos) * 0.3)
    
    # Iterar sobre los archivos
    for i, archivo in enumerate(archivos):
        # Crear la ruta del archivo
        ruta_archivo = Path(carpeta, archivo)
        
        # Crear la ruta de destino
        if i < cantidad_test:
            carpeta_destino = "test"
        else:
            carpeta_destino = "train"
        
        ruta_carpeta_destino = Path(carpeta_dataset, carpeta_destino, carpeta.name)
        ruta_destino = Path(ruta_carpeta_destino, archivo)

        # Revizar si la carpeta destino existe, si no, crearla
        if not ruta_carpeta_destino.exists():
            ruta_carpeta_destino.mkdir(parents=True)
        
        # Abrir la imagen original
        imagen_original = Image.open(ruta_archivo)

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(cv2.imread(str(ruta_archivo)), cv2.COLOR_BGR2GRAY)

        # Detectar caras en la imagen
        cara = detector_caras(gray)

        # Recortar la región de la cara de la imagen original con un margen de 20 pixeles
        x, y, w, h = cara[0].left(), cara[0].top(), cara[0].width(), cara[0].height()
        recorte = imagen_original.crop((x - 160, y - 160, x + w + 160, y + h + 160))

        # Redimensionar la imagen recortada a 64x64 pixeles
        imagen_redimensionada = recorte.resize((64, 64))

        # Guardar la imagen redimensionada en la carpeta de destino
        imagen_redimensionada.save(ruta_destino)

        imagenes_redimensionadas.append(imagen_redimensionada)

        # Cerrar la imagen original
        imagen_original.close()

print(f"Se redimensionaron {len(imagenes_redimensionadas)} imágenes")
# Mostrar las imágenes redimensionadas en una cuadrícula de 5x5
fig, axs = plt.subplots(5, 5, figsize=(15, 15))
for i, ax in enumerate(axs.flat):
    ax.imshow(imagenes_redimensionadas[i])
    ax.axis("off")
plt.show()