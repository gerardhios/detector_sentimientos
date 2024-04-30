import os
from pathlib import Path
from PIL import Image

# Ruta de la carpeta raiz del dataset
carpeta_dataset = Path("dataset")

# Ruta de la carpeta con las imágenes originales
carpeta_imagenes_originales = Path(carpeta_dataset, "original")

# Obtener la lista de los nombres de las carpetas dentro de la carpeta de imágenes originales
carpetas = [carpeta for carpeta in carpeta_imagenes_originales.iterdir() if carpeta.is_dir()]

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

        # Redimensionar la imagen a 64x64 píxeles
        imagen_redimensionada = imagen_original.resize((64, 64))

        # Guardar la imagen redimensionada en la carpeta de destino
        imagen_redimensionada.save(ruta_destino)

        # Cerrar la imagen original
        imagen_original.close()
