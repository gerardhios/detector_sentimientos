import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import dlib
from pathlib import Path

carpeta_dataset = Path("dataset")
# Ruta del modelo
ruta_modelo = Path(carpeta_dataset.parent, 'modelo_predictor_sentimientos.keras')
# Cargar el modelo
model = load_model(ruta_modelo)
emociones_dict = {0: 'Enojado', 1: 'Feliz', 2: 'Neutro', 4: 'Triste', 3: 'Sorprendido'}

# Inicializar el detector de caras de dlib
detector_caras = dlib.get_frontal_face_detector()

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()

    if ret:
        # Convertir el fotograma a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar caras en el fotograma
        emociones = detector_caras(gray)

        for emocion in emociones:
            x, y, w, h = emocion.left(), emocion.top(), emocion.width(), emocion.height()

            # Revisar que x, y, w y h sean mayores a 0
            if x < 0 or y < 0 or w < 0 or h < 0:
                continue

            # Revisar que 

            # Recortar la región de la cara
            recorte = frame[y:y+h, x:x+w]
            recorte = cv2.resize(recorte, (64, 64))
            recorte = recorte.astype("float") / 255.0
            recorte = img_to_array(recorte)
            recorte = np.expand_dims(recorte, axis=0)
            prediction = model.predict(recorte)
            predicted_emotion = emociones_dict[np.argmax(prediction)]
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar el fotograma
        cv2.imshow('Detector de emociones', frame)

        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la captura de video y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
