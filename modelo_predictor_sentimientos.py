from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from pathlib import Path

carpeta_dataset = Path("dataset")

# Carpeta de imagenes de entrenamiento
train_dir = Path(carpeta_dataset, 'train')

# Carpeta de imagenes de prueba
test_dir = Path(carpeta_dataset, 'test')

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='sparse'
)

# Crear el modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')  
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_generator, epochs=400, validation_data=test_generator)

# Ruta del modelo
ruta_modelo = Path(carpeta_dataset.parent, 'modelo_predictor_sentimientos.keras')

model.save(ruta_modelo)
loss, accuracy = model.evaluate(test_generator)
print(f'Precisión: {accuracy}')
print('Clases: ',train_generator.class_indices)