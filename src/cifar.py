import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, Callback

# Configuración
TAMANO_IMG = 100
BATCH_SIZE = 32
EPOCHS = 60

# Clase personalizada para graficar en tiempo real
class TrainingPlot(Callback):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.ion()  # Modo interactivo

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.losses.append(logs['loss'])
        self.accuracies.append(logs['accuracy'])
        self.val_losses.append(logs['val_loss'])
        self.val_accuracies.append(logs['val_accuracy'])
        
        # Actualizar gráficas
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.plot(self.epochs, self.losses, label='Entrenamiento')
        self.ax1.plot(self.epochs, self.val_losses, label='Validación')
        self.ax1.set_title('Pérdida por Época')
        self.ax1.set_xlabel('Época')
        self.ax1.set_ylabel('Pérdida')
        self.ax1.legend()
        
        self.ax2.plot(self.epochs, self.accuracies, label='Entrenamiento')
        self.ax2.plot(self.epochs, self.val_accuracies, label='Validación')
        self.ax2.set_title('Precisión por Época')
        self.ax2.set_xlabel('Época')
        self.ax2.set_ylabel('Precisión')
        self.ax2.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

# Cargar imágenes desde tus carpetas
def cargar_datos(directorio):
    x = []
    y = []
    
    # Clase Marcador (etiqueta 0)
    marcador_count = 0
    for archivo in os.listdir(os.path.join(directorio, "Marcador")):
        imagen = cv2.imread(os.path.join(directorio, "Marcador", archivo))
        if imagen is not None:
            imagen = cv2.resize(imagen, (TAMANO_IMG, TAMANO_IMG))
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1)
            x.append(imagen)
            y.append(0)  # 0 para Marcador
            marcador_count += 1
    
    # Clase Lápiz (etiqueta 1)
    lapiz_count = 0
    for archivo in os.listdir(os.path.join(directorio, "Lapiz")):
        imagen = cv2.imread(os.path.join(directorio, "Lapiz", archivo))
        if imagen is not None:
            imagen = cv2.resize(imagen, (TAMANO_IMG, TAMANO_IMG))
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1)
            x.append(imagen)
            y.append(1)  # 1 para Lápiz
            lapiz_count += 1
    
    print(f"\n📊 Estadísticas del Dataset:")
    print(f"- Marcadores: {marcador_count} imágenes")
    print(f"- Lápices: {lapiz_count} imágenes")
    print(f"- Total: {marcador_count + lapiz_count} imágenes\n")
    
    return np.array(x), np.array(y)

# Cargar datos
x, y = cargar_datos('dataset')

# Normalizar imágenes
x = x.astype(float)/255

# Función para mostrar ejemplos balanceados
def mostrar_ejemplos_balanceados(x, y, num_ejemplos=12):
    # Separar índices por clase
    idx_marcador = np.where(y == 0)[0]
    idx_lapiz = np.where(y == 1)[0]
    
    # Seleccionar igual número de ejemplos para cada clase
    num_por_clase = min(num_ejemplos//2, len(idx_marcador), len(idx_lapiz))
    ejemplos_marcador = np.random.choice(idx_marcador, num_por_clase, replace=False)
    ejemplos_lapiz = np.random.choice(idx_lapiz, num_por_clase, replace=False)
    
    # Crear figura
    plt.figure(figsize=(12, 6))
    plt.suptitle('Ejemplos del Dataset', fontsize=16, y=1.05)
    
    # Mostrar marcadores
    for i, idx in enumerate(ejemplos_marcador):
        plt.subplot(2, num_por_clase, i+1)
        plt.imshow(x[idx].reshape(TAMANO_IMG, TAMANO_IMG), cmap='gray')
        plt.title("Marcador", pad=10)
        plt.axis('off')
    
    # Mostrar lápices
    for i, idx in enumerate(ejemplos_lapiz):
        plt.subplot(2, num_por_clase, num_por_clase+i+1)
        plt.imshow(x[idx].reshape(TAMANO_IMG, TAMANO_IMG), cmap='gray')
        plt.title("Lápiz", pad=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Mostrar ejemplos balanceados (12 en total: 6 de cada clase)
mostrar_ejemplos_balanceados(x, y, num_ejemplos=12)

# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(x)

# Dividir datos en entrenamiento y validación
split = int(len(x)*0.85)
x_entrenamiento = x[:split]
x_validacion = x[split:]
y_entrenamiento = y[:split]
y_validacion = y[split:]

# Modelo CNN
modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(TAMANO_IMG,TAMANO_IMG,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modeloCNN.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Callbacks
plot_callback = TrainingPlot()
tensorboard_cnn = TensorBoard(log_dir='logs/cnn')

# Entrenamiento con aumento de datos
data_gen_entrenamiento = datagen.flow(x_entrenamiento, y_entrenamiento, batch_size=BATCH_SIZE)

history = modeloCNN.fit(
    data_gen_entrenamiento,
    epochs=EPOCHS,
    validation_data=(x_validacion, y_validacion),
    steps_per_epoch=int(np.ceil(len(x_entrenamiento)/float(BATCH_SIZE))),
    validation_steps=int(np.ceil(len(x_validacion)/float(BATCH_SIZE))),
    callbacks=[plot_callback, tensorboard_cnn]
)

# Guardar modelo
modeloCNN.save('modelo_marcador_lapiz.h5')
print("\n✅ Modelo entrenado y guardado como 'modelo_marcador_lapiz.h5'")

# Gráficas finales de entrenamiento
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida por Época')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión por Época')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')  # Guardar las gráficas
plt.show()