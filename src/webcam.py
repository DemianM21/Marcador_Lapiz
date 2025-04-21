import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar modelo
model = load_model('modelo_marcador_lapiz.h5')

# Configuración de la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocesar la imagen
    resized = cv2.resize(frame, (100, 100))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray.reshape(100, 100, 1) / 255.0
    input_img = np.expand_dims(normalized, axis=0)
    
    # Predecir
    prediction = model.predict(input_img)
    class_id = int(prediction[0][0] > 0.5)
    label = "Marcador" if class_id == 0 else "Lapiz"
    confidence = prediction[0][0] if class_id == 1 else 1 - prediction[0][0]
    
    # Mostrar resultado
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Clasificador Marcador vs Lapiz", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()