# 🖍️ Clasificador de Imágenes: Marcador vs Lápiz

En este proyecto se entrena una red neuronal convolucional (CNN) en TensorFlow para distinguir entre imágenes de **marcadores** y **lápices**. A partir de un conjunto de imágenes organizadas en carpetas (dentro del dataset), el modelo aprende a clasificar visualmente cada objeto, utilizando técnicas de preprocesamiento, aumento de datos y visualización dinámica de métricas.

---

## 🚀 Características principales

- ✅ Clasificación binaria de imágenes (`Marcador` vs `Lápiz`)
- 🧠 Red neuronal convolucional (CNN) con 3 capas convolucionales
- 🔁 Aumento de datos con rotaciones, zoom, desplazamiento, etc.
- 📊 Visualización de métricas (loss/accuracy) en tiempo real y al final
- 💾 Guardado del modelo entrenado (`modelo_marcador_lapiz.h5`)

---

## 🧱 Requisitos

```bash
pip install tensorflow opencv-python matplotlib
📁 Estructura esperada del dataset
Copiar
Editar
dataset/
├── Marcador/
│   ├── imagen1.jpg
│   ├── imagen2.jpg
│   └── ...
├── Lapiz/
│   ├── imagen1.jpg
│   ├── imagen2.jpg
│   └── ...
🧪 Ejecución del script
bash
Copiar
Editar
python main.py
Esto realizará los siguientes pasos automáticamente:

Carga y preprocesamiento de imágenes (escala de grises, normalización).

Visualización de ejemplos balanceados del dataset.

Aumento de datos y entrenamiento de la CNN.

Visualización de métricas por época.

Guardado del modelo entrenado y métricas.

🔎 Salidas generadas
🧠 modelo_marcador_lapiz.h5 — modelo entrenado listo para usar

📈 training_metrics.png — gráfica de pérdida y precisión por época

🖼️ Visualización de métricas
Ejemplo de cómo se verá la gráfica final guardada:

markdown
Copiar
Editar
Pérdida (loss) y Precisión (accuracy)
--------------------------------------
|       📉 loss       |     📈 accuracy     |
| ------------------ | ------------------ |
| Entrenamiento      | Entrenamiento      |
| Validación         | Validación         |
También se muestra en tiempo real mientras el modelo entrena.


# Cargar modelo entrenado
modelo = load_model('modelo_marcador_lapiz.h5')

# Predecir con una nueva imagen preprocesada (100x100, escala de grises, normalizada)
resultado = modelo.predict(imagen_preprocesada)
print("Marcador" if resultado < 0.5 else "Lápiz")

---
## 🚀 Conclusión
El modelo de entrenamiento utilizado fue un CCN con data aumentation, además se agrego un dropout de 0.5 para funcionar sin overfiting, pero fue necesario acercar el objeto a la webcam para que realize la detección con alta precisión. 