# 🦺 Detección de Equipos de Protección Personal (PPE) con YOLOv8

**Autor:** Juan Diego Chaparro Garcia  
**Modelo base:** YOLOv8n (Ultralytics)  
**Dataset:** PPE Factory — Roboflow  
**Demo:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://TU-LINK-AQUI)

---

## 📋 Descripción

Este proyecto implementa un sistema de detección automática de Equipos de Protección Personal (EPP) en entornos industriales usando el modelo de detección de objetos **YOLOv8**. El modelo es entrenado mediante Transfer Learning sobre un dataset etiquetado obtenido desde Roboflow, y es capaz de detectar EPP en imágenes y videos en tiempo real.

---

## 🚀 Tecnologías utilizadas

| Herramienta | Versión | Uso |
|---|---|---|
| Python | 3.12 | Lenguaje principal |
| Ultralytics | 8.4+ | Framework YOLOv8 |
| Roboflow | 1.3+ | Gestión y descarga del dataset |
| PyTorch | 2.10+ | Backend de entrenamiento |
| ONNX | — | Exportación del modelo |
| Google Colab | — | Entorno de ejecución (GPU T4) |

---

## 📁 Estructura del proyecto

```
ppe-factory-3/
├── train/
│   └── images/        # Imágenes de entrenamiento
├── valid/
│   └── images/        # Imágenes de validación
├── test/
│   └── images/        # Imágenes de prueba
└── data.yaml          # Configuración del dataset y clases

runs/
└── ppe_detector/
    └── weights/
        ├── best.pt    # Mejor modelo entrenado (PyTorch)
        └── best.onnx  # Modelo exportado (ONNX)
```

---

## ⚙️ Instalación

```bash
pip install ultralytics roboflow
```

---

## 🔄 Flujo del notebook

1. **Instalación de dependencias** — `ultralytics` y `roboflow`
2. **Descarga del dataset** — Conexión a Roboflow y descarga en formato YOLOv8
3. **Conteo de imágenes** — Verificación de imágenes por partición (train/valid/test)
4. **Inspección de etiquetas** — Lectura de clases desde `data.yaml`
5. **Carga del modelo** — Modelo preentrenado `yolov8n.pt`
6. **Entrenamiento** — 50 épocas, imgsz=416, batch=16
7. **Carga del mejor modelo** — Checkpoint `best.pt`
8. **Validación** — Métricas mAP50, mAP50-95, Precisión y Recall
9. **Predicción en imagen** — Inferencia sobre imagen individual de test
10. **Predicción en batch** — Inferencia sobre todas las imágenes de test
11. **Descarga de video** — Video de ejemplo desde GitHub
12. **Predicción en video** — Inferencia y conversión a `.mp4` con ffmpeg
13. **Exportación a ONNX** — Formato portable para despliegue
14. **Descarga de modelos** — Descarga de `best.pt` y `best.onnx`

---

## 📊 Parámetros de entrenamiento

| Parámetro | Valor |
|---|---|
| Épocas | 50 |
| Tamaño de imagen | 416 × 416 |
| Batch size | 16 |
| Modelo base | yolov8n.pt |
| Optimizador | Auto |
| Dispositivo | GPU (CUDA) |

---

## 📈 Métricas de evaluación

El modelo es evaluado con las siguientes métricas estándar de detección de objetos:

- **mAP50** — Mean Average Precision con IoU threshold 0.50
- **mAP50-95** — mAP promedio con IoU de 0.50 a 0.95
- **Precisión** — Proporción de detecciones correctas
- **Recall** — Proporción de objetos reales detectados

---

## 💾 Exportación del modelo

El modelo entrenado puede exportarse a ONNX para su uso en distintas plataformas:

```python
model.export(format="onnx", opset=12)
```

---

## 📝 Notas

- Se requiere una API key de Roboflow para descargar el dataset.
- Se recomienda ejecutar en Google Colab con GPU habilitada (T4 o superior).
- El video de ejemplo puede descargarse directamente desde el repositorio de referencia en GitHub.
# Modelo-Predictivo-EPP-YOLO8
