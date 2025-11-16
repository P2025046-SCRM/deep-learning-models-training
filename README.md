# Deep Learning Models Training

Repositorio para el entrenamiento y despliegue de modelos de deep learning para clasificación de residuos reciclables utilizando la arquitectura InceptionV3.

## Descripción

Este repositorio contiene el código utilizado para entrenar y desplegar los modelos de clasificación de residuos madereros (WFWaste) en dos capas:

- **Capa 1 (Binaria)**: Clasifica si un residuo es "Reciclable" o "NoReciclable"
- **Capa 2 (Multiclase)**: Clasifica los residuos reciclables en 4 categorías: "Biomasa", "Metales", "Plastico", "Retazos"

## Archivos del Repositorio

### 1. `score.py`

Script de scoring para Azure Machine Learning que implementa el pipeline de inferencia en producción. Este archivo contiene:

- **Función `init()`**: Carga los modelos entrenados al iniciar el servidor
  - Modelo binario (Capa 1): `binary_inceptionv3_model_ft_1.keras` (Fine-tuned)
  - Modelo multiclase (Capa 2): `inceptionv3_model_1.keras` (Feature Extraction)

- **Función `run()`**: Procesa las peticiones HTTP y enruta según el modo:
  - `prod`: Pipeline completo de dos capas (predicción L1, y si es reciclable, predicción L2)
  - `layer1`: Solo predicción de la capa 1 (binaria)
  - `layer2`: Solo predicción de la capa 2 (multiclase)
  - `health`: Verificación del estado del servicio

- **Características principales**:
  - Preprocesamiento de imágenes en formato base64
  - Redimensionamiento a 299x299 (tamaño de entrada de InceptionV3)
  - Almacenamiento de imágenes procesadas en Azure Blob Storage
  - Medición de tiempos de inferencia
  - Manejo de errores y validación de entrada con Pydantic

### 2. `Training_Pipeline_Binary_InceptionV3.ipynb`

Notebook de Jupyter para entrenar el modelo de clasificación binaria (Capa 1). Incluye:

- **Preprocesamiento de datos**:
  - División del dataset en train/validation/test (80/10/10)
  - Aplicación de pesos de clase para manejar desbalance de datos
  - Data augmentation (flips, rotaciones, zoom, contraste)

- **Entrenamiento con Transfer Learning**:
  - **Feature Extraction**: Entrenamiento con base de InceptionV3 congelada
  - **Fine-tuning**: Entrenamiento adicional descongelando la base de InceptionV3 con learning rate reducido (1e-5)

- **Evaluación**:
  - Métricas: Accuracy, Precision, Recall, AUC, F1-Score
  - Matriz de confusión
  - Curvas ROC
  - Reportes de clasificación

- **Resultados**:
  - **Modelo ganador: Fine-tuned**
  - El modelo fine-tuned (`binary_inceptionv3_model_ft_1.keras`) obtuvo mejores resultados que el modelo de Feature Extraction

### 3. `Training_Pipeline_InceptionV3.ipynb`

Notebook de Jupyter para entrenar el modelo de clasificación multiclase (Capa 2). Incluye:

- **Preprocesamiento de datos**:
  - División del dataset en train/validation/test (80/10/10)
  - Data augmentation (flips, rotaciones, zoom, contraste)
  - 4 clases: Biomasa, Metales, Plastico, Retazos

- **Entrenamiento con Transfer Learning**:
  - **Feature Extraction**: Entrenamiento con base de InceptionV3 congelada
  - **Fine-tuning**: Entrenamiento adicional descongelando la base de InceptionV3 con learning rate reducido (1e-5)

- **Evaluación**:
  - Métricas: Accuracy, Precision, Recall, AUC, F1-Score (macro)
  - Matriz de confusión
  - Curvas ROC multiclase
  - Reportes de clasificación por clase

- **Resultados**:
  - **Modelo ganador: Feature Extraction**
  - El modelo de Feature Extraction (`inceptionv3_model_1.keras`) obtuvo mejores resultados que el modelo Fine-tuned

## Arquitectura del Modelo

Ambos modelos utilizan **InceptionV3** como arquitectura base con las siguientes características:

- **Tamaño de entrada**: 299x299 píxeles
- **Preprocesamiento**: Preprocesamiento específico de InceptionV3
- **Clasificador personalizado**:
  - Global Average Pooling 2D
  - Dense layer (128 unidades, ReLU)
  - Dropout (0.2)
  - Dense layer de salida (1 unidad con sigmoid para binario, 4 unidades con softmax para multiclase)

## Pipeline de Inferencia

El pipeline de producción funciona de la siguiente manera:

1. **Recepción de imagen**: La imagen llega en formato base64
2. **Preprocesamiento**: Decodificación, redimensionamiento y normalización
3. **Capa 1 (Binaria)**: Clasificación Reciclable/NoReciclable
4. **Capa 2 (Multiclase)**: Si es Reciclable, clasificación en Biomasa/Metales/Plastico/Retazos
5. **Almacenamiento**: La imagen procesada se guarda en Azure Blob Storage
6. **Respuesta**: Retorna predicciones, confianzas, URL de la imagen y metadatos de tiempo

## Requisitos

- Python 3.x
- TensorFlow/Keras
- Azure Machine Learning SDK
- Azure Storage Blob
- Pydantic
- NumPy
- PIL (Pillow)

## Uso

### Entrenamiento

1. Abrir los notebooks en Google Colab o Jupyter
2. Configurar las rutas de los datasets
3. Ejecutar las celdas en orden
4. Los modelos entrenados se guardarán en las rutas especificadas

### Despliegue

El archivo `score.py` está diseñado para ser desplegado como un endpoint de Azure Machine Learning. El modelo debe estar empaquetado con la siguiente estructura:

```
package_v1/
├── 1stLayerClassification/
│   └── 1/
│       └── binary_inceptionv3_model_ft_1.keras
└── 2ndLayerClassification/
    └── 1/
        └── inceptionv3_model_1.keras
```

## Notas

- Los modelos fueron entrenados con datos desbalanceados, por lo que se utilizaron pesos de clase en el modelo binario
- El modelo binario utiliza Fine-tuned como estrategia ganadora
- El modelo multiclase utiliza Feature Extraction como estrategia ganadora
- Las imágenes procesadas se almacenan automáticamente en Azure Blob Storage para trazabilidad
