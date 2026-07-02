import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ----------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Clasificador de Flores IA",
    layout="centered"
)

st.title("Clasificador de Tipos de Flores con IA - Joksan Zavala - 20201900395")
st.write("Sube la fotografía de una flor para que la Inteligencia Artificial identifique qué tipo es.")

# ----------------------------------------------------------------
# CARGAR EL MODELO ENTRENADO
# ----------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    # Carga el archivo h5 que generaste en tu Jupyter Notebook
    return tf.keras.models.load_model("modelo_flores.h5")

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error("No se pudo cargar el modelo 'modelo_flores.h5'. Asegúrate de que esté en la misma carpeta que este script.")
    st.stop()

# Diccionario para traducir y mostrar las clases de forma elegante
clases_espanol = {
    'daisy': 'Margarita',
    'dandelion': 'Diente de León',
    'rose': 'Rosa',
    'sunflower': 'Girasol',
    'tulip': 'Tulipán'
    }

# Lista ordenada de las clases de tu dataset (debe coincidir con el orden del entrenamiento)
# Por orden alfabético de las carpetas originales en Kaggle
clases_lista = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# ----------------------------------------------------------------
# INTERFAZ DE USUARIO - SUBIDA DE ARCHIVOS
# ----------------------------------------------------------------
archivo_subido = st.file_uploader("Elige una imagen de una flor...", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # 1. Mostrar la imagen en la app web
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption="Imagen subida por el usuario", use_container_width=True)
    
    st.write("Analizando la imagen...")
    
    # 2. Preprocesamiento de la imagen (Idéntico al del cuaderno)
    img_rgb = imagen.convert("RGB")
    img_redimensionada = cv2.resize(np.array(img_rgb), (224, 224))
    img_normalizada = img_redimensionada.astype(float) / 255.0
    
    # 3. Predicción
    prediccion = modelo.predict(img_normalizada.reshape(-1, 224, 224, 3))
    indice_maximo = np.argmax(prediccion[0], axis=-1)
    
    nombre_ingles = clases_lista[indice_maximo]
    nombre_final = clases_espanol.get(nombre_ingles, nombre_ingles)
    porcentaje_confianza = prediccion[0][indice_maximo] * 100
    
    # 4. Mostrar Resultados Exitosos
    st.success(f"### ¡Predicción Completada!")
    st.metric(label="Flor Detectada", value=nombre_final)
    st.info(f"Nivel de confianza de la IA: **{porcentaje_confianza:.2f}%**")
