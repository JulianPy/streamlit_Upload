import streamlit as st
import numpy as np
import cv2 as cv
from modelo.unet import unet
from proceso import imagenProceso, removerAreas, aumentoTam, cuadrarRect


def saludo():
    # Título de la App
    st.header("Anest App")
    # Descripción del aplicativo
    texto = """ Esta aplicación permite extraer la información relevante de los 
    dispositivos de ul ultrasonido  
    """
    st.write(texto)


def camara():
    # Cargar imagen o tomar foto
    uploaded_file = st.file_uploader("Cargar o Tomar la Foto")

    if uploaded_file is not None:
        # Extraer la imagen en formato Bytes
        st.image(uploaded_file.getvalue())
        # Decodificar la imagen para ser  leida como una lista
        imagen = cv.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv.IMREAD_GRAYSCALE)
        # Convertir la lista en array
        img_array = np.array(imagen)
        # Creación del modelo
        modelo = unet()
        # Cargar los pesos pre-entrenados del modelo
        modelo.load_weights('pesos/pesosBalanceBlancos.h5')
        # Procesar la imagen-array
        p = imagenProceso(img_array)
        # Pasar la imagen procesada a la etapa de inferencia
        prediccion = modelo.predict(p)
        # Limitar la predicción
        aux = prediccion < 1.0
        prediccion[aux] = 0
        # Pasar de un tensor-imagen a una imagen que se pueda mostrar
        prediccion = prediccion[0, :, :, 0]
        # Eliminar areas pequeñas de la imagen
        dd = removerAreas(prediccion)
        # Redondear los valores del preproces anterior
        n = np.round(aumentoTam(dd, img_array.shape))
        # Calcular el rectángulo que encierra la predicción
        cc = cuadrarRect(n)
        # Multiplicar el rectángulo con la imagen original
        ee = np.multiply(cc, img_array) / 255.0
        # Mostrar la imagen
        st.image(ee)


saludo()
camara()
