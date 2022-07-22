import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image
from modelo.unet import unet
from proceso import imagenProceso, removerAreas, aumentoTam, cuadrarRect
def saludo():
    st.header(" Anest App")

def camara():
    #picture = st.camera_input("Take a picture")
    uploaded_file = st.file_uploader("Suba la foto")

    if uploaded_file is not None:

        st.image(uploaded_file.getvalue())
        imagen = cv.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv.IMREAD_GRAYSCALE)

        # Lectura del Archivo
        # st.write("filename:", uploaded_file.name)
        #img = Image.open(picture)
        img_array = np.array(imagen)
        #print(img_array.shape)
        #st.image(picture)
        modelo = unet()
        modelo.load_weights('pesos/pesosBalanceBlancos.h5')
        p = imagenProceso(img_array)
        prediccion = modelo.predict(p)
        #print(prediccion.min(), prediccion.max(), sum(prediccion))
        aux = prediccion < 1.0
        prediccion[aux] = 0
        prediccion = prediccion[0, :, :, 0]
        #print(prediccion.shape)
        dd = removerAreas(prediccion)
        n = np.round(aumentoTam(dd, img_array.shape))
        cc = cuadrarRect(n)
        ee = np.multiply(cc, img_array)/255.0
        print(ee.max())
        print(ee.min())
        #print(n.shape)
        st.image(ee)

        btn = st.button("Descargar Imagen")
        if btn:
            cv.imwrite('imagen.png', np.uint8(255*ee))
            print("ya")
        #descarga = st.button("Download")
        #im = Image.fromarray(ee * 255)
        #print(prediccion)




saludo()
camara()