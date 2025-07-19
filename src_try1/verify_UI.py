import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog, Button, Label

# ========== CONFIGURACIÓN ==========

# Ruta de tu modelo
modelo_path = 'D:/Informacion/Desktop/Code/best_modelo_plantas.h5'

# Cargar modelo
modelo = load_model(modelo_path)
print("✅ Modelo cargado exitosamente.")

# Clases
clases = ['Common Duckweeds (Lemna minor)', 
          'Common Water Hyacinth (Eichornia crassipes)', 
          'Heartleaf False Pickerelweed (Monochoria korsakowii)', 
          'Water Lettuce (Pistia stratiotes)']

# ========== FUNCIÓN DE PREDICCIÓN ==========

def predecir_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)) # ajusta según tu input_shape

    img_array = np.expand_dims(img_resized, axis=0)
    img_array = img_array / 255.0

    predicciones = modelo.predict(img_array)
    clase_predicha_idx = np.argmax(predicciones)
    confianza = predicciones[0][clase_predicha_idx]
    clase_predicha = clases[clase_predicha_idx]

    # Mostrar resultado en ventana
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Predicción: {clase_predicha}\nConfianza: {confianza:.2f}")
    plt.show()

    # Guardar imagen con texto si deseas
    texto = f"{clase_predicha} ({confianza:.2f})"
    img_texto = cv2.putText(img_rgb.copy(), texto, (10,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.imwrite("resultado_prediccion.jpg", cv2.cvtColor(img_texto, cv2.COLOR_RGB2BGR))
    print(f"✅ Imagen procesada. Clase: {clase_predicha}, Confianza: {confianza:.2f}")

# ========== INTERFAZ TKINTER ==========

def elegir_imagen():
    archivo = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp")]
    )
    if archivo:
        predecir_imagen(archivo)

# Crear ventana
ventana = Tk()
ventana.title("Clasificador de Plantas IA")
ventana.geometry("400x200")

# Botón para elegir imagen
btn_elegir = Button(ventana, text="Elegir Imagen y Predecir", command=elegir_imagen, font=("Arial", 12))
btn_elegir.pack(pady=50)

# Etiqueta de instrucción
label = Label(ventana, text="Clasificador de Plantas IA", font=("Arial", 14))
label.pack()

# Ejecutar
ventana.mainloop()
