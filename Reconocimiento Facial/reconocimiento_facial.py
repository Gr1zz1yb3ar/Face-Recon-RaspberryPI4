import cv2
from picamera2 import Picamera2
import numpy as np
import pickle
import RPi.GPIO as GPIO
from time import sleep, time
import datetime

# Configuración de los pines GPIO
relay_pin = [26]
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay_pin, GPIO.OUT)
GPIO.output(relay_pin, 0)

# Cargar las etiquetas de los usuarios
with open('labels', 'rb') as f:
    dicti = pickle.load(f)

# Configuración de la cámara usando Picamera2
camera = Picamera2()
camera.configure(camera.create_still_configuration())  # Configura la cámara para capturas fijas
camera.start()

# Cargar clasificadores Haar y LBP para rostros y ojos
faceCascadeHaar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Haar para rostros
faceCascadeLBP = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")  # LBP para rostros
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  # Cargar el clasificador de ojos
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

font = cv2.FONT_HERSHEY_SIMPLEX

# Diccionario para almacenar el último tiempo de detección de cada rostro
last_detection_time = {}

# Reducción de la resolución de la imagen para mejorar el rendimiento
width = 640  # Ajusta el ancho según lo que desees
height = 480  # Ajusta la altura según lo que desees

# Controlar la frecuencia de la detección (detectaremos un rostro cada 5 cuadros)
frame_count = 0
frame_interval = 5

# Umbral de confianza para el reconocimiento facial
CONFIDENCE_THRESHOLD = 50  # Ajusta el umbral para evitar falsas detecciones

# Variables para gestionar el tiempo de activación del relé
relay_active_time = None  # Hora en que se activó el relé
relay_duration = 5  # Duración en segundos que el relé debe permanecer activo
relays_active = False  # Estado de si el relé está activo

# Bucle de captura y procesamiento de imágenes
while True:
    # Capturar la imagen
    frame = camera.capture_array()  # Usar capture_array para obtener la imagen en formato numpy
    frame = cv2.resize(frame, (width, height))  # Reducir la resolución para mejorar el rendimiento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Incrementar el contador de frames
    frame_count += 1

    if frame_count % frame_interval == 0:  # Solo procesar cada 'frame_interval' frames
        # Detectar rostros usando el clasificador Haar
        facesHaar = faceCascadeHaar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_face = False
        for (x, y, w, h) in facesHaar:
            roiGray = gray[y:y+h, x:x+w]

            # Validar la detección con el clasificador LBP
            facesLBP = faceCascadeLBP.detectMultiScale(roiGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(facesLBP) > 0:  # Si LBP también detecta un rostro
                # Detectar los ojos dentro del rostro para verificar que es un rostro real
                eyes = eyeCascade.detectMultiScale(roiGray)

                # Filtrar falsas detecciones (si no hay ojos, es posible que no sea un rostro real)
                if len(eyes) > 0:
                    id_, conf = recognizer.predict(roiGray)

                    # Si la confianza es mayor que el umbral, se marca como "Desconocido"
                    if conf > CONFIDENCE_THRESHOLD:
                        name = "Desconocido"
                    else:
                        # Buscar el nombre correspondiente al ID
                        name = "Desconocido"
                        for n, value in dicti.items():
                            if value == id_:
                                name = n

                    # Si el rostro es reconocido (conf <= CONFIDENCE_THRESHOLD y nombre no es "Desconocido")
                    if name != "Desconocido":
                        detected_face = True
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f" {name}", (x, y), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

                        # Si el relé no está activo, activarlo
                        if not relays_active:
                            GPIO.output(relay_pin, 1)  # Activar el relé
                            relay_active_time = time()  # Guardar el tiempo en que se activó el relé
                            relays_active = True  # Marcar que el relé está activo

                        # Verificar si han pasado más de 60 segundos desde la última detección del mismo rostro
                        current_time = time()
                        if name not in last_detection_time or (current_time - last_detection_time[name] > 60):
                            last_detection_time[name] = current_time  # Actualizar el tiempo de la última detección

                            # Guardar nombre, fecha y hora en un archivo de texto
                            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with open("detecciones.txt", "a") as file:
                                file.write(f"{name}, {current_datetime}\n")
                            print(f"Guardado: {name}, {current_datetime}")
                    else:
                        # Si el rostro no está reconocido o la confianza es baja, no hacer nada con el relé
                        pass

        # Verificar si el relé debe apagarse después de 5 segundos
        if relays_active and (time() - relay_active_time) >= relay_duration:
            GPIO.output(relay_pin, 0)  # Desactivar el relé
            relays_active = False  # Marcar que el relé está apagado

    # Mostrar la imagen con los rostros detectados
    cv2.imshow('frame', frame)

    # Esperar 15ms antes de mostrar el siguiente frame (para fluidez)
    key = cv2.waitKey(15)  # Cambiar el tiempo de espera a 15ms para mejorar la fluidez

    if key == 27:  # Presionar ESC para salir
        break

# Liberar recursos al finalizar
cv2.destroyAllWindows()
GPIO.cleanup()
