import cv2
from picamera2 import Picamera2
import numpy as np
import os
import sys

# Verificar si los archivos Haar Cascade existen
face_cascade_path_haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # Clasificador Haar para rostros
face_cascade_path_lbp = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"  # Clasificador LBP para rostros
eye_tree_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"  # Clasificador de ojos
eye_glasses_cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"  # Clasificador de ojos con gafas

# Verificar si los archivos existen
for xml_path in [face_cascade_path_haar, face_cascade_path_lbp, eye_tree_cascade_path, eye_glasses_cascade_path]:
    if not os.path.exists(xml_path):
        print(f"Error: El archivo '{xml_path}' no se encuentra en el directorio.")
        sys.exit()

# Cargar los clasificadores Haar y LBP
faceCascadeHaar = cv2.CascadeClassifier(face_cascade_path_haar)
faceCascadeLBP = cv2.CascadeClassifier(face_cascade_path_lbp)
eyeTreeCascade = cv2.CascadeClassifier(eye_tree_cascade_path)
eyeGlassesCascade = cv2.CascadeClassifier(eye_glasses_cascade_path)

# Verificar si los clasificadores se cargaron correctamente
for cascade in [faceCascadeHaar, faceCascadeLBP, eyeTreeCascade, eyeGlassesCascade]:
    if cascade.empty():
        print("Error: No se pudo cargar uno de los clasificadores Haar.")
        sys.exit()

# Configuración de la cámara
camera = Picamera2()
camera.configure("main")  # "main" es una de las configuraciones predefinidas

# Cambiar la resolución de la cámara a 720p (1280x720)
camera.camera_controls['Resolution'] = (1280, 720)  # Asegura que la resolución esté configurada a 720p
camera.start()

# Pedir el nombre al usuario y crear el directorio
name = input("¿Cual es su nombre?")
dirName = "./images/" + name
print(f"Directorio: {dirName}")

# Crear el directorio si no existe
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directorio creado.")
else:
    print("El nombre ya existe.")
    sys.exit()

captured_images = []  # Lista para almacenar las imágenes y su calidad

# Función para medir la nitidez de una imagen
def measure_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

count = 1

# Bucle para capturar 60 imágenes
while count <= 60:
    # Capturar la imagen desde la cámara
    frame = camera.capture_array()

    # Mostrar la imagen capturada para depuración
    cv2.imshow("Captura de imagen", frame)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar los rostros en la imagen utilizando el clasificador Haar
    faces_haar = faceCascadeHaar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    # Detectar los rostros en la imagen utilizando el clasificador LBP
    faces_lbp = faceCascadeLBP.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

    # Combinar las detecciones de Haar y LBP
    faces = []

    # Añadir rostros detectados por Haar
    for (x, y, w, h) in faces_haar:
        faces.append((x, y, w, h))

    # Añadir rostros detectados por LBP si no están ya en faces
    for (x, y, w, h) in faces_lbp:
        # Evitar duplicados (rostros ya detectados por Haar)
        if not any([x == fx and y == fy and w == fw and h == fh for (fx, fy, fw, fh) in faces]):
            faces.append((x, y, w, h))

    print(f"Rostros detectados: {len(faces)}")  # Ver cuántos rostros fueron detectados

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extraer la región del rostro en escala de grises
            roiGray = gray[y:y+h, x:x+w]

            # Detectar los ojos dentro del rostro utilizando el clasificador de ojos
            eyes = eyeTreeCascade.detectMultiScale(roiGray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if len(eyes) == 0:
                eyes = eyeGlassesCascade.detectMultiScale(roiGray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

            # Si se detectan ojos, se considera que es un rostro real
            if len(eyes) >= 2:  # Asegurarse de que se detectan al menos dos ojos
                if roiGray.shape[0] >= 30 and roiGray.shape[1] >= 30:  # Tamaño mínimo ajustable
                    # Medir la nitidez de la imagen
                    sharpness = measure_sharpness(frame)

                    # Añadir la imagen y su nitidez a la lista
                    captured_images.append((sharpness, frame, f"{dirName}/face_{count}.jpg"))
                    print(f"Imagen de rostro {count} capturada con nitidez {sharpness}.")
                    count += 1
                else:
                    print(f"Error: La región del rostro es demasiado pequeña para guardar imagen {count}.")
            else:
                print("No se detectaron suficientes ojos dentro del rostro.")
    else:
        print("No se detectaron rostros en este frame.")

    # Esperar brevemente (opcional, para no sobrecargar el procesador)
    cv2.waitKey(1)

# Ordenar las imágenes por su nitidez (de mayor a menor)
captured_images.sort(key=lambda x: x[0], reverse=True)

# Guardar las 30 mejores imágenes
for i in range(min(30, len(captured_images))):
    sharpness, img, path = captured_images[i]
    cv2.imwrite(path, img)
    print(f"Imagen de rostro {i+1} guardada en {path}")

print("Proceso finalizado.")
