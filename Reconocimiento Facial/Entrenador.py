import os
import numpy as np
from PIL import Image
import cv2
import pickle

# Cargar clasificadores Haar y LBP
faceCascadeHaar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Clasificador Haar para rostro
faceCascadeLBP = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")  # Clasificador LBP para rostro
eyeTreeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  # Clasificador de ojos
eyeGlassesCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")  # Clasificador de ojos con gafas

# Verificar si los clasificadores se cargaron correctamente
for cascade in [faceCascadeHaar, faceCascadeLBP, eyeTreeCascade, eyeGlassesCascade]:
    if cascade.empty():
        print("Error: No se pudo cargar uno de los clasificadores Haar.")
        exit()

# Inicializar el reconocedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Directorio donde se almacenan las imágenes
baseDir = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(baseDir, "images")

# Cargar las etiquetas de los usuarios desde el archivo pickle
if os.path.exists("labels"):
    with open("labels", "rb") as f:
        labelIds = pickle.load(f)
else:
    labelIds = {}

# Cargar el modelo entrenado si existe
if os.path.exists("trainer.yml"):
    recognizer.read("trainer.yml")
    print("Modelo cargado.")
else:
    print("No se encontró un modelo entrenado previamente.")

# Lista de imágenes procesadas anteriormente (recuperar los datos existentes del modelo)
processed_images = set()

# Verificar si existe un archivo que registre las imágenes procesadas
if os.path.exists("processed_images.txt"):
    with open("processed_images.txt", "r") as f:
        processed_images = set(f.read().splitlines())  # Cargar las imágenes procesadas previamente

currentId = max(labelIds.values(), default=1)  # Iniciar desde el ID más alto existente

yLabels = []
xTrain = []

# Recorrer el directorio de imágenes
for root, dirs, files in os.walk(imageDir):
    print(f"Procesando directorio: {root}")
    for file in files:
        print(f"Procesando archivo: {file}")
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root)

            # Si el label no existe, crear un nuevo ID
            if label not in labelIds:
                labelIds[label] = currentId
                currentId += 1

            id_ = labelIds[label]
            pilImage = Image.open(path).convert("L")  # Convertir la imagen a escala de grises
            imageArray = np.array(pilImage, "uint8")

            # Verificar si la imagen ya fue procesada
            if path in processed_images:
                print(f"Imagen {path} ya procesada, se omite.")
                continue  # Si ya ha sido procesada, la omitimos

            # Detectar rostros en la imagen utilizando ambos clasificadores (Haar y LBP)
            faces_haar = faceCascadeHaar.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5)
            faces_lbp = faceCascadeLBP.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5)

            # Combinar las detecciones de Haar y LBP (sin duplicados)
            faces = []

            # Añadir rostros detectados por Haar
            for (x, y, w, h) in faces_haar:
                faces.append((x, y, w, h))

            # Añadir rostros detectados por LBP si no están ya en faces
            for (x, y, w, h) in faces_lbp:
                if not any([x == fx and y == fy and w == fw and h == fh for (fx, fy, fw, fh) in faces]):
                    faces.append((x, y, w, h))

            print(f"Rostros detectados: {len(faces)}")  # Ver cuántos rostros fueron detectados

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Extraer la región del rostro en escala de grises
                    roiGray = imageArray[y:y+h, x:x+w]

                    # Detectar los ojos dentro del rostro utilizando el clasificador de ojos
                    eyes = eyeTreeCascade.detectMultiScale(roiGray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
                    if len(eyes) == 0:
                        eyes = eyeGlassesCascade.detectMultiScale(roiGray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

                    # Si detectamos ojos, consideramos que es una detección válida de rostro
                    if len(eyes) > 0:
                        xTrain.append(roiGray)
                        yLabels.append(id_)

                        # Registrar la imagen como procesada
                        processed_images.add(path)

# Guardar las etiquetas en un archivo pickle
with open("labels", "wb") as f:
    pickle.dump(labelIds, f)

# Guardar las imágenes procesadas en un archivo de texto
with open("processed_images.txt", "w") as f:
    f.write("\n".join(processed_images))  # Guardar las imágenes procesadas

# Entrenar el modelo
recognizer.train(xTrain, np.array(yLabels))
recognizer.save("trainer.yml")

print(f"Entrenamiento completado.")
print(f"Etiquetas guardadas: {labelIds}")
