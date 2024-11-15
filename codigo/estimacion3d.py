"""
Usando las matrices de transformacion y P2
"""


import numpy as np
import torch
from ultralytics import YOLO
import cv2

# Función para leer la matriz de calibración desde el archivo y obtener la matriz P2
def leer_matriz_calibracion(archivo_calibracion):
    matriz_P2 = None
    with open(archivo_calibracion, 'r') as f:
        for line in f:
            if line.startswith("P2:"):
                datos = line.split()[1:]
                matriz_P2 = np.array(datos, dtype=np.float32).reshape(3, 4)
                break
    return matriz_P2

# Función para estimar la posición 3D
def estimar_posicion_3d(matriz_P2_inv, x_min, x_max, y_max, camera_height):
    # Calcular u y v (promedio de los límites horizontales y el límite inferior vertical)
    u = (x_min + x_max) * 0.5
    v = y_max
    
    # Coordenadas en la imagen (u, v, 1)
    coordenadas_imagen = np.array([u, v, 1])

    # Calcular p = P2_inv * coordenadas_imagen
    p = matriz_P2_inv @ coordenadas_imagen
    
    # Calcular el factor de escala k usando la altura de la cámara
    k = camera_height / p[1]
    
    # Estimar las coordenadas 3D
    X = p[0] * k
    Y = p[1] * k  # Altura de la cámara en el eje Y
    Z = p[2] * k
    
    return X, Y, Z

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt')

# Rutas de los archivos
image_path = '/home/esteban/universidad/curso/datasets/subset_kitti/image_2/000019.png'
archivo_calibracion = '/home/esteban/universidad/curso/datasets/subset_kitti/calib/000019.txt'
archivo_validacion = '/home/esteban/universidad/curso/datasets/subset_kitti/label_2/000019.txt'

# Cargar la imagen
img = cv2.imread(image_path)

# Verificar si la imagen se cargó correctamente
if img is None:
    print("Error: no se pudo cargar la imagen. Verifica la ruta del archivo.")
else:
    # Leer la matriz de calibración P2 y calcular su inversa
    matriz_P2 = leer_matriz_calibracion(archivo_calibracion)
    matriz_P2_inv = np.linalg.pinv(matriz_P2)

    # Altura de la cámara en metros (esto puede variar según el montaje de la cámara)
    camera_height = 1.77

    # Predecir objetos en la imagen
    results = model.predict(img)

    # Lista para almacenar estimaciones de posición 3D
    estimaciones = []

    # Iterar sobre cada detección y calcular la posición 3D
    for r in results:
        for box in r.boxes:
            clase = r.names[int(box.cls.item())]
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            # Calcular posición 3D
            X, Y, Z = estimar_posicion_3d(matriz_P2_inv, x_min, x_max, y_max, camera_height)
            
            # Guardar las estimaciones en la lista
            estimaciones.append((clase, X, Y, Z))
            
            # Imprimir las coordenadas 3D estimadas en la consola
            print(f"Clase: {clase}, X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}")

            # Dibujar el bounding box y la estimación de distancia en la imagen
            img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            img = cv2.putText(img, f"{clase} Dist: {Z:.2f}m", (int(x_min), int(y_min - 10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar la imagen con los bounding boxes y las distancias
    cv2.imshow("Detecciones y Estimaciones 3D", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
