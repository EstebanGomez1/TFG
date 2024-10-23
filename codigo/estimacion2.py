import torch
from ultralytics import YOLO
import cv2
import numpy as np

# Función para leer la matriz de calibración desde un archivo
def leer_matriz_calibracion(archivo_calibracion):
    matrices = {}
    with open(archivo_calibracion, 'r') as f:
        for line in f.readlines():
            line = line.strip()  # Eliminar espacios en blanco
            if not line or ':' not in line:  # Ignorar líneas vacías o incorrectas
                continue
            
            key, value = line.split(':', 1)
            value = np.array([float(x) for x in value.split()])
            if key.startswith('P'):  # Matrices P0, P1, P2, P3
                matrices[key] = value.reshape(3, 4)
            elif key == 'R0_rect':  # Matriz de rotación
                matrices[key] = value.reshape(3, 3)
            elif key.startswith('Tr_'):  # Matrices de transformación
                matrices[key] = value.reshape(3, 4)
    return matrices

def calcular_distancia(P0, altura_objeto, u, h):
    """
    Calcula la distancia Z desde la cámara al objeto usando la matriz de calibración P0.
    
    Args:
        P0 (np.ndarray): Matriz de calibración P0.
        altura_objeto (float): Altura real del objeto sobre el nivel del suelo.
        u (float): Coordenada en el eje X del centro del bounding box.
        h (float): Altura del bounding box en píxeles.

    Returns:
        float: Distancia Z a la cámara en metros.
    """
    # Extraer la distancia focal de P0
    f = P0[0, 0]  # Focal en píxeles

    # Calcular la distancia Z
    Z = (altura_objeto * f) / h
    return Z



def proyectar_y_dibujar_posicion_3D(img, P0, X, Y, Z, u, v):
    # Implementación de la función para proyectar y dibujar la posición 3D
    punto_3D = np.array([X, Y, Z, 1])  # Añadir 1 para la transformación homogénea
    punto_2D = P0 @ punto_3D  # Multiplicación matricial

    # Normalizar las coordenadas
    punto_2D /= punto_2D[2]  # Dividir por Z para obtener las coordenadas en 2D

    # Dibujar el punto proyectado en la imagen
    img = cv2.circle(img, (int(punto_2D[0]), int(punto_2D[1])), 5, (0, 0, 255), -1)  # Proyección 3D
    img = cv2.circle(img, (int(u), int(v)), 5, (255, 0, 0), -1)  # Centro del bounding box

    return img

# Cargar el modelo YOLOv8 (preentrenado)
model = YOLO('yolov8n.pt')  # Usa 'yolov8n.pt' o el modelo que prefieras

# Ruta de la imagen y del archivo de calibración
image_path = '/home/esteban/universidad/curso/datasets/subset_kitti/image_2/000125.png'
archivo_calibracion = '/home/esteban/universidad/curso/datasets/subset_kitti/calib/000125.txt'

# Leer la imagen usando OpenCV
img = cv2.imread(image_path)

# Leer la matriz de calibración desde el archivo
matrices_calibracion = leer_matriz_calibracion(archivo_calibracion)
P0 = matrices_calibracion['P0']  # Usamos la matriz de la cámara 0

# Altura real del objeto (en metros)
altura_objeto = 0.75 # Ajusta a 0.75 metros

# Realizar la detección
results = model.predict(img)

# Acceder a los resultados
for r in results:
    for box in r.boxes:  # Recorrer cada detección (bounding box)
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
        u = (x_min + x_max) / 2  # Centro en el eje X
        v = (y_min + y_max) / 2  # Centro en el eje Y
        
        # Calcular la altura del bounding box en píxeles
        h = y_max - y_min  # Altura del bounding box en píxeles

        # Calcular la distancia
        distancia = calcular_distancia(P0, altura_objeto, u, h)

        # Imprimir las coordenadas del bounding box y su centro
        print(f"Bounding box: ({x_min}, {y_min}), ({x_max}, {y_max})")
        print(f"Centro del bounding box (u, v): ({u}, {v})")
        print(f"Distancia estimada a la cámara: {distancia:.2f} metros")
        
        # Dibujar el bounding box y el centro del bounding box
        img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        img = cv2.circle(img, (int(u), int(v)), 5, (255, 0, 0), -1)

# Mostrar la imagen con los bounding boxes dibujados
cv2.imshow("Detecciones YOLOv8", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
