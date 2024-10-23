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

# Función para obtener la posición 3D usando la matriz de calibración
def obtener_posicion_3d(P, u, v):
    # Usar Z=1 porque se asume que el centro del bounding box está a esta altura sobre el suelo
    z = 1.0  

    # Crear el vector homogéneo
    punto_2D_homogeneo = np.array([u, v, 1])  # Coordenadas 2D

    # Obtener la inversa de la matriz de proyección (P)
    P_inv = np.linalg.pinv(P)  # Usamos pseudo-inversa para estabilidad numérica

    # Estimar el punto 3D en coordenadas homogéneas
    punto_3D_homogeneo = P_inv @ (punto_2D_homogeneo * z)

    # Convertir de coordenadas homogéneas a coordenadas 3D
    X = punto_3D_homogeneo[0] / punto_3D_homogeneo[3]
    Y = punto_3D_homogeneo[1] / punto_3D_homogeneo[3]
    Z = punto_3D_homogeneo[2] / punto_3D_homogeneo[3]  # Aquí Z será 1 si z=1

    return X, Y, Z



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
image_path = '/home/esteban/universidad/curso/datasets/subset_kitti/image_2/000013.png'
archivo_calibracion = '/home/esteban/universidad/curso/datasets/subset_kitti/calib/000013.txt'

# Leer la imagen usando OpenCV
img = cv2.imread(image_path)

# Leer la matriz de calibración desde el archivo
matrices_calibracion = leer_matriz_calibracion(archivo_calibracion)
P2 = matrices_calibracion['P2']  # Usamos la matriz de la cámara 2


# Realizar la detección
results = model.predict(img)


# Acceder a los resultados
for r in results:
    for box in r.boxes:  # Recorrer cada detección (bounding box)
        # Mover el tensor a la CPU antes de convertir a NumPy
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

        # Calcular el centro del bounding box
        u = (x_min + x_max) / 2  # Centro en el eje X
        v = (y_min + y_max) / 2  # Centro en el eje Y

        # Imprimir las coordenadas del bounding box y su centro
        print(f"Bounding box: ({x_min}, {y_min}), ({x_max}, {y_max})")
        print(f"Centro del bounding box (u, v): ({u}, {v})")
        
        # Estimar la posición 3D usando la matriz de calibración y Z = 0
        X, Y, Z = obtener_posicion_3d(P2, u, v)
        print(f"Posición 3D estimada: X={X}, Y={Y}, Z={Z}")
        
        # Calcular la distancia a la cámara solo en base a X e Y
        distancia = np.sqrt(X**2 + Y**2)
        print(f"Distancia estimada a la cámara: {distancia:.2f} metros")
        
        # Puedes dibujar el bounding box en la imagen (opcional)
        img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        img = cv2.circle(img, (int(u), int(v)), 5, (255, 0, 0), -1)  # Dibujar el centro del bounding box

# Mostrar la imagen con los bounding boxes dibujados
cv2.imshow("Detecciones YOLOv8", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
