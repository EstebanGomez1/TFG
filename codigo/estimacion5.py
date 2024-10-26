import torch
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# Función para leer la matriz de calibración desde un archivo
def leer_matriz_calibracion(archivo_calibracion):
    matrices = {}
    with open(archivo_calibracion, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            value = np.array([float(x) for x in value.split()])
            if key.startswith('P'):
                matrices[key] = value.reshape(3, 4)
            elif key == 'R0_rect':
                matrices[key] = value.reshape(3, 3)
            elif key.startswith('Tr_'):
                matrices[key] = value.reshape(3, 4)
    return matrices

# Función para leer el archivo de validación de KITTI
def leer_archivo_validacion(archivo_validacion):
    verdad_terreno = []
    with open(archivo_validacion, 'r') as f:
        for line in f.readlines():
            datos = line.split()
            clase = datos[0]
            if clase == 'Car' or clase == 'Truck':
                x_min = float(datos[4])
                y_min = float(datos[5])
                x_max = float(datos[6])
                y_max = float(datos[7])
                u_centro = (x_min + x_max) / 2
                v_centro = (y_min + y_max) / 2
                
                # Coordenadas 3D en el espacio
                x3d = float(datos[11])
                y3d = float(datos[12])
                z3d = float(datos[13])
                
                verdad_terreno.append((clase, x3d, y3d, z3d, x_min, y_min, x_max, y_max))
    return verdad_terreno

# Función para calcular IoU entre dos bounding boxes
def calcular_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# Función para calcular la distancia usando una matriz de calibración específica
def calcular_distancia(matriz_proyeccion, altura_imagen_px, altura_real):
    focal_length_y = matriz_proyeccion[1, 1]
    distancia = (altura_real * focal_length_y) / altura_imagen_px
    return distancia

# Función para calcular la posición X usando el centro del bounding box
def calcular_x(matriz_proyeccion, u, distancia_z):
    focal_length_x = matriz_proyeccion[0, 0]
    u0 = matriz_proyeccion[0, 2]
    x = ((u - u0) * distancia_z) / focal_length_x
    return x

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt')

# Rutas de los archivos
image_path = '/home/esteban/universidad/curso/datasets/subset_kitti/image_2/000007.png'
archivo_calibracion = '/home/esteban/universidad/curso/datasets/subset_kitti/calib/000007.txt'
archivo_validacion = '/home/esteban/universidad/curso/datasets/subset_kitti/label_2/000007.txt'

# Cargar la imagen
img = cv2.imread(image_path)
img_height = img.shape[0]

# Leer matrices de calibración
matrices_calibracion = leer_matriz_calibracion(archivo_calibracion)

# Predecir objetos en la imagen
results = model.predict(img)

# Leer verdad de terreno del archivo de validación
verdad_terreno = leer_archivo_validacion(archivo_validacion)

# Lista para almacenar estimaciones de posición 3D
estimaciones = []

# Contadores para IDs de cada clase
id_contadores = defaultdict(int)

# Procesar las detecciones
detecciones = []
for r in results:
    for box in r.boxes:
        clase = r.names[int(box.cls.item())]
        
        # Incrementar el contador de la clase para asignar el ID
        id_contadores[clase] += 1
        clase_id = f"{clase} {id_contadores[clase]}"
        
        # Establece la altura real según la clase detectada
        if clase == 'car':
            altura_real = 1.5
        elif clase == 'truck':
            altura_real = 2.5  
        elif clase == 'van':
            altura_real = 2
        elif clase == 'person':
            altura_real = 1.7 
        else:
            altura_real = 1
        
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
        altura_imagen_px = y_max - y_min
        
        # Calcular la distancia usando las matrices de calibración y obtener promedio
        distancias = [
            calcular_distancia(matrices_calibracion['P0'], altura_imagen_px, altura_real),
            calcular_distancia(matrices_calibracion['P1'], altura_imagen_px, altura_real),
            calcular_distancia(matrices_calibracion['P2'], altura_imagen_px, altura_real),
            calcular_distancia(matrices_calibracion['P3'], altura_imagen_px, altura_real)
        ]
        distancia_promedio = np.mean(distancias)
        
        # Calcular la posición X usando el centro del bounding box
        u = (x_min + x_max) / 2
        x_estimado = calcular_x(matrices_calibracion['P0'], u, distancia_promedio)
        
        # Almacenar la detección con bounding box y estimaciones
        deteccion = {
            "clase_id": clase_id,
            "x": x_estimado,
            "y": altura_real,
            "z": distancia_promedio,
            "bbox": (x_min, y_min, x_max, y_max)
        }
        detecciones.append(deteccion)
        print(f"Clase: {clase_id}, Estimación 3D: X={x_estimado:.2f}, Y={altura_real:.2f}, Z={distancia_promedio:.2f}")
        
        # Dibujar el bounding box y el ID en la imagen
        img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        img = cv2.putText(img, clase_id, (int(x_min), int(y_min - 10)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Comparar cada detección con cada validación usando IoU para encontrar coincidencias
asociaciones = []
for deteccion in detecciones:
    max_iou = 0
    mejor_validacion = None
    for validacion in verdad_terreno:
        iou = calcular_iou(deteccion['bbox'], validacion[4:8])
        if iou > max_iou:
            max_iou = iou
            mejor_validacion = validacion

    # Asociar detección con la validación de mayor IoU
    if mejor_validacion:
        asociaciones.append((deteccion, mejor_validacion))

# Calcular y mostrar el error por cada asociación
for i, (deteccion, validacion) in enumerate(asociaciones):
    error_x = deteccion['x'] - validacion[1]
    error_y = deteccion['y'] - validacion[2]
    error_z = deteccion['z'] - validacion[3]
    print(f"{deteccion['clase_id']}")
    print(f"Error en X: {error_x:.2f} metros")
    print(f"Error en Y: {error_y:.2f} metros")
    print(f"Error en Z: {error_z:.2f} metros")
    print('-' * 30)

# Mostrar la imagen con los bounding boxes dibujados
cv2.imshow("Detecciones YOLOv8", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
