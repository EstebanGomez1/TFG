import torch
from ultralytics import YOLO
import cv2
import numpy as np

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
            if clase == 'Car' or clase == 'Truck':  # Solo nos interesan los coches y camiones
                x3d = float(datos[11])  # Coordenada X 3D
                y3d = float(datos[12])  # Coordenada Y 3D
                z3d = float(datos[13])  # Coordenada Z 3D
                verdad_terreno.append((clase, x3d, y3d, z3d))
    return verdad_terreno

# Cálculo refinado de la distancia con más información sobre la geometría de la cámara
def calcular_distancia(P0, altura_objeto, u, v, h, img_height):
    f = P0[0, 0]  # Focal en píxeles
    Z = (altura_objeto * f) / h

    # Ajuste basado en la distancia del centro de la imagen
    centro_imagen_v = img_height / 2  # Centro en el eje Y
    delta_v = v - centro_imagen_v
    Y = Z * delta_v / f  # Aproximación de la coordenada Y (altura en el mundo real)

    return Z, Y

# Calcular error por coordenada (X, Y, Z)
def calcular_error_por_coordenada(estimaciones, verdad_terreno):
    errores = []
    for estimacion, verdad in zip(estimaciones, verdad_terreno):
        error_x = estimacion[0] - verdad[1]
        error_y = estimacion[1] - verdad[2]
        error_z = estimacion[2] - verdad[3]
        errores.append((error_x, error_y, error_z))
    return errores

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt')

image_path = '/home/esteban/universidad/curso/datasets/subset_kitti/image_2/000002.png'
archivo_calibracion = '/home/esteban/universidad/curso/datasets/subset_kitti/calib/000002.txt'
archivo_validacion = '/home/esteban/universidad/curso/datasets/subset_kitti/label_2/000002.txt'

img = cv2.imread(image_path)
img_height = img.shape[0]

matrices_calibracion = leer_matriz_calibracion(archivo_calibracion)
P0 = matrices_calibracion['P0']

results = model.predict(img)

# Leer verdad de terreno del archivo de validación
verdad_terreno = leer_archivo_validacion(archivo_validacion)

# Estimaciones para comparar
estimaciones = []

for r in results:
    for box in r.boxes:
        clase = r.names[int(box.cls.item())]
        if clase == 'car':
            altura_objeto = 1.5
        elif clase == 'truck':
            altura_objeto = 2.5  
        elif clase == 'van':
            altura_objeto = 2
        elif clase == 'person':
            altura_objeto = 1.7 
        else:
            altura_objeto = 1


        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
        u = (x_min + x_max) / 2
        v = (y_min + y_max) / 2
        h = y_max - y_min

        # Cálculo refinado de distancia y altura
        distancia, altura_real = calcular_distancia(P0, altura_objeto, u, v, h, img_height)

        # Almacenar la estimación (X = 0 asumiendo que el coche está centrado en la imagen)
        estimaciones.append((0, altura_real, distancia))
        print(f"Clase: {clase}, Estimación 3D: X=0, Y={altura_real:.2f}, Z={distancia:.2f}")

        # Dibujar el bounding box y el centro del bounding box
        img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        img = cv2.circle(img, (int(u), int(v)), 5, (255, 0, 0), -1)
        cv2.putText(img, f"{clase}", (int(x_min), int(y_min - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Calcular el error por coordenada (X, Y, Z)
errores_por_coordenada = calcular_error_por_coordenada(estimaciones, verdad_terreno)

# Mostrar los errores por cada objeto
for i, (clase, error) in enumerate(zip([v[0] for v in verdad_terreno], errores_por_coordenada)):
    error_x, error_y, error_z = error
    print(f"Objeto {i+1} ({clase}):")
    print(f"Error en X: {error_x:.2f} metros")
    print(f"Error en Y: {error_y:.2f} metros")
    print(f"Error en Z: {error_z:.2f} metros")
    print('-' * 30)

# Mostrar la imagen con los bounding boxes dibujados
cv2.imshow("Detecciones YOLOv8", img)
cv2.waitKey(0)
cv2.destroyAllWindows()