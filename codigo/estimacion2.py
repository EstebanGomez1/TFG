import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def leer_puntos_lidar(ruta_archivo):
    puntos = np.fromfile(ruta_archivo, dtype=np.float32).reshape(-1, 4)
    return puntos[:, :3]  # Solo X, Y, Z

def leer_matrices_calibracion(archivo_calibracion):
    P2 = None
    R0_rect = None
    Tr_velo_to_cam = None
    with open(archivo_calibracion, 'r') as f:
        for line in f:
            if line.startswith("P2:"):
                P2 = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
            elif line.startswith("R0_rect:"):
                R0_rect = np.array(line.split()[1:], dtype=np.float32).reshape(3, 3)
            elif line.startswith("Tr_velo_to_cam:"):
                Tr_velo_to_cam = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
    return P2, R0_rect, Tr_velo_to_cam

def proyectar_punto_3d_a_2d(punto_3d, P2, R0_rect, Tr_velo_to_cam):
    punto_hom = np.append(punto_3d, 1)
    punto_cam = R0_rect @ (Tr_velo_to_cam @ punto_hom)[:3]
    punto_img_hom = P2 @ np.append(punto_cam, 1)
    u, v, w = punto_img_hom
    u /= w
    v /= w
    return u, v

def filtrar_outliers(puntos, threshold=1):
    # Calcula la media y la desviación estándar
    media = np.mean(puntos, axis=0)
    desviacion_std = np.std(puntos, axis=0)
    
    # Filtrar puntos que están dentro de threshold desviaciones de la media
    puntos_filtrados = [punto for punto in puntos if np.all(np.abs(punto - media) <= threshold * desviacion_std)]
    return np.array(puntos_filtrados)

def encontrar_centro_bounding_box(puntos_lidar, bounding_box, P2, R0_rect, Tr_velo_to_cam):
    x_min, y_min, x_max, y_max = bounding_box
    puntos_en_bounding_box = []

    # Proyectar cada punto LiDAR y verificar si está dentro del bounding box
    for punto in puntos_lidar:
        u, v = proyectar_punto_3d_a_2d(punto, P2, R0_rect, Tr_velo_to_cam)
        if x_min <= u <= x_max and y_min <= v <= y_max:
            puntos_en_bounding_box.append(punto)
    
    if puntos_en_bounding_box:
        # Filtrar outliers usando la desviación estándar
        puntos_en_bounding_box = filtrar_outliers(puntos_en_bounding_box)
        
        if len(puntos_en_bounding_box) > 0:
            # Calcular el centro en 3D usando la media de los puntos filtrados
            centro_3d = np.mean(puntos_en_bounding_box, axis=0)
            return centro_3d
    print("No se encontraron puntos LiDAR significativos dentro del bounding box.")
    return None

# Rutas de los archivos
ruta_imagen = '/home/esteban/universidad/curso/datasets/subset_kitti/image_2/000019.png'
ruta_lidar = '/home/esteban/universidad/curso/datasets/subset_kitti/velodyne/000019.bin'
ruta_calibracion = '/home/esteban/universidad/curso/datasets/subset_kitti/calib/000019.txt'

# Cargar la imagen
imagen = cv2.imread(ruta_imagen)

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt')

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: no se pudo cargar la imagen. Verifica la ruta del archivo.")
else:
    # Detectar objetos en la imagen
    results = model.predict(imagen)

    # Leer los puntos LiDAR del archivo
    puntos_lidar = leer_puntos_lidar(ruta_lidar)

    # Leer las matrices de calibración
    P2, R0_rect, Tr_velo_to_cam = leer_matrices_calibracion(ruta_calibracion)

    # Procesar cada bounding box detectado por YOLO
    for r in results:
        for box in r.boxes:
            clase = r.names[int(box.cls.item())]
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

            # Estimar el centro 3D del bounding box usando los puntos LiDAR
            centro_3d = encontrar_centro_bounding_box(puntos_lidar, [x_min, y_min, x_max, y_max], P2, R0_rect, Tr_velo_to_cam)
            
            if centro_3d is not None:
                # Proyectar el centro 3D a la imagen 2D
                u, v = proyectar_punto_3d_a_2d(centro_3d, P2, R0_rect, Tr_velo_to_cam)
                
                # Dibujar el bounding box, el centro 3D proyectado y la distancia en la imagen
                cv2.rectangle(imagen, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.circle(imagen, (int(u), int(v)), 5, (255, 0, 0), -1)
                distancia = np.linalg.norm(centro_3d)
                cv2.putText(imagen, f"{clase} Dist: {distancia:.2f}m", (int(x_min), int(y_min - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar la imagen con los bounding boxes y los centros proyectados
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
