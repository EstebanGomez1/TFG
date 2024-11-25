import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    punto_hom = np.append(punto_3d, 1) #punto homogeneo
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
            return centro_3d, puntos_en_bounding_box
    print("No se encontraron puntos LiDAR significativos dentro del bounding box.")
    return None


def filtrar_puntos_por_distancia(puntos, centro, umbral):
    """
    Filtra los puntos que están dentro de una distancia umbral desde un centro dado.
    """
    distancias = np.linalg.norm(puntos - centro, axis=1)
    puntos_filtrados = puntos[distancias <= umbral]
    return puntos_filtrados



def visualizar_puntos_3d(puntos, titulo="Visualización 3D", limite=-4):
    """
    Visualiza los puntos en 3D, mostrando solo aquellos que están por encima de un valor límite en el eje Z.

    Parámetros:
        puntos (ndarray): Nube de puntos LiDAR (X, Y, Z).
        titulo (str): Título de la visualización.
        limite (float): Límite inferior para la coordenada Z. Solo se visualizan los puntos con Z > limite.
    """
    # Filtrar los puntos que están por encima del límite en Z
    puntos_filtrados = puntos[puntos[:, 2] > limite]

    if len(puntos_filtrados) == 0:
        print(f"No hay puntos con coordenada Z por encima del límite {limite}.")
        return

    # Crear la visualización
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        puntos_filtrados[:, 0], 
        puntos_filtrados[:, 1], 
        puntos_filtrados[:, 2], 
        s=1, 
        c=puntos_filtrados[:, 2], 
        cmap='viridis'
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{titulo}")
    plt.show()



###################################


def filtrar_puntos_cercanos_al_centro(puntos, centro, umbral):
    """
    Filtra los puntos LiDAR que están dentro de una distancia umbral al centro del bounding box.

    Parámetros:
        puntos (ndarray): Puntos LiDAR (X, Y, Z).
        centro (ndarray): Coordenadas del centro 3D del bounding box (X, Y, Z).
        umbral (float): Distancia máxima permitida desde el centro.

    Retorna:
        ndarray: Puntos filtrados.
    """
    # Calcular la distancia euclidiana de cada punto al centro
    distancias = np.linalg.norm(puntos - centro, axis=1)

    # Filtrar los puntos que están dentro del umbral
    puntos_filtrados = puntos[distancias <= umbral]

    return puntos_filtrados


from sklearn.cluster import DBSCAN

def filtrar_nucleo_principal_2d(puntos, centro_3d, eps=1.0, min_samples=5):
    """
    Filtra los puntos LiDAR dejando solo el núcleo principal más cercano al centro del bounding box 
    usando DBSCAN y calculando distancias en el plano X-Y.

    Parámetros:
        puntos (ndarray): Puntos LiDAR (X, Y, Z).
        centro_3d (ndarray): Centro 3D del bounding box.
        eps (float): Distancia máxima entre puntos para considerarlos vecinos.
        min_samples (int): Mínimo número de puntos para formar un núcleo.

    Retorna:
        ndarray: Puntos pertenecientes al núcleo más cercano al centro del bounding box.
    """
    if not isinstance(puntos, np.ndarray):
        puntos = np.array(puntos)

    # Aplicar DBSCAN usando solo las componentes X-Y
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(puntos[:, :2])

    # Obtener etiquetas de los clusters
    etiquetas = clustering.labels_

    # Verificar si hay clusters válidos
    etiquetas_validas = np.unique(etiquetas[etiquetas >= 0])
    if len(etiquetas_validas) == 0:
        print("No se encontró un núcleo principal.")
        return np.array([])

    # Calcular el centroide de cada cluster en el plano X-Y
    centroides = {
        etiqueta: np.mean(puntos[etiquetas == etiqueta][:, :2], axis=0)
        for etiqueta in etiquetas_validas
    }

    # Convertir el centro del bounding box a X-Y
    centro_xy = centro_3d[:2]

    # Encontrar el cluster cuyo centroide esté más cerca del centro del bounding box en X-Y
    distancias_al_centro = {
        etiqueta: np.linalg.norm(centroide - centro_xy)
        for etiqueta, centroide in centroides.items()
    }
    cluster_mas_cercano = min(distancias_al_centro, key=distancias_al_centro.get)

    # Filtrar los puntos del cluster más cercano
    puntos_filtrados = puntos[etiquetas == cluster_mas_cercano]

    return puntos_filtrados



from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def diagnosticar_dbscan(puntos, eps=1.0, min_samples=5):
    """
    Aplica DBSCAN y diagnostica los clusters generados.

    Parámetros:
        puntos (ndarray): Puntos LiDAR (X, Y, Z).
        eps (float): Distancia máxima entre puntos para considerarlos vecinos.
        min_samples (int): Mínimo número de puntos para formar un núcleo.

    Retorna:
        clustering (DBSCAN): Modelo de clustering.
    """
    if not isinstance(puntos, np.ndarray):
        puntos = np.array(puntos)

    # Aplicar DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(puntos[:, :2])

    # Visualizar los clusters asignados
    etiquetas = clustering.labels_
    plt.scatter(puntos[:, 0], puntos[:, 1], c=etiquetas, cmap='viridis', s=5)
    plt.title(f"Diagnóstico DBSCAN: eps={eps}, min_samples={min_samples}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Etiqueta del Cluster")
    plt.show()

    return clustering





##################################


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

    puntos_filtrados_totales = []

    """
    # Procesar cada bounding box detectado por YOLO
    for r in results:
        for box in r.boxes:
            clase = r.names[int(box.cls.item())]
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

            # Estimar el centro 3D del bounding box usando los puntos LiDAR
            centro_3d = encontrar_centro_bounding_box(puntos_lidar, [x_min, y_min, x_max, y_max], P2, R0_rect, Tr_velo_to_cam)
            
            if centro_3d is not None:
                # Filtrar puntos LiDAR a una distancia de 6m del centro en 3D
                puntos_filtrados = filtrar_puntos_por_distancia(puntos_lidar, centro_3d, umbral=6.0)
                puntos_filtrados_totales.extend(puntos_filtrados)
    """

###############################


# Crear una lista global para almacenar los puntos filtrados de todos los objetos
todos_los_puntos = []

contador = 0 

# Procesar cada bounding box detectado por YOLO
for r in results:
    for box in r.boxes:
        if contador >= 2:  # Salir del bucle después de 2 iteraciones
            break

        clase = r.names[int(box.cls.item())]
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

        # Obtener el centro 3D y los puntos dentro del bounding box
        centro_3d, puntos = encontrar_centro_bounding_box(
            puntos_lidar, [x_min, y_min, x_max, y_max], P2, R0_rect, Tr_velo_to_cam
        )
        punto_menor_z = puntos[np.argmin(puntos[:, 2])]  # Encontrar el punto más bajo en Z
        menor_z = punto_menor_z[2]  # Coordenada Z mínima

        if centro_3d is not None and len(puntos) > 0:
            # Filtrar puntos LiDAR a una distancia de 3m del centro en 3D
            puntos_filtrados = filtrar_puntos_por_distancia(puntos_lidar, centro_3d, umbral=3.0)

            # Diagnóstico DBSCAN
            clustering = diagnosticar_dbscan(puntos_filtrados, eps=0.5, min_samples=10)

            # Obtener etiquetas y filtrar núcleo principal
            etiquetas = clustering.labels_
            etiquetas_validas, conteos = np.unique(etiquetas[etiquetas >= 0], return_counts=True)

            if len(etiquetas_validas) > 0:
                # Calcular el centroide de cada cluster
                centroides = {
                    etiqueta: np.mean(puntos_filtrados[etiquetas == etiqueta][:, :2], axis=0)
                    for etiqueta in etiquetas_validas
                }

                # Encontrar el cluster más cercano al centro
                centro_xy = centro_3d[:2]
                distancias_al_centro = {
                    etiqueta: np.linalg.norm(centroide - centro_xy)
                    for etiqueta, centroide in centroides.items()
                }
                cluster_mas_cercano = min(distancias_al_centro, key=distancias_al_centro.get)

                # Filtrar puntos del cluster más cercano
                puntos_filtrados_dbscan = puntos_filtrados[etiquetas == cluster_mas_cercano]

                # Recortar puntos por el límite inferior en Z
                puntos_recortados = puntos_filtrados_dbscan[puntos_filtrados_dbscan[:, 2] > menor_z]

                if len(puntos_recortados) > 0:
                    # Agregar los puntos recortados del objeto actual a la lista global
                    todos_los_puntos.append(puntos_recortados)

                visualizar_puntos_3d(
                    puntos_recortados,
                    titulo="Todos los Objetos Detectados en el Espacio 3D",
                    limite=menor_z
                )

                contador += 1

# Combinar todos los puntos en un único array
todos_los_puntos = np.vstack(todos_los_puntos) if len(todos_los_puntos) > 0 else np.array([])

# Visualizar todos los puntos en un único espacio 3D
if len(todos_los_puntos) > 0:
    visualizar_puntos_3d(
        todos_los_puntos,
        titulo="Todos los Objetos Detectados en el Espacio 3D",
        limite=-4
    )
else:
    print("No se encontraron puntos válidos para visualizar.")

