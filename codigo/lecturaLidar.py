import os
import torch
import numpy as np

# Ruta a la carpeta que contiene los archivos LiDAR
ruta_carpeta = '/home/esteban/universidad/curso/datasets/subset_kitti/velodyne'

# Función para leer un archivo LiDAR .bin
def leer_puntos_lidar(ruta_archivo):
    # Cargar los datos LiDAR del archivo
    puntos = np.fromfile(ruta_archivo, dtype=np.float32).reshape(-1, 4)  # (X, Y, Z, Intensity)
    return puntos[:, :3]  # Solo usamos X, Y, Z

# Lista para almacenar los datos LiDAR como tensores
puntos_lidar_tensors = []

# Leer y procesar los archivos LiDAR
archivos = sorted(os.listdir(ruta_carpeta))  # Ordenar para procesar secuencialmente
for archivo in archivos:
    if archivo.endswith('.bin'):  # Verificar que sea un archivo .bin
        ruta_archivo = os.path.join(ruta_carpeta, archivo)
        puntos = leer_puntos_lidar(ruta_archivo)  # Leer los puntos LiDAR
        tensor_puntos = torch.tensor(puntos, dtype=torch.float32)  # Convertir a tensor
        puntos_lidar_tensors.append(tensor_puntos)

# Verificar tamaños de los tensores
for i, tensor in enumerate(puntos_lidar_tensors):
    print(f"Archivo {i}: Tamaño del tensor: {tensor.shape}")  # Muestra la forma de cada tensor

# Acceder a un archivo LiDAR específico
indice = 0  # Cambiar este índice para ver diferentes archivos
tensor_ejemplo = puntos_lidar_tensors[indice]
print(f"Puntos del archivo {indice}:\n{tensor_ejemplo}")

