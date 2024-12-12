import cv2
import numpy as np
import pickle
import funciones
from ultralytics import YOLO

# Definicion de rutas

idImagen = "000032"
ruta_kitti = '/home/esteban/universidad/curso/datasets/subset_kitti'

ruta_imagen = f'{ruta_kitti}/image_2/{idImagen}.png'
ruta_lidar = f'{ruta_kitti}/velodyne/{idImagen}.bin'
ruta_calibracion = f'{ruta_kitti}/calib/{idImagen}.txt'
ruta_label = f'{ruta_kitti}/label_2/{idImagen}.txt'
ruta_diccionario = 'diccionario.pkl'

# Cargar imagen
imagen = cv2.imread(ruta_imagen)

# Diccionario de umbrales por clase
umbrales_clase = {
    'truck': 6.0,
    'car': 3.0,
    'van': 4.0,
    'pedestrian': 2.0
}

funciones.inferencia(imagen, idImagen, ruta_label, ruta_lidar, ruta_diccionario, ruta_calibracion)