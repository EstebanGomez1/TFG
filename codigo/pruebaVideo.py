import cv2
import os

video_path = '/home/esteban/universidad/curso/datasets/trafico/archive (1)/video/cctv052x2004080516x01638.avi'

# Verificar si el archivo existe
if not os.path.exists(video_path):
    print("El archivo no existe. Verifica la ruta.")
else:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error al abrir el video.")
    else:
        print("Video abierto correctamente.")

    cap.release()
