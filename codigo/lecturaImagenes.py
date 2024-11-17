import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Ruta a la carpeta que contiene las imágenes
ruta_carpeta = '/home/esteban/universidad/curso/datasets/subset_kitti/image_2'

# Transformaciones para las imágenes
transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar las imágenes a 224x224
    transforms.ToTensor(),  # Convertir la imagen en un tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización
])

# Lista para almacenar las imágenes como tensores
imagenes_tensor = []

# Cargar las imágenes en el rango especificado
for i in range(51):  # Desde 000000.png hasta 000050.png
    nombre_archivo = f"{i:06d}.png"  # Formato de nombre de archivo
    ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)
    
    if os.path.exists(ruta_imagen):  # Verificar si el archivo existe
        # Cargar la imagen y aplicar las transformaciones
        imagen = Image.open(ruta_imagen).convert("RGB")  # Convertir a RGB si es necesario
        imagen_tensor = transformaciones(imagen)
        imagenes_tensor.append(imagen_tensor)
    else:
        print(f"Advertencia: La imagen {ruta_imagen} no existe.")

# Convertir la lista de tensores en un único tensor
imagenes_tensor = torch.stack(imagenes_tensor)  # Dimensión: (N, C, H, W)
print(f"Tamaño del tensor de imágenes: {imagenes_tensor.shape}")  # (N, 3, 224, 224)

# Mostrar una imagen para verificar el contenido del tensor
indice = 1  # Cambia este valor para ver diferentes imágenes
imagen_mostrar = imagenes_tensor[indice]
imagen_mostrar = imagen_mostrar.permute(1, 2, 0)  # Cambiar las dimensiones a (H, W, C)
imagen_mostrar = imagen_mostrar * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Desnormalizar
imagen_mostrar = torch.clamp(imagen_mostrar, 0, 1)  # Asegurar que los valores estén entre 0 y 1

plt.imshow(imagen_mostrar.numpy())
plt.axis("off")
plt.show()
