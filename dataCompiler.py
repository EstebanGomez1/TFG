import torch
import torch.nn as nn
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
#from PTv3.model import PointTransformerV3
from typing import List
import math
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
from pprint import pprint
from torch.utils.data import ConcatDataset, DataLoader

def load_dictionary(file_pkl):
    try:
        with open(file_pkl, 'rb') as file:
            diccionario = pickle.load(file)
        print(f"Diccionario {file_pkl} cargado exitosamente.")
    except FileNotFoundError:
        print(f"Error: El archivo {file_pkl} no fue encontrado.")
        diccionario = {}
    return diccionario

class_to_idx = {
    'car': 0,
    'pedestrian': 1,
    'van': 2,
    'cyclist': 3,
    'truck': 4,
    'person': 5,
    'tram': 6,
    'misc': 7
}

def generar_ventanas2(objetos_dict, ventana=3):
    """
    Aplica una ventana deslizante sobre las imágenes de cada objeto y obtiene como salida la etiqueta de la última imagen de la ventana.
    """
    ventanas = []

    for objeto_id, imagenes in objetos_dict.items():
        img_ids = sorted(imagenes.keys()) 
        puntos = [imagenes[img_id]["points"] for img_id in img_ids]
        labels = [imagenes[img_id]["labels"] for img_id in img_ids]
        num_imagenes = len(puntos)

        for i in range(num_imagenes - ventana + 1):
            ventana_actual = puntos[i:i + ventana]
            salida = labels[i + ventana - 1]  # Último label de la ventana
            ventanas.append((ventana_actual, salida))

    print(ventanas)
    return ventanas

def generar_ventanas3(objetos_dict, ventana=3):
    ventanas_por_objeto = []

    # Ordenar los objetos por su id (importante para consistencia)
    for objeto_id in sorted(objetos_dict.keys()):
        imagenes = objetos_dict[objeto_id]
        # Ordenar las imagenes por su id (importante para consistencia temporal)
        img_ids = sorted(imagenes.keys())
        puntos = [imagenes[img_id]["points"] for img_id in img_ids]
        labels = [imagenes[img_id]["labels"] for img_id in img_ids]
        num_imagenes = len(puntos)
        
        ventanas = []
        for i in range(num_imagenes - ventana + 1):
            ventana_actual = puntos[i:i + ventana]
            salida = labels[i + ventana - 1]
            ventanas.append((ventana_actual, salida))

        ventanas_por_objeto.append(ventanas)
    print(ventanas_por_objeto)
    return ventanas_por_objeto

def generar_ventanas(objetos_dict, ventana=3):
    inputs = []
    outputs = []
    # Ordenar los objetos por su id (importante para consistencia)
    for objeto_id in sorted(objetos_dict.keys()):
        imagenes = objetos_dict[objeto_id]
        # Ordenar las imagenes por su id (importante para consistencia temporal)
        img_ids = sorted(imagenes.keys())
        puntos = [imagenes[img_id]["points"] for img_id in img_ids]
        labels = [imagenes[img_id]["labels"] for img_id in img_ids]
        num_imagenes = len(puntos)
        
        ventanas_objeto = []
        labels_objeto = []
        for i in range(num_imagenes - ventana + 1):
            ventana_actual = puntos[i:i + ventana]   # Lista de nubes de puntos
            salida = labels[i + ventana - 1]         # Etiqueta correspondiente
            ventanas_objeto.append(ventana_actual)
            labels_objeto.append(salida)

        inputs.append(ventanas_objeto)
        outputs.append(labels_objeto)

    dataset = {
        "inputs": inputs,   
        "outputs": outputs 
    }
    return dataset

def parse_label(label_dict):
    return torch.tensor([
        class_to_idx.get(label_dict.get("class", "car"), -1),
        label_dict.get("x", 0.0),
        label_dict.get("y", 0.0),
        label_dict.get("z", 0.0),
        label_dict.get("length", 0.0),
        label_dict.get("height", 0.0),
        label_dict.get("width", 0.0),
        label_dict.get("rot_y", 0.0)
    ], dtype=torch.float32)

import json
import pprint
class WindowedDataset(Dataset):
    def __init__(self, objetos_dict, ventana=3):
        self.ventana = ventana
        self.inputs = []
        self.outputs = []
        skip_classes = {4, 6, 7}
        """with open("objetos_dict1.txt", "w") as f:
            pprint.pprint(objetos_dict, stream=f, indent=2, width=120)"""
        for objeto_id in sorted(objetos_dict.keys()):
            imagenes = objetos_dict[objeto_id]
            img_ids = sorted(imagenes.keys())
            """with open("objetos_dict.txt", "w") as f:
                pprint.pprint(objetos_dict[objeto_id], stream=f, indent=2, width=120)"""
            puntos = [imagenes[img_id]["points"] for img_id in img_ids]
            labels = [imagenes[img_id]["labels"] for img_id in img_ids]
            num_imagenes = len(puntos)

            for i in range(num_imagenes - ventana + 1):
                ventana_actual = puntos[i:i + ventana]
                salida = labels[i + ventana - 1]
                clase_idx = class_to_idx.get(salida.get("class", "car"), -1)
                if clase_idx in skip_classes:
                    continue  # salta esta muestra

                tensor_ventana = [torch.tensor(p) for p in ventana_actual]
                tensor_salida = parse_label(salida)

                self.inputs.append(tensor_ventana)
                self.outputs.append(tensor_salida)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def normalize_crop_and_box(points, box):
    """
    points: Tensor Nx4 (x, y, z, intensidad)
    box: Tensor 7 (x, y, z, l, w, h, alpha)
    """
    # Centro del crop (podrías usar el centroide o el centro de la box)
    crop_center = points[:, :3].mean(dim=0)  # [3], centroide XYZ

    # Normaliza puntos
    points[:, :3] = points[:, :3] - crop_center

    # Normaliza bounding box
    box[1:4] = box[1:4] - crop_center

    return points, box

def normalize_crop_and_box2(points, box):
    """
    points: Tensor Nx4 (x, y, z, intensidad)
    box: Tensor 8 (clase, x, y, z, h, w, l, alpha)
    """
    # Centro de la caja (en KITTI: box[1:4])
    box_center = box[1:4].clone()  # [x, y, z]

    # Normaliza puntos
    points[:, :3] = points[:, :3] - box_center

    # Normaliza bounding box (centro en origen)
    box[1:4] = box[1:4] - box_center  # ahora box[1:4] = (0,0,0)

    return points, box

def normalize_crop_and_box3(points, box):
    """
    Normaliza los puntos y la box respecto al centro del crop, no al centro de la caja.
    points: Tensor Nx4 (x, y, z, intensidad)
    box: Tensor 8 (clase, x, y, z, h, w, l, alpha)
    """
    # Calcula el centro del crop (media de los puntos)
    crop_center = points[:, :3].mean(dim=0)  # [x, y, z]

    # Normaliza puntos (resta el centro del crop)
    points_normalized = points.clone()
    points_normalized[:, :3] -= crop_center

    # Normaliza la caja (resta el centro del crop a la posición de la box)
    box_normalized = box.clone()
    box_normalized[1:4] -= crop_center

    return points, box, crop_center

def custom_collate3_fn(batch):
    # batch = [([tensor1, tensor2, ...], label_tensor), ...]
    entradas = []
    salidas = []
    centros = []  # Para guardar los centroides (si quieres desnormalizar luego)
    for item in batch:
        crop_tensors = []
        for tensor in item[0]:  # lista de tensores del crop
            puntos, box= normalize_crop_and_box2(tensor.clone(), item[1].clone())
            crop_tensors.append(puntos)
            # Como todas las boxes del batch son la misma, solo agregamos una vez
        entradas.extend(crop_tensors)
        salidas.append(box)
        #centros.append(item[1][1:4].clone())
    return entradas, salidas #, centros

def custom_collate4_fn(batch):
    # batch = [([tensor1, tensor2, ...], label_tensor), ...]
    entradas = []
    salidas = []
    centros = []  # Para guardar los centroides (si quieres desnormalizar luego)
    for item in batch:
        crop_tensors = []
        centros_crop = [] 
        for tensor in item[0]:  # lista de tensores del crop
            puntos, box, centro= normalize_crop_and_box3(tensor.clone(), item[1].clone()) # con 4 los mejores resultados hasta ahora
            crop_tensors.append(puntos)
            centros_crop.append(centro)
            # Como todas las boxes del batch son la misma, solo agregamos una vez
        entradas.extend(crop_tensors)
        salidas.append(box)
        centros.append(torch.stack(centros_crop, dim=0).mean(dim=0))
    return entradas, salidas , centros




#-------------------------------------------------------------

def distancia_centros(ventana, label):
    todas = torch.cat(ventana, dim=0)
    centroide_global = todas[:, :3].mean(dim=0)
    coords_label = label[1:4]  
    distancia = torch.norm(centroide_global - coords_label)
    return distancia

def normalize_crop(points):
    centro = points[:, :3].mean(dim=0)
    points = points.clone()
    points[:, :3] -= centro
    return points, centro

def custom_collate5_fn(batch):
    salto = False
    # batch = [([tensor1, tensor2, ...], label_tensor), ...]
    entradas = []
    salidas = []
    centros = []  # Guardar los centroides de cada ventana
    for item in batch:
        crop_tensors = []
        centros_crop = []
        for tensor in item[0]: 
            puntos, centro = normalize_crop(tensor)
            crop_tensors.append(puntos)
            centros_crop.append(centro)
        entradas.extend(crop_tensors)
        centros_crop = torch.stack(centros_crop, dim=0)  # (N, 3)
        centro_ventana = centros_crop.mean(dim=0)         # (3,)
        box = item[1].clone()
        box[1:4] -= centro_ventana
        """print(f"centros_crop {centros_crop}")
        print(f"centro {centro_ventana}")
        print(f"label original {item[1]}")
        print(f"label final {box}")"""
        #distancia = torch.norm(box[1:4])
        distancia =distancia_centros(crop_tensors, box)
        if distancia > 2:
            salto = True
        salidas.append(box)
        centros.append(centro_ventana)
    return entradas, salidas, centros, salto


#-------------------------------------------------------------


def custom_collate6aux_fn(batch):
    # batch = [([tensor1, tensor2, ...], label_tensor), ...]
    entradas = []
    salidas = []
    centros = []  # Para guardar los centroides (si quieres desnormalizar luego)
    for item in batch:
        crop_tensors = []
        centros_crop = [] 
        for tensor in item[0]:  # lista de tensores del crop
            #puntos, box, centro= normalize_crop_and_box3(tensor.clone(), item[1].clone()) # con 4 los mejores resultados hasta ahora
            crop_tensors.append(tensor)
            # Como todas las boxes del batch son la misma, solo agregamos una vez
        entradas.extend(crop_tensors)
        salidas.append(item[1])
        #centros.append(torch.stack(centros_crop, dim=0).mean(dim=0))
    return entradas, salidas #, centros










def get_window_center(list_of_points):
    # Concatenar todas las nubes de puntos de la ventana
    all_points = torch.cat(list_of_points, dim=0)  # [N_total, 4]
    centroide = all_points[:, :3].mean(dim=0)      # [3]
    return centroide

def custom_collate2_fn(batch):
    # batch = [([tensor1, tensor2, ...], label_tensor), ...]
    entradas = []
    for item in batch:
        # item[0] es [tensor1, tensor2, ...]
        entradas.extend(item[0])  # aplanar todas las listas de tensores
    salidas = [item[1] for item in batch]
    return entradas, salidas

def custom_collate11_fn(batch):
    entradas = [item[0] for item in batch]  # lista de listas de tensores
    salidas = torch.stack([item[1] for item in batch])  # tensor [batch, 8]
    return entradas, salidas

def custom_collate_fn(batch):
    entradas = [item[0] for item in batch]  # lista de listas de tensores
    salidas = [item[1] for item in batch]   # lista de tensores label
    return entradas, salidas

# Función de collation para batches
"""def ventana_collate(batch):
    entradas = [item["entrada"] for item in batch]  # Lista de listas de tensores
    salidas = torch.stack([item["salida"] for item in batch])  # Labels en batch

    # Encontrar la longitud máxima de las nubes de puntos en todas las entradas del batch
    max_len = max(len(seq) for entrada in entradas for seq in entrada)  # La longitud máxima de la secuencia dentro de cada entrada

    # Rellenar las secuencias con ceros para que tengan la misma longitud
    entradas_padded = [
        [torch.cat([seq, torch.ones(max_len - len(seq), seq.size(1))], dim=0) if len(seq) < max_len else seq for seq in entrada]
        for entrada in entradas
    ]

    # Apilar las secuencias rellenas
    entradas_padded = [torch.stack(entrada) for entrada in entradas_padded]

    return {"entrada": entradas_padded, "salida": salidas}"""
def ventana_collate(batch):
    # Filtrar entradas con salidas vacías
    batch = [item for item in batch if item[1].numel() > 0]
    if len(batch) == 0:
        raise ValueError("Batch vacío: todas las salidas están vacías.")

    entradas = [item[0] for item in batch]  # Lista de [W, L_i, C]
    salidas = [item[1] for item in batch]   # Lista de [W] (o variable)

    # Paso 1: calcular longitudes máximas
    max_ventanas = max(entrada.size(0) for entrada in entradas)       # W
    max_puntos = max(entrada.size(1) for entrada in entradas)         # L
    num_canales = entradas[0].size(2)                                 # C

    # Paso 2: rellenar con padding cada entrada → [W, L, C]
    entradas_padded = []
    for entrada in entradas:
        ventanas = []
        for ventana in entrada:
            if ventana.size(0) < max_puntos:
                padding = torch.zeros(max_puntos - ventana.size(0), num_canales)
                ventana = torch.cat([ventana, padding], dim=0)
            ventanas.append(ventana)
        if len(ventanas) < max_ventanas:
            # Rellenar con ventanas vacías si faltan
            dummy_ventana = torch.zeros(max_puntos, num_canales)
            ventanas += [dummy_ventana] * (max_ventanas - len(ventanas))
        entrada_tensor = torch.stack(ventanas)  # [W, L, C]
        entradas_padded.append(entrada_tensor)

    entradas_batch = torch.stack(entradas_padded)  # [B, W, L, C]

    # Paso 3: pad también salidas si varían en longitud
    max_len_salidas = max(salida.size(0) for salida in salidas)
    salidas_padded = [
        torch.cat([salida, torch.full((max_len_salidas - salida.size(0),), 0, dtype=torch.long)])
        if salida.size(0) < max_len_salidas else salida
        for salida in salidas
    ]
    salidas_batch = torch.stack(salidas_padded)

    return {"entrada": entradas_batch, "salida": salidas_batch}


def format_dicc(diccionario):
    """
    Cambia el formato del diccionario para tener un diccionario de objetos donde cada objeto 
    tiene un diccionario de imagenes donde aparece con sus nubes y labels asociados
    """
    objetos_dict = {}
    for img_id in diccionario:
        for obj in diccionario[img_id]:
            puntos = diccionario[img_id][obj]["points"]  # Nube de puntos
            label = diccionario[img_id][obj]["label"]    # Info del objeto
            obj_id = label["id"]
            if( obj_id not in objetos_dict):
                objetos_dict[obj_id]={}
            objetos_dict[obj_id][img_id] = {
                "points": puntos,
                "labels": label
            }           
    return objetos_dict

def correr_prueba(ventana_dataloader):
    print("Entrando a correr_prueba...")
    batches_leidos = 0
    for batch in ventana_dataloader:
        print("Batch completo de entradas:")
        for i, entrada in enumerate(batch["entrada"]):
            print(f"\n🔹 Entrada {i}:")
            for j, tensor in enumerate(entrada):
                # Mostrar la forma solo si es tensor/array, si es lista muestra el tamaño
                if hasattr(tensor, "shape"):
                    print(f" Tensor {j}: shape={tensor.shape}\n")
                elif hasattr(tensor, "size"):
                    print(f" Tensor {j}: size={tensor.size()}\n")
                elif isinstance(tensor, list):
                    print(f" Tensor {j}: list len={len(tensor)}\n")
                else:
                    print(f" Tensor {j}: type={type(tensor)} value={tensor}\n")
        print("\nBatch de salidas:")
        print(batch["salida"])
        break  # Solo mostramos un batch para no llenar la consola
    if batches_leidos == 0:
        print("⚠️ No se leyó ningún batch (el DataLoader está vacío o no itera)")

def getDataLoader(diccionario):
    dataset = WindowedDataset(format_dicc(diccionario))
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
### Uso del modelo ###

# diccionario = load_dictionary("diccionario0000.pkl")
# ventanas_generadas = generar_ventanas(format_dicc(diccionario), ventana=3)
# ventana_dataset = VentanaDataset(ventanas_generadas)
# ventana_dataloader = DataLoader(ventana_dataset, batch_size=4, shuffle=False, collate_fn=ventana_collate)




def ver_dicc(diccionario):
    for imagen in diccionario:
        print(f"\n---------------------------------------{imagen}-----------------------------------------------\n")
        pprint(diccionario[imagen])

def ver_format_dicc(diccionario):
    for obj_id in diccionario:
        print(f"\n---------------------------------------{obj_id}-----------------------------------------------\n")
        pprint(diccionario[obj_id])

def ver_dataSet(ventana_dataset):
    for i in range(3):  # Mira las primeras 3 muestras
        ventana, label = ventana_dataset[i]
        print(f"Sample {i}:")
        print(f"  Ventana (lista de {len(ventana)} tensores):")
        for j, t in enumerate(ventana):
            print(f"    Tensor {j}: shape={t.shape}, dtype={t.dtype}")
        print(f"  Label tensor: shape={label.shape}, values={label}")
        print("-"*40)
            
        

def data_compiler(numDiccs=2):
    diccionarios =[]
    #for i in range(0, numDiccs):
    #    #nombre_archivo = f"diccionario{str(i).zfill(4)}.pkl"
    nombre_archivo = 'dicc3.pkl'
    if os.path.exists(nombre_archivo):
        diccionario = load_dictionary(nombre_archivo)
        diccionarios.append(diccionario)
    else:
        print(f"Archivo {nombre_archivo} no encontrado, se salta.")
    return diccionarios


# Ejemplo de diccionario
diccionario_prueba = {
    "1": {
        "img_01": {"points": "nube1", "labels": "etiquetas1-1"},
        "img_02": {"points": "nube2", "labels": "etiquetas2-1"},
        "img_03": {"points": "nube3", "labels": "etiquetas3-1"},
        "img_04": {"points": "nube4", "labels": "etiquetas4-1"},
        "img_05": {"points": "nube5", "labels": "etiquetas5-1"},
    },
    "2": {
        "img_01": {"points": "nube1", "labels": "etiquetas1-2"},
        "img_02": {"points": "nube2", "labels": "etiquetas2-2"},
        "img_03": {"points": "nube3", "labels": "etiquetas3-2"},
        "img_04": {"points": "nube4", "labels": "etiquetas4-2"},
        "img_05": {"points": "nube5", "labels": "etiquetas5-2"},
    }
}



#print("-------------------")
#datal = getDataLoader(data_compiler()[0])
#print("Longitud del dataset:", len(datal.dataset))
#correr_prueba(datal)

#diccNormal = data_compiler()[0]
#ver_dicc(dicc)
#dicc_format = format_dicc(dicc)
#ver_format_dicc(dicc_format)
#pprint(dicc_format)
"""
dicc = diccionario_prueba
dicc = format_dicc(diccNormal)"""

"""ventanas_generadas = generar_ventanas(dicc, ventana=3)
print(f"long outs: {len(ventanas_generadas['outputs'])}")
print(ventanas_generadas['outputs'][1][0])
print(len(ventanas_generadas['inputs'][1][0]))"""

#ventanas_generadas = format_dicc(diccNormal)
#pprint(ventanas_generadas)
#ventana_dataset = WindowedDataset(ventanas_generadas)


"""dicc = load_dictionary("diccionario0.pkl")
dataloader = getDataLoader(dicc)
for i, batch in enumerate(dataloader):
    if i == 159:
        #torch.set_printoptions(threshold=float('inf'))
        print("Iteración:", i)
        print("Inputs:", batch[0])
        print("Labels:", batch[1])
print("-------------------")"""

##### Fusionar diccionario #####

def getMultiDataLoader2(diccionarios):
    datasets = []
    for diccionario in diccionarios:
        datos_formateados = format_dicc(diccionario)
        dataset = WindowedDataset(datos_formateados)
        datasets.append(dataset)

    dataset_total = ConcatDataset(datasets)
    return dataset_total #DataLoader(dataset_total, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

def getMultiDataLoader(diccionarios):
    datasets = []
    for diccionario in diccionarios:
        datos_formateados = format_dicc(diccionario)
        dataset = WindowedDataset(datos_formateados)
        datasets.append(dataset)

    dataset_total = ConcatDataset(datasets)
    return DataLoader(dataset_total, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

def load_dicctionaries(total=6):
    lista_diccionarios = []
    for i in range(total):  # de 0 a 5
        #dicc_name = f"diccionarios/diccionario{i}.pkl"
        dicc_name = f"dicc{i}.pkl"
        dicc = load_dictionary(dicc_name)
        lista_diccionarios.append(dicc)
    return lista_diccionarios


"""print("-------------------")
dataloader = getMultiDataLoader(load_dicctionaries(5))
for i, batch in enumerate(dataloader):
    if i == 2890:
        torch.set_printoptions(threshold=float('inf'))
        print("Iteración:", i)
        print("Inputs:", len(batch[0][0]))
        print("Labels:", batch[1])
print(f"numero de ventanas totales: {i}")"""

