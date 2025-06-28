import os
#os.environ["TORCH_USE_CUDA_DSA"] = "1"
import torch
import torch.nn as nn
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
from PTv3.model import PointTransformerV3
from typing import List
import math
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import dataCompiler
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np
from torch.utils.data import random_split, DataLoader
sys.path.append('IoU/2D-3D-IoUs-main')
from IoU import IoU3D, IoUs2D
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import funciones4
import logging

"""logging.basicConfig(
    filename='errores.log',          # Nombre del archivo de log
    filemode='w',
    level=logging.ERROR,             # Nivel mínimo a registrar
    format='%(asctime)s %(levelname)s: %(message)s',  # Formato de cada línea
    datefmt='%Y-%m-%d %H:%M:%S'
)
metrics_logger = logging.getLogger('metrics_logger')
metrics_logger.setLevel(logging.INFO)"""

# Logger para errores
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('errores.log', mode='w')
error_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)

# Logger para métricas
metrics_logger = logging.getLogger('metrics_logger')
metrics_logger.setLevel(logging.INFO)
metrics_handler = logging.FileHandler('metricas.log', mode='w')
metrics_formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
metrics_handler.setFormatter(metrics_formatter)
metrics_logger.addHandler(metrics_handler)

def save_checkpoint(model, optimizer, epoch, path, best_metric=None):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(state, path)
    print(f"Checkpoint guardado en {path}")

# Para cargar el checkpoint
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint cargado de {path} (época {epoch})")
    return epoch

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

class PTv3_deteccion(nn.Module):
    def __init__(self, grid_size: int):
        super(PTv3_deteccion, self).__init__()
        self.grid_size = grid_size

        self.point_encoder = PointTransformerV3(
            in_channels=1,
            enc_depths=(1, 1, 1, 1, 1),
            enc_num_head=(1, 2, 4, 4, 4),
            enc_patch_size=(64, 64, 64, 64, 64),
            enc_channels=(32, 64, 96, 96, 128),
            dec_depths=(1, 1, 1, 1),
            dec_channels=(96, 64, 64, 32),
            dec_num_head=(4, 2, 2, 2),
            dec_patch_size=(64, 64, 64, 64),
            mlp_ratio=1.0,
            qkv_bias=False
        )

        self.clf_head = nn.Linear(8, 8)      # 8 clases
        self.norm = nn.LayerNorm(8)
        self.reg_head = nn.Linear(8, 6)      # 6 valores
        #self.cyc_head_sin = nn.Linear(8, 1)
        #self.cyc_head_cos = nn.Linear(8, 1)      # 1 valor
        self.cyc_head_sin = nn.Sequential(nn.Linear(8, 1), nn.Tanh())
        self.cyc_head_cos = nn.Sequential(nn.Linear(8, 1), nn.Tanh())
        """self.cyc_head_sin = nn.Sequential(
            nn.Linear(8, 8),
            nn.LayerNorm(8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

        self.cyc_head_cos = nn.Sequential(
            nn.Linear(8, 8),
            nn.LayerNorm(8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )"""

        # usamos una MLP para pasar a 8 caracteristicas
        self.feature_layer = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

    def forward(self, ventana: list):
        device = next(self.parameters()).device
        ventana = [f.to(device) for f in ventana]
        ventana = [f.squeeze(0) if f.ndim == 3 and f.shape[0] == 1 else f for f in ventana]
        all_points = torch.cat(ventana, dim=0)
        offset = torch.tensor([f.shape[0] for f in ventana], device=device).cumsum(0)
        points_dict = {
            "feat": all_points[:, 3:],
            "coord": all_points[:, :3],
            "offset": offset,
            "grid_size": self.grid_size,
        }
        point_features = self.point_encoder(points_dict)
        feat = points_dict["feat"]
        if feat.shape[1] == 0:
            feat = torch.zeros((feat.shape[0], 1), device=feat.device)
            points_dict["feat"] = feat
        feats_mean = torch.mean(point_features["feat"], dim=0, keepdim=True)  # [1, D]
        embedding = self.feature_layer(feats_mean)
        normalized_embedding = self.norm(embedding)
        logits = self.clf_head(normalized_embedding)     # [batch, 8]
        reg_out = self.reg_head(normalized_embedding)    # [batch, 6]
        sin_out = self.cyc_head_sin(normalized_embedding)
        cos_out = self.cyc_head_cos(normalized_embedding)
        #cyc_out = sin_out + cos_out   # [batch, 1]
        cyc_out = torch.cat([sin_out, cos_out], dim=1) 
        return logits, reg_out, cyc_out
    
"""def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)"""

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Usando un gain más pequeño para evitar valores iniciales grandes
        nn.init.xavier_uniform_(m.weight, gain=0.5)  
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

"""def init_weights(m):
    if isinstance(m, nn.Linear):
        # Inicialización completamente aleatoria con valores razonables
        fan_in = m.weight.size(1)
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)"""
    
#seed = 10
seed = int(time.time())

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Si usas CUDA:
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Verificar si CUDA está disponible
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


diccionario = dataCompiler.load_dicctionaries(16)
dataset_total = dataCompiler.getMultiDataLoader2(diccionario)

# 3. Split en train y val n TODO: añadir semilla, fijar los mismos datos
#seed = 30
seed = int(time.time())

generator = torch.Generator().manual_seed(seed)
total_len = len(dataset_total)
val_len = int(0.1 * total_len)
train_len = total_len - val_len
train_dataset, val_dataset = random_split(dataset_total, [train_len, val_len], generator=generator)

# 4. DataLoaders finales: **IMPORTANTE: pasa tu custom_collate_fn**
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=True, 
    num_workers=8, 
    pin_memory=True, 
    persistent_workers=True,
    collate_fn=dataCompiler.custom_collate5_fn
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=8, 
    pin_memory=True, 
    persistent_workers=True,
    collate_fn=dataCompiler.custom_collate5_fn
)

# == PARAMETROS ==
num_clases = len(class_to_idx)
epochs = 10

# == MODELOS ==
model = PTv3_deteccion(grid_size=0.4).to(device)
model.apply(init_weights)
# == OPTIMIZADOR Y LOSSES ==
"""opt = optim.AdamW(
    list(model.parameters()),
    lr=0.0001
)"""
opt = optim.AdamW(
    list(model.parameters()),
    lr=0.0001,
    weight_decay=1e-4   
)
"""from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(opt, mode='min', patience=1, factor=0.5, min_lr=1e-6)"""

start_epoch = 0
"""if os.path.exists("checkpoint_epoch10.pth"):
    start_epoch = load_checkpoint(model, opt, "checkpoint_epoch10.pth")"""

#criterio = nn.MSELoss()
criterio = nn.SmoothL1Loss(beta=0.2)
#criterio = nn.SmoothL1Loss()
#criterio = nn.L1Loss()
weights = torch.tensor([
    0.1,  # car
    2.0,  # pedestrian
    2.0,  # van
    4.0,  # cyclist
    1e-2, # truck (casi ignora)
    1e-2, # person (casi ignora)
    1e-2, # tram (casi ignora)
    1e-2, # misc (casi ignora)
], dtype=torch.float32, device='cuda')
#weights = torch.tensor(weights, dtype=torch.float32, device='cuda')
criterio_clase = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
num_classes = 8
precisions = []
recalls = []
mean_IoI3Ds  = []

for epoch in range(start_epoch, epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # ----------- ENTRENAMIENTO -----------
    model.train();
    running_loss = 0.0
    contador = 0
    contador_saltos = 0
    #TODO: añadir checkpoints
    opt.zero_grad()
    for entrada, salida, centros, salto in tqdm(train_dataloader):       
        loss_total = 0
        if salto:
            #metrics_logger.info("Salto por outlier")
            contador_saltos += 1
            continue
        try:           
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    metrics_logger.error(f"⚠️ Parámetro {name} contiene NaN/Inf antes de forward (batch {contador})")
            #opt.zero_grad()
            """metrics_logger.info(f"entrada---: {entrada}")
            metrics_logger.info(f"salida---: {salida}")"""
            logits, reg_out, cyc_out = model(entrada)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                metrics_logger.info("⚠️ logits contiene NaN/Inf")
            if torch.isnan(reg_out).any() or torch.isinf(reg_out).any():
                metrics_logger.info("⚠️ reg_out contiene NaN/Inf")
            if torch.isnan(cyc_out).any() or torch.isinf(cyc_out).any():
                metrics_logger.info("⚠️ cyc_out contiene NaN/Inf")
                """metrics_logger.info(f"entrda= {entrada}")
                metrics_logger.info(f"salida= {salida}")"""
            # Prepara los targets
            target_clase   = salida[0][0].unsqueeze(0).long().to(device)
            target_clase = torch.where(
                (target_clase >= 0) & (target_clase < num_classes),
                target_clase,
                torch.full_like(target_clase, -1)
            )
            # Coordenadas relativas al centro del crop
            target_lineal  = salida[0][1:7].unsqueeze(0).to(device) #retocar para solo las dimensiones
            target_ciclico = salida[0][7].view(1,1).to(device)
            """metrics_logger.info(f"salida de clase: {logits}")
            metrics_logger.info(f"target de clase: {target_clase}")"""
            #metrics_logger.info(f"target label: {salida}")
            # Calcula las pérdidas  
            loss_class   = criterio_clase(logits, target_clase)
            loss_lineal  = criterio(reg_out, target_lineal)
            target_sin = torch.sin(target_ciclico)
            target_cos = torch.cos(target_ciclico)
            target_cyc = torch.cat([target_sin, target_cos], dim=1)
            loss_ciclico = criterio(cyc_out, target_cyc)           
            alpha = 1 #TODO: modificar para aumentar o disminuir la importancia de la parte lineal
            beta = 1
            
            #metrics_logger.info(f"clase: {loss_class}")
            # metrics_logger.info(f"lineal: {loss_lineal*alpha}")
            # metrics_logger.info(f"ciclico: {beta*loss_ciclico}")
            loss_total   = loss_class + loss_lineal*alpha + loss_ciclico
            if loss_total > 10:
                metrics_logger.info(f"loss: {loss_total}")
            #loss_total = loss_class + 5.0 * loss_lineal + 5.0 * loss_ciclico
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                error_logger.info(f"clase: {loss_class}")
                error_logger.info(f"lineal: {loss_lineal}")
                error_logger.info(f"ciclico: {loss_ciclico}")
                opt.zero_grad() 
                sys.exit(1)
                continue
            (loss_total / 3).backward()
            #total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            
            if (contador+1) % 3 == 0:
                 total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                 if(total_norm >50):
                    metrics_logger.info(f"total_nomr--->: {total_norm} en iter: {contador}")
                    # metrics_logger.info(f"entrada== {entrada}")
                    # metrics_logger.info(f"entrada== {salida}")
            opt.step()
            opt.zero_grad()
            running_loss += loss_total.item()
            contador = contador +1
        except Exception as e:
            error_logger.exception(e)
            metrics_logger.info(f"entrada= {entrada}")
            metrics_logger.info(f"salida= {salida}")
            continue

    metrics_logger.info(f"----------------------------------------------------------------fin etapa----------------------------------------------")
    print(f"Loss entrenamiento: {running_loss/len(train_dataloader):.4f}")
    metrics_logger.info(f"Loss entrenamiento: {running_loss/len(train_dataloader):.4f}")
    metrics_logger.info(f"saltos totales en epoch: {contador_saltos}")
    # ----------- VALIDACIÓN Y MÉTRICAS -----------
    model.eval();
    val_loss = 0.0
    all_preds, all_targets = [], []
    all_iou3d = []
    with torch.no_grad():
        for entrada, salida, centros, salto in tqdm(val_dataloader):
            contador = contador +1
            try:
                logits, reg_out, cyc_out = model(entrada)
                # Prepara los targets
                target_clase   = salida[0][0].unsqueeze(0).long().to(device)
                target_clase = torch.where(
                    (target_clase >= 0) & (target_clase < num_classes),
                    target_clase,
                    torch.full_like(target_clase, -1)
                )
                target_lineal  = salida[0][1:7].unsqueeze(0).to(device)
                target_ciclico = salida[0][7].view(1,1).to(device)
                # Calcula las pérdidas
                # Backpropagation & update

                loss_class   = criterio_clase(logits, target_clase)
                loss_lineal  = criterio(reg_out, target_lineal)
                target_sin = torch.sin(target_ciclico)
                target_cos = torch.cos(target_ciclico)
                target_cyc = torch.cat([target_sin, target_cos], dim=1)
                loss_ciclico = criterio(cyc_out, target_cyc)       
                alpha = 0.5
                loss_total   =  loss_class + loss_lineal*alpha + loss_ciclico
                val_loss += loss_total.item()
                # Para métricas
                pred_classes = logits.argmax(dim=1).cpu().numpy()
                true_classes = target_clase.cpu().numpy()
                all_preds.extend(pred_classes)
                all_targets.extend(true_classes)

                # ========== IoU 3D por elemento ==========
                # 1. Calcula coordenadas globales sumando el centro del crop
                pred_xyz_global = reg_out[:, :3] + centros[0].to(reg_out.device)          # [1, 3]
                true_xyz_global = target_lineal[:, :3] + centros[0].to(target_lineal.device)  # [1, 3]
                # 2. Crea cajas [x, y, z, w, h, l, alpha] para predicción y ground truth
                # reg_out[:, 3:] --> [w, h, l]
                # cyc_out --> [alpha]
                pred_box = torch.cat([pred_xyz_global, reg_out[:, 3:], cyc_out], dim=1)   # [1, 7]
                true_box = torch.cat([true_xyz_global, target_lineal[:, 3:], target_ciclico], dim=1) # [1, 7]

                # 3. Añade dimensión batch extra si lo pide tu función IoU3D (usualmente espera [B, N, 7])
                pred_boxes = pred_box.unsqueeze(0)   # [1, 1, 7]
                true_boxes = true_box.unsqueeze(0)   # [1, 1, 7]

                # 4. Calcula IoU3D
                ious = IoU3D(pred_boxes, true_boxes).squeeze(0).detach().cpu().numpy()   # [N] o [batch]
                ious = np.clip(ious, 0, 1)
                all_iou3d.extend(ious.tolist())
                """if torch.isnan(pred_box).any() or torch.isinf(pred_box).any():
                    metrics_logger.info(f"IoU pred problematica:{pred_box}")
                    metrics_logger.info(f"IoU pred problematica-> {reg_out}")
                    metrics_logger.info(f"contador = {contador}")
                    metrics_logger.info(entrada)
                    metrics_logger.info(salida)
                if torch.isnan(true_box).any() or torch.isinf(true_box).any():
                    metrics_logger.info(f"IoU true problematica:{true_box}")
                    metrics_logger.info(f"contador = {contador}")
                    metrics_logger.info(entrada)
                    metrics_logger.info(salida)"""

            except Exception as e:
                #print(e)
                error_logger.info(f"contador = {contador}")
                error_logger.info(entrada)
                error_logger.info(salida)
                error_logger.exception(e)
                continue

    #scheduler.step(val_loss)
    # Métricas de validación
    # --- Filtra los ignorados de las métricas si usas ignore_index en CrossEntropyLoss ---
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mask = (all_targets >= 0) & (all_targets < num_clases)
    all_preds = all_preds[mask]
    all_targets = all_targets[mask]

    # Métricas de validación
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    try:
        ap = average_precision_score(
            np.eye(num_clases)[all_targets],
            np.eye(num_clases)[all_preds],
            average=None
        )
    except Exception:
        ap = [0.0] * num_clases

    mean_IoI3D = np.mean(all_iou3d) if all_iou3d else 0.0

    print(f"Loss validación: {val_loss/len(val_dataloader):.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | Mean 3D IoU: {mean_IoI3D:.4f}")
    metrics_logger.info(f"Loss validación: {val_loss/len(val_dataloader):.4f}")
    metrics_logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | Mean 3D IoU: {mean_IoI3D:.4f}")
    for i, class_name in enumerate(class_to_idx):
        print(f'AP ({class_name}): {ap[i]:.4f}')

    precisions.append(precision)
    recalls.append(recall)
    mean_IoI3Ds.append(mean_IoI3D)

    if (epoch +1) % 2 == 0:
        save_checkpoint(model, opt, epoch+1, f"checkpoint_epoch{epoch+1}.pth")



# --- Graficar métricas ---
epochs_range = range(1, len(precisions) + 1)
plt.figure(figsize=(10,6))
plt.plot(epochs_range, precisions, label='Precision', marker='o')
plt.plot(epochs_range, recalls, label='Recall', marker='o')
plt.plot(epochs_range, mean_IoI3Ds, label='Mean 3D IoU', marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Valor')
plt.title('Precision, Recall y Mean 3D IoU por Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("metricas2.png")
plt.show()