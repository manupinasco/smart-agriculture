import sys

sys.path.append("..")

import kagglehub
from logging_config import logger
import shutil
import os

FINAL_PATH="../data/raw"
DATASET_NAME="smart-agriculture.csv"

try:
    # Descargar última versión del dataset
    source_path = kagglehub.dataset_download(
        "chitrakumari25/smart-agricultural-production-optimizing-engine"
    )
    
    
except Exception as er:
    logger.error(f"Hubo un error en la carga del dataset: {str(er)}")
    raise

items=os.listdir(source_path)

subfolders = [i for i in items if os.path.isdir(os.path.join(source_path, i))]

if subfolders:
    
    # Si hay subcarpetas, usamos la primera
    file_dir = os.path.join(source_path, subfolders[0])
    
else:
    file_dir = source_path

files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]

if not files:
    logger.error("No se encontró ningún dataset para procesar")
    raise FileNotFoundError("No se encontró ningún archivo para procesar")

old_path=os.path.join(file_dir,files[0])
new_path=os.path.join(file_dir,DATASET_NAME)

# Renombramos el dataset
os.rename(old_path,new_path)

# Movemos el dataset
shutil.move(new_path,FINAL_PATH)

if subfolders:
    os.rmdir(file_dir)
