#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Preparar Dataset de Productos en Gondolas
Proyecto Integrador - IFTS24
"""

import os
import shutil
from pathlib import Path

def crear_estructura_basica():
    """Crea la estructura basica de directorios para el dataset"""
    print("Creando estructura de dataset...")

    # Estructura YOLO estandar
    estructura = {
        'datasets/images/train': 'Imagenes de entrenamiento',
        'datasets/images/val': 'Imagenes de validacion',
        'datasets/images/test': 'Imagenes de test',
        'datasets/labels/train': 'Etiquetas de entrenamiento (.txt)',
        'datasets/labels/val': 'Etiquetas de validacion (.txt)',
        'datasets/labels/test': 'Etiquetas de test (.txt)'
    }

    for directorio, descripcion in estructura.items():
        os.makedirs(directorio, exist_ok=True)
        print(f"  [OK] {directorio} - {descripcion}")

    print("Estructura de dataset creada exitosamente")

def crear_data_yaml():
    """Crea el archivo data.yaml con la configuracion del dataset"""
    print("\nCreando archivo de configuracion data.yaml...")

    data_yaml = """
# Configuracion del Dataset para Deteccion de Productos en Gondolas
# Proyecto Integrador - IFTS24

path: ./datasets  # Ruta raiz del dataset
train: images/train  # Imagenes de entrenamiento
val: images/val      # Imagenes de validacion
test: images/test    # Imagenes de test (opcional)

# Nombres de las clases (productos a detectar)
names:
  0: coca-cola
  1: sprite
  2: fanta
  3: pepsi
  4: seven-up
  5: bottle      # Clase generica para fallback

nc: 6  # Numero total de clases
"""

    with open('datasets/data.yaml', 'w', encoding='utf-8') as f:
        f.write(data_yaml.strip())

    print("Archivo data.yaml creado en datasets/data.yaml")

def crear_ejemplo_anotaciones():
    """Crea ejemplos de archivos de anotacion en formato YOLO"""
    print("\nCreando ejemplos de anotaciones...")

    # Ejemplo de anotacion para una imagen
    ejemplo_anotacion = """# Formato YOLO: clase_id x_center y_center width height
# Todas las coordenadas estan normalizadas (0-1)
# clase_id: 0=coca-cola, 1=sprite, 2=fanta, 3=pepsi, 4=seven-up, 5=bottle

# Ejemplo: Coca-Cola en el centro de la imagen
0 0.5 0.5 0.2 0.3

# Ejemplo: Sprite abajo a la izquierda
1 0.2 0.8 0.15 0.25

# Ejemplo: Fanta arriba a la derecha
2 0.8 0.2 0.18 0.28
"""

    with open('datasets/ejemplo_anotacion.txt', 'w', encoding='utf-8') as f:
        f.write(ejemplo_anotacion)

    print("Archivo de ejemplo creado: datasets/ejemplo_anotacion.txt")

def crear_instrucciones_dataset():
    """Crea archivo con instrucciones para preparar el dataset"""
    print("\nCreando instrucciones para el dataset...")

    instrucciones = """
INSTRUCCIONES PARA PREPARAR TU DATASET
Proyecto Integrador - IFTS24

1. RECOPILAR IMAGENES
Necesitas fotos de gondolas con productos. Recomendaciones:
- Minimo 100 imagenes por clase para buen entrenamiento
- Variedad de angulos, iluminacion y posiciones
- Resolucion minima: 640x640 pixeles
- Formatos: JPG, JPEG, PNG

2. ESTRUCTURA DE ARCHIVOS
Coloca tus archivos asi:
datasets/
├── images/
│   ├── train/     # 70% de tus imagenes
│   ├── val/       # 20% de tus imagenes
│   └── test/      # 10% de tus imagenes
└── labels/
    ├── train/     # Archivos .txt correspondientes
    ├── val/       # Archivos .txt correspondientes
    └── test/      # Archivos .txt correspondientes

3. ANOTACIONES (ETIQUETAS)
Para cada imagen, necesitas un archivo .txt con el mismo nombre:
- imagen1.jpg → imagen1.txt
- gondola_001.png → gondola_001.txt

Formato de cada linea en el archivo .txt:
clase_id x_center y_center width height

Donde:
- clase_id: numero de clase (0=coca-cola, 1=sprite, etc.)
- x_center, y_center: coordenadas del centro del bounding box (0-1)
- width, height: ancho y alto del bounding box (0-1)

4. EJEMPLOS DE ANOTACIONES

Archivo: imagen1.txt
Contenido:
0 0.5 0.5 0.2 0.3    # Coca-Cola: centro, 20% ancho, 30% alto
1 0.2 0.8 0.15 0.25  # Sprite: abajo-izq, 15% ancho, 25% alto

5. HERRAMIENTAS PARA ANOTAR
- LabelImg (gratuito): https://github.com/heartexlabs/labelImg
- Roboflow (web): https://roboflow.com/
- CVAT (avanzado): https://cvat.org/

6. VALIDACION
Despues de anotar, verifica:
- Cada imagen tiene su archivo .txt correspondiente
- Las coordenadas estan entre 0 y 1
- Los clase_id son correctos (0-5)
- No hay bounding boxes vacios

7. ENTRENAMIENTO
Una vez preparado el dataset, ejecuta:
python train_yolo_model.py

Exito con tu proyecto integrador!
"""

    with open('INSTRUCCIONES_DATASET.md', 'w', encoding='utf-8') as f:
        f.write(instrucciones.strip())

    print("Instrucciones creadas: INSTRUCCIONES_DATASET.md")

def verificar_estructura():
    """Verifica que la estructura del dataset este correcta"""
    print("\nVerificando estructura del dataset...")

    directorios_requeridos = [
        'datasets/images/train',
        'datasets/images/val',
        'datasets/labels/train',
        'datasets/labels/val'
    ]

    archivos_requeridos = [
        'datasets/data.yaml',
        'INSTRUCCIONES_DATASET.md'
    ]

    print("Directorios:")
    for directorio in directorios_requeridos:
        if os.path.exists(directorio):
            print(f"  [OK] {directorio}")
        else:
            print(f"  [ERROR] {directorio} - NO EXISTE")

    print("\nArchivos:")
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            print(f"  [OK] {archivo}")
        else:
            print(f"  [ERROR] {archivo} - NO EXISTE")

def main():
    print("="*60)
    print("PREPARACION DE DATASET - PRODUCTOS EN GONDOLAS")
    print("="*60)
    print("Proyecto Integrador - IFTS24")

    # Crear estructura
    crear_estructura_basica()

    # Crear configuracion
    crear_data_yaml()

    # Crear ejemplos
    crear_ejemplo_anotaciones()

    # Crear instrucciones
    crear_instrucciones_dataset()

    # Verificar
    verificar_estructura()

    print("\n" + "="*60)
    print("PREPARACION COMPLETADA!")
    print("="*60)
    print("Estructura de dataset lista en carpeta 'datasets/'")
    print("Lee las instrucciones en: INSTRUCCIONES_DATASET.md")
    print("Agrega tus imagenes y anotaciones")
    print("Luego ejecuta: python train_yolo_model.py")
    print("="*60)

if __name__ == "__main__":
    main()