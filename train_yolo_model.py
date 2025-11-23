#!/usr/bin/env python3
"""
Script de Entrenamiento YOLOv8 para Detección de Productos en Góndolas
Proyecto Integrador - IFTS24
"""

import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path

def verificar_gpu():
    """Verifica si hay GPU disponible"""
    print("🔍 Verificando GPU...")
    if torch.cuda.is_available():
        print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"📊 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("⚠️ No hay GPU disponible. El entrenamiento será más lento en CPU.")
        return False

def crear_config_dataset():
    """Crea archivo de configuración data.yaml para el dataset"""
    # Usar la configuración existente si ya existe
    data_yaml_path = 'datasets/data.yaml'
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Usando configuración existente de data.yaml")
        return config

    # Configuración por defecto si no existe
    config = {
        'path': './datasets',  # Ruta al dataset
        'train': 'images/train',  # Imágenes de entrenamiento
        'val': 'images/val',     # Imágenes de validación
        'test': 'images/test',   # Imágenes de test (opcional)

        'names': {
            0: 'coca-cola',
            1: 'sprite',
            2: 'fanta',
            3: 'pepsi',
            4: 'seven-up',
            5: 'bottle',  # Clase genérica para fallback
        },

        'nc': 6,  # Número de clases
    }

    # Crear directorio si no existe
    os.makedirs('datasets', exist_ok=True)

    # Guardar configuración
    with open('datasets/data.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print("✅ Archivo data.yaml creado en datasets/data.yaml")
    return config

def crear_estructura_dataset():
    """Crea la estructura de directorios para el dataset"""
    dirs = [
        'datasets/images/train',
        'datasets/images/val',
        'datasets/images/test',
        'datasets/labels/train',
        'datasets/labels/val',
        'datasets/labels/test'
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    print("✅ Estructura de dataset creada:")
    for dir_path in dirs:
        print(f"  📁 {dir_path}")

def descargar_dataset_ejemplo():
    """Descarga un dataset de ejemplo pequeño para pruebas"""
    print("🔄 Descargando dataset de ejemplo...")

    # Crear algunas imágenes de ejemplo (simuladas)
    # En un caso real, aquí descargarías un dataset como SKU-110K o similar

    print("ℹ️ Para un entrenamiento real, necesitas:")
    print("  📸 Imágenes de góndolas con productos")
    print("  📝 Archivos de anotación (.txt) con bounding boxes")
    print("  🏷️ Formato YOLO: clase_id x_center y_center width height")

    # Crear archivo de ejemplo
    ejemplo_txt = """# Ejemplo de anotación YOLO (formato: clase_id x_center y_center width height)
# Coordenadas normalizadas (0-1)
# clase 0 = coca-cola, 1 = sprite, etc.

# Ejemplo de anotaciones para una imagen:
# 0 0.5 0.5 0.2 0.3  # coca-cola en el centro
# 1 0.2 0.8 0.15 0.25 # sprite abajo a la izquierda
"""

    with open('datasets/ejemplo_anotaciones.txt', 'w') as f:
        f.write(ejemplo_txt)

    print("✅ Archivo de ejemplo creado: datasets/ejemplo_anotaciones.txt")

def entrenar_modelo(config, epochs=50, batch_size=16):
    """Entrena el modelo YOLOv8"""

    print("🚀 Iniciando entrenamiento...")
    print(f"📊 Épocas: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🎯 Número de clases: {config['nc']}")

    # Cargar modelo base
    try:
        model = YOLO('yolov8n.pt')  # Modelo nano como base
        print("✅ Modelo base cargado: yolov8n.pt")
    except Exception as e:
        print(f"❌ Error cargando modelo base: {e}")
        return None

    # Configurar entrenamiento
    try:
        results = model.train(
            data='datasets/data.yaml',
            epochs=epochs,
            batch=batch_size,
            imgsz=640,  # Tamaño de imagen
            save=True,
            save_period=10,  # Guardar cada 10 épocas
            cache=False,  # Cache de imágenes
            device='cpu',  # Usar CPU
            workers=4,  # Número de workers para data loading
            project='runs/train',  # Directorio de resultados
            name='stock_counter',  # Nombre del experimento
            exist_ok=True,  # Sobrescribir si existe
            pretrained=True,  # Usar pesos pre-entrenados
            optimizer='auto',  # Optimizador automático
            verbose=True,  # Output detallado
            seed=42,  # Para reproducibilidad
        )

        print("✅ Entrenamiento completado!")
        print(f"📁 Resultados guardados en: runs/train/stock_counter/")

        # Cargar mejor modelo entrenado
        best_model_path = 'runs/train/stock_counter/weights/best.pt'
        if os.path.exists(best_model_path):
            print(f"🏆 Mejor modelo guardado en: {best_model_path}")
            print("💡 Copia este archivo a la carpeta raíz como 'best.pt' para usarlo en la app")

        return results

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return None

def validar_modelo():
    """Valida el modelo entrenado"""
    try:
        model = YOLO('runs/train/stock_counter/weights/best.pt')
        print("✅ Modelo cargado para validación")

        # Ejecutar validación
        results = model.val()
        print("✅ Validación completada")
        return results

    except Exception as e:
        print(f"❌ Error en validación: {e}")
        return None

def main():
    print("="*60)
    print("🤖 ENTRENAMIENTO YOLOv8 - DETECCIÓN DE PRODUCTOS EN GÓNDOLAS")
    print("="*60)
    print("📚 Proyecto Integrador - IFTS24")

    # Verificar GPU
    gpu_disponible = verificar_gpu()

    # Crear configuración del dataset
    print("\n📋 Creando configuración del dataset...")
    config = crear_config_dataset()

    # Crear estructura de directorios
    print("\n🏗️ Creando estructura del dataset...")
    crear_estructura_dataset()

    # Descargar dataset de ejemplo
    print("\n📥 Preparando dataset de ejemplo...")
    descargar_dataset_ejemplo()

    # Preguntar si quiere entrenar
    print("\n" + "="*60)
    print("❓ CONFIGURACIÓN DE ENTRENAMIENTO")
    print("="*60)

    try:
        epochs = 50  # Default
        batch_size = 8  # Default

        print("\n🚀 CONFIGURACIÓN FINAL:")
        print(f"  📊 Épocas: {epochs}")
        print(f"  📦 Batch size: {batch_size}")
        print(f"  🎯 Clases: {config['nc']} ({', '.join(config['names'].values())})")
        print(f"  💻 GPU: {'Sí' if gpu_disponible else 'No'}")

        confirmar = 's'  # Auto confirm

        if confirmar in ['s', 'si', 'yes', 'y']:
            # Entrenar modelo
            results = entrenar_modelo(config, epochs, batch_size)

            if results:
                print("\n🎉 ¡Entrenamiento exitoso!")
                print("📁 Revisa la carpeta 'runs/train/stock_counter/' para ver los resultados")

                # Ofrecer validación
                validar = input("❓ ¿Quieres validar el modelo entrenado? (s/n): ").lower().strip()
                if validar in ['s', 'si', 'yes', 'y']:
                    validar_modelo()

        else:
            print("❌ Entrenamiento cancelado")

    except KeyboardInterrupt:
        print("\n👋 Entrenamiento cancelado por el usuario")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "="*60)
    print("📚 RECURSOS PARA ENTRENAMIENTO REAL")
    print("="*60)
    print("Para un entrenamiento real necesitas:")
    print("📸 Dataset de imágenes de góndolas")
    print("🏷️ Anotaciones en formato YOLO (.txt)")
    print("💪 GPU recomendada para mejor rendimiento")
    print("⏰ 1-2 horas de entrenamiento típico")
    print("="*60)

if __name__ == "__main__":
    main()