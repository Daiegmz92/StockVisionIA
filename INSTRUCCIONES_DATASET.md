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