# StockVision AI - Analisis de Share of Shelf

## Herramienta Profesional de Auditoria Automatizada

### Deteccion Inteligente: YOLOv8 (Forma) + OpenCV (Color)

---

## OBJETIVO DEL PROYECTO

Desarrollar una aplicacion web avanzada que automatice el analisis de participacion en gondola (Share of Shelf) mediante vision artificial hibrida, combinando deteccion de formas con analisis de color para identificar marcas de productos.

La aplicacion proporciona:

1. Conteo automatico de productos en gondolas
2. Clasificacion por color (Rojo para Coca-Cola, Azul para PepsiCo)
3. Calculo de metricas de participacion de mercado
4. Exportacion de reportes en formato CSV
5. Visualizacion profesional con metricas KPI y graficos

---

## INSTALACION Y EJECUCION

### Paso 1: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Paso 2: Ejecutar la Aplicacion
```bash
streamlit run app.py
```

### Paso 3: Acceder
- Abre tu navegador en: http://localhost:8501

---

## INTERFAZ DE USUARIO

### Sidebar de Configuracion
- Logo e informacion del proyecto
- Instrucciones de uso
- Slider de sensibilidad IA (0.1 - 0.9)

### Pestanias Principales

#### Analisis Visual
- Uploader de imagen con drag & drop
- Boton de analisis primario y ancho
- Visualizacion lado a lado: Imagen original vs procesada
- Deteccion en tiempo real con bounding boxes y etiquetas

#### Reporte de Datos
- Metricas KPI: Total productos, Marca lider, Dominio de gondola
- Grafico de barras de participacion de mercado
- Tabla detallada con porcentajes
- Mensajes informativos para casos sin deteccion

#### Exportar
- Boton de descarga de reporte CSV
- Confirmacion de exito del analisis

---

## TECNOLOGIA

### Modelo de Deteccion
- YOLOv8n: Modelo pre-entrenado para deteccion de botellas
- OpenCV: Analisis de color HSV para clasificacion de marcas
- Logica hibrida: Forma + Color para mayor precision

### Clasificacion por Color
- Rojo (Familia Coca-Cola): Coca-Cola, Fanta, Sprite
- Azul (Familia PepsiCo): Pepsi, Seven-Up
- Gris (Otros/Generico): Productos no identificados

---

## METRICAS Y RESULTADOS

### Salida del Analisis
```
Total Productos: 37
Marca Lider: Familia Coca-Cola (35.1%)
Participacion:
- Familia Coca-Cola: 35.1%
- Familia PepsiCo: 29.7%
- Otros/Generico: 35.2%
```

### Exportacion CSV
- Archivo: reporte_share_of_shelf.csv
- Columnas: Marca, Share (%)
- Ordenado por participacion descendente

---

## ARCHIVOS DEL PROYECTO

```
stockvision-marketing/
├── app.py                    # Aplicacion principal Streamlit
├── requirements.txt          # Dependencias Python
├── packages.txt              # Librerias sistema para Streamlit Cloud
├── best.pt                   # Modelo YOLOv8 (opcional)
├── README.md                 # Esta documentacion
├── datasets/data.yaml        # Configuracion dataset (referencia)
├── prepare_dataset.py        # Scripts de preparacion
├── train_yolo_model.py       # Entrenamiento YOLO
├── run_training.py           # Entrenamiento automatico
├── colab_dataset_downloader.ipynb # Descarga masiva dataset
└── [otros scripts auxiliares]
```

---

## DESPLIEGUE EN STREAMLIT CLOUD

### Preparacion
1. Subir a GitHub los archivos principales
2. Crear app en share.streamlit.io
3. Configurar repository y main file path: app.py

### Archivos Requeridos en GitHub
- app.py
- requirements.txt
- packages.txt
- README.md

---

## DEPENDENCIAS

```txt
streamlit
ultralytics
pandas
Pillow
numpy
opencv-python-headless
```

### Librerias Sistema (Linux)
```txt
libgl1-mesa-glx
libglib2.0-0
```

---

## SOPORTE

Problemas de deteccion?
- Ajusta la sensibilidad en el sidebar
- Usa imagenes bien iluminadas
- Evita fondos complejos

Errores de instalacion?
- Verifica Python 3.7+
- Instala dependencias: pip install -r requirements.txt

Despliegue en nube?
- Incluye packages.txt para OpenCV
- Usa opencv-python-headless en requirements.txt

---

## LICENCIA Y CREDITOS

Proyecto Final Integrador - IFTS N24
Creado por: Daiana E. Gomez

Proyecto de demostracion - StockVision AI
Tecnologias: YOLOv8, Streamlit, OpenCV
Fecha: Noviembre 2024

---

Tu herramienta de analisis de gondolas esta lista para usar!

Ejecuta streamlit run app.py y comienza a auditar tus gondolas automaticamente.