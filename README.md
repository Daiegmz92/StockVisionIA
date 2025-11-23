# 📊 StockVision AI - Análisis de Share of Shelf

## **Herramienta Profesional de Auditoría Automatizada**

### **Detección Inteligente: YOLOv8 (Forma) + OpenCV (Color)**

---

## 🎯 **OBJETIVO DEL PROYECTO**

Desarrollar una aplicación web avanzada que automatice el análisis de participación en góndola (Share of Shelf) mediante visión artificial híbrida, combinando detección de formas con análisis de color para identificar marcas de productos.

**La aplicación proporciona:**

1. **📦 Conteo automático** de productos en góndolas
2. **🎨 Clasificación por color** (Rojo para Coca-Cola, Azul para PepsiCo)
3. **📊 Cálculo de métricas** de participación de mercado
4. **📥 Exportación de reportes** en formato CSV
5. **📈 Visualización profesional** con métricas KPI y gráficos

---

## 🚀 **INSTALACIÓN Y EJECUCIÓN**

### **Paso 1: Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### **Paso 2: Ejecutar la Aplicación**
```bash
streamlit run app.py
```

### **Paso 3: Acceder**
- Abre tu navegador en: `http://localhost:8501`

---

## 🖼️ **INTERFAZ DE USUARIO**

### **Sidebar de Configuración**
- **Logo e información** del proyecto
- **Instrucciones** de uso
- **Slider de sensibilidad** IA (0.1 - 0.9)

### **Pestañas Principales**

#### **🖼️ Análisis Visual**
- **Uploader de imagen** con drag & drop
- **Botón de análisis** primario y ancho
- **Visualización lado a lado**: Imagen original vs procesada
- **Detección en tiempo real** con bounding boxes y etiquetas

#### **📊 Reporte de Datos**
- **Métricas KPI**: Total productos, Marca líder, Dominio de góndola
- **Gráfico de barras** de participación de mercado
- **Tabla detallada** con porcentajes
- **Mensajes informativos** para casos sin detección

#### **📥 Exportar**
- **Botón de descarga** de reporte CSV
- **Confirmación de éxito** del análisis

---

## 🤖 **TECNOLOGÍA**

### **Modelo de Detección**
- **YOLOv8n**: Modelo pre-entrenado para detección de botellas
- **OpenCV**: Análisis de color HSV para clasificación de marcas
- **Lógica híbrida**: Forma + Color para mayor precisión

### **Clasificación por Color**
- **Rojo (Familia Coca-Cola)**: Coca-Cola, Fanta, Sprite
- **Azul (Familia PepsiCo)**: Pepsi, Seven-Up
- **Gris (Otros/Genérico)**: Productos no identificados

---

## 📊 **MÉTRICAS Y RESULTADOS**

### **Salida del Análisis**
```
📦 Total Productos: 37
🏆 Marca Líder: Familia Coca-Cola (35.1%)
📊 Participación:
- Familia Coca-Cola: 35.1%
- Familia PepsiCo: 29.7%
- Otros/Genérico: 35.2%
```

### **Exportación CSV**
- **Archivo**: reporte_share_of_shelf.csv
- **Columnas**: Marca, Share (%)
- **Ordenado** por participación descendente

---

## 📋 **ARCHIVOS DEL PROYECTO**

```
stockvision-marketing/
├── app.py                    # Aplicación principal Streamlit
├── requirements.txt          # Dependencias Python
├── packages.txt              # Librerías sistema para Streamlit Cloud
├── best.pt                   # Modelo YOLOv8 (opcional)
├── README.md                 # Esta documentación
├── datasets/data.yaml        # Configuración dataset (referencia)
├── prepare_dataset.py        # Scripts de preparación
├── train_yolo_model.py       # Entrenamiento YOLO
├── run_training.py           # Entrenamiento automático
├── colab_dataset_downloader.ipynb # Descarga masiva dataset
└── [otros scripts auxiliares]
```

---

## 🚀 **DESPLIEGUE EN STREAMLIT CLOUD**

### **Preparación**
1. **Subir a GitHub** los archivos principales
2. **Crear app** en share.streamlit.io
3. **Configurar** repository y main file path: app.py

### **Archivos Requeridos en GitHub**
- app.py
- requirements.txt
- packages.txt
- README.md

---

## 🔧 **DEPENDENCIAS**

```txt
streamlit
ultralytics
pandas
Pillow
numpy
opencv-python-headless
```

### **Librerías Sistema (Linux)**
```txt
libgl1-mesa-glx
libglib2.0-0
```

---

## 📞 **SOPORTE**

**¿Problemas de detección?**
- Ajusta la sensibilidad en el sidebar
- Usa imágenes bien iluminadas
- Evita fondos complejos

**¿Errores de instalación?**
- Verifica Python 3.7+
- Instala dependencias: `pip install -r requirements.txt`

**¿Despliegue en nube?**
- Incluye packages.txt para OpenCV
- Usa opencv-python-headless en requirements.txt

---

## 📝 **LICENCIA**

Proyecto de demostración - StockVision AI
Tecnologías: YOLOv8, Streamlit, OpenCV
Fecha: Noviembre 2024

---

**🌟 ¡Tu herramienta de análisis de góndolas está lista para usar!**

Ejecuta `streamlit run app.py` y comienza a auditar tus góndolas automáticamente.