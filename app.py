import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURACIÓN DE PÁGINA (UX: Branding) ---
st.set_page_config(
    page_title="StockVision AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS (UI: Estética) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE MODELO (Cacheado para velocidad) ---
@st.cache_resource
def load_generic_model():
    # Usamos el modelo nano oficial, rápido y eficiente para detectar botellas
    return YOLO('yolov8n.pt')

# --- LÓGICA DE DETECCIÓN DE COLOR (El Corazón Híbrido) ---
def detect_brand_color(image_crop):
    """
    Analiza si el recorte de la imagen es mayoritariamente ROJO o AZUL.
    Retorna la marca asociada.
    """
    # Convertir a formato HSV (Matiz, Saturación, Valor) para mejor detección de color
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)

    # Rangos de color ajustados para packaging de bebidas
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([130, 255, 255])

    # Crear máscaras (filtros)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Contar píxeles
    pixels_red = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)
    pixels_blue = cv2.countNonZero(mask_blue)
    total_pixels = image_crop.shape[0] * image_crop.shape[1]

    # Decisión basada en mayoría de color (Heurística)
    # Coca (Rojo) -> Color visualización RGB (255, 0, 0)
    # Pepsi (Azul) -> Color visualización RGB (0, 0, 255)
    if pixels_red > pixels_blue and pixels_red > (total_pixels * 0.05):
        return "Familia Coca-Cola", (255, 0, 0) 
    elif pixels_blue > pixels_red and pixels_blue > (total_pixels * 0.05):
        return "Familia PepsiCo", (0, 0, 255)
    else:
        return "Otros / Genérico", (128, 128, 128)

# --- FUNCIÓN PRINCIPAL ---
def main():
    # --- SIDEBAR (Configuración) ---
    with st.sidebar:
        # Logo o imagen representativa
        st.image("https://cdn-icons-png.flaticon.com/512/3029/3029337.png", width=80)
        st.title("StockVision AI")
        st.markdown("**Herramienta de auditoría de Share of Shelf automatizada.**")
        
        st.divider()
        st.header("⚙️ Configuración")
        conf = st.slider("Sensibilidad IA", 0.1, 0.9, 0.25, help="Ajusta qué tan estricto es el modelo al detectar objetos.")

        st.divider()
        st.markdown("### 📝 Instrucciones")
        st.info("1. Sube foto\n2. Analiza\n3. Exporta reporte")

    # --- CABECERA ---
    st.title("Auditoría de Góndola")
    st.markdown("**Análisis de Share of Shelf (SoS) mediante Visión Artificial Híbrida**")
    
    model = load_generic_model()

    # --- ÁREA DE CARGA ---
    uploaded_file = st.file_uploader("Subir imagen de góndola", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        img_array = np.array(image_pil)

        # Tabs para organizar la información
        tab1, tab2, tab3 = st.tabs(["🖼️ Análisis Visual", "📊 Reporte de Datos", "📥 Exportar"])

        with tab1:
            if st.button("🔍 Ejecutar Análisis", type="primary", use_container_width=True):
                with st.spinner("🤖 Procesando imagen con Arquitectura Híbrida..."):
                    # 1. Detección con YOLO (Solo clase 39: Botellas)
                    results = model(image_pil, conf=conf, classes=[39])
                    
                    img_final = img_array.copy()
                    data_marketing = []

                    # 2. Iterar sobre detecciones
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Validar bordes
                            y1, y2 = max(0, y1), min(img_array.shape[0], y2)
                            x1, x2 = max(0, x1), min(img_array.shape[1], x2)
                            
                            if (x2-x1) < 10 or (y2-y1) < 10: continue

                            # 3. Clasificación con OpenCV (Color)
                            bottle_crop = img_array[y1:y2, x1:x2]
                            brand_name, color_rgb = detect_brand_color(bottle_crop)
                            
                            # 4. Dibujar Bounding Box
                            cv2.rectangle(img_final, (x1, y1), (x2, y2), color_rgb, 2)
                            cv2.rectangle(img_final, (x1, y1-25), (x2, y1), color_rgb, -1)
                            cv2.putText(img_final, brand_name, (x1 + 5, y1 - 8),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                            # 5. Calcular Área (Métrica de Negocio)
                            area = (x2-x1) * (y2-y1)
                            data_marketing.append({"Marca": brand_name, "Area": area})

                    # Guardar en estado de sesión (Persistencia)
                    st.session_state['img_final'] = img_final
                    st.session_state['data_marketing'] = data_marketing
                    st.session_state['analyzed'] = True

            # Mostrar resultado visual
            if st.session_state.get('analyzed'):
                col_orig, col_proc = st.columns(2)
                with col_orig:
                    st.image(image_pil, caption="Imagen Original", use_container_width=True)
                with col_proc:
                    st.image(st.session_state['img_final'], caption="Detección Híbrida", use_container_width=True)

        with tab2:
            if st.session_state.get('analyzed') and st.session_state['data_marketing']:
                df = pd.DataFrame(st.session_state['data_marketing'])
                
                # Agrupar datos por Marca
                df_res = df.groupby("Marca")["Area"].sum().reset_index()
                total_area = df_res["Area"].sum()
                df_res["Share (%)"] = ((df_res["Area"] / total_area) * 100).round(2)
                df_res = df_res.sort_values("Share (%)", ascending=False)

                # KPIs
                kpi1, kpi2, kpi3 = st.columns(3)
                ganador = df_res.iloc[0]
                kpi1.metric("Total Detectados", f"{len(df)} u.")
                kpi2.metric("Líder de Góndola", ganador['Marca'])
                kpi3.metric("Share of Shelf", f"{ganador['Share (%)']}%")

                st.divider()

                # Gráfico
                c_chart, c_table = st.columns([2, 1])
                with c_chart:
                    st.subheader("Participación de Mercado")
                    fig = px.bar(df_res, x="Marca", y="Share (%)", color="Marca",
                                 color_discrete_map={
                                    "Familia Coca-Cola": "#FF6B6B",
                                    "Familia PepsiCo": "#4ECDC4",
                                    "Otros / Genérico": "#95A5A6"
                                 })
                    st.plotly_chart(fig, use_container_width=True)
                
                with c_table:
                    st.subheader("Datos")
                    st.dataframe(df_res[["Marca", "Share (%)"]], hide_index=True, use_container_width=True)

            elif st.session_state.get('analyzed'):
                st.warning("⚠️ No se detectaron productos válidos.")
            else:
                st.info("👈 Ejecuta el análisis primero.")

        with tab3:
            if st.session_state.get('analyzed') and st.session_state['data_marketing']:
                st.success("✅ Reporte generado")
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar CSV", data=csv, file_name='reporte_stockvision.csv', mime='text/csv')
            else:
                st.info("Primero analiza una imagen.")
    else:
        st.info("👆 Sube una foto para comenzar.")
        st.markdown("---")
        st.markdown("Desarrollado con **YOLOv8** + **OpenCV** por Daiana Gomez")

if __name__ == "__main__":
    main()
