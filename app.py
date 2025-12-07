import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pandas as pd
import io
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
        color: #2c3e50 !important;
    }
    .stMetric label {
        color: #2c3e50 !important;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #2c3e50 !important;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE MODELO (Cacheado para velocidad) ---
@st.cache_resource
def load_generic_model():
    return YOLO('yolov8n.pt')

# --- LÓGICA DE DETECCIÓN DE COLOR ---
def detect_brand_color(image_crop):
    # Convertir a HSV
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)

    # Rangos (Mantenemos tu lógica original)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([130, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    pixels_red = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)
    pixels_blue = cv2.countNonZero(mask_blue)
    total_pixels = image_crop.shape[0] * image_crop.shape[1]

    # Corrección de colores para visualización en RGB (Streamlit usa RGB)
    # Coca (Rojo) -> (255, 0, 0)
    # Pepsi (Azul) -> (0, 0, 255)
    if pixels_red > pixels_blue and pixels_red > (total_pixels * 0.05):
        return "Familia Coca-Cola", (255, 0, 0) 
    elif pixels_blue > pixels_red and pixels_blue > (total_pixels * 0.05):
        return "Familia PepsiCo", (0, 0, 255)
    else:
        return "Otros / Genérico", (128, 128, 128)

# --- FUNCIÓN PRINCIPAL ---
def main():
    # --- SIDEBAR (Configuración y Ayuda) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3029/3029337.png", width=80)
        st.title("StockVision AI")
        st.markdown("**Herramienta de auditoría de Share of Shelf automatizada.**")

        st.divider()
        st.header("⚙️ Configuración")
        conf = st.slider("Sensibilidad IA", 0.1, 0.9, 0.25, help="Valores más altos detectan menos objetos pero con mayor precisión.")

        st.divider()
        st.markdown("### 📝 Instrucciones")
        st.markdown("""
        1. 📤 Sube una foto de la góndola.
        2. 🔍 El sistema detectará botellas.
        3. 🎨 Clasificará por color (Rojo/Azul).
        4. 📊 Revisa el reporte final.
        """)

    # --- CABECERA PRINCIPAL ---
    col_title, col_logo = st.columns([3, 1])
    with col_title:
        st.title("Auditoría de Góndola")
        st.markdown("**Análisis de Share of Shelf (SoS) mediante Visión Artificial Híbrida**")
    
    # Cargar modelo (Feedback visual silencioso)
    model = load_generic_model()

    # --- ÁREA DE CARGA (UX: Drag & Drop claro) ---
    uploaded_file = st.file_uploader("Subir imagen", type=['jpg', 'jpeg', 'png'], help="Arrastra tu imagen aquí", label_visibility="collapsed")

    if uploaded_file:
        # Procesamiento inicial
        image_pil = Image.open(uploaded_file)
        img_array = np.array(image_pil)

        # UX: Usar Tabs para organizar la vista "Antes" y "Después"
        tab1, tab2, tab3 = st.tabs(["🖼️ Análisis Visual", "📊 Reporte de Datos", "📥 Exportar"])

        with tab1:
            if st.button("🔍 Ejecutar Análisis", type="secondary", use_container_width=True):
                with st.spinner("🤖 La IA está auditando la góndola..."):
                    # Detección
                    results = model(image_pil, conf=conf, classes=[39])
                    
                    img_final = img_array.copy()
                    data_marketing = []
                    total_area_px = 0

                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Validar límites
                            y1, y2 = max(0, y1), min(img_array.shape[0], y2)
                            x1, x2 = max(0, x1), min(img_array.shape[1], x2)
                            
                            # Filtro de tamaño (evitar ruido muy pequeño)
                            if (x2-x1) < 10 or (y2-y1) < 10: continue

                            bottle_crop = img_array[y1:y2, x1:x2]
                            brand_name, color_rgb = detect_brand_color(bottle_crop)
                            
                            # Dibujar (Líneas más finas para mejor legibilidad)
                            cv2.rectangle(img_final, (x1, y1), (x2, y2), color_rgb, 2)
                            
                            # Fondo para el texto (Legibilidad)
                            cv2.rectangle(img_final, (x1, y1-25), (x2, y1), color_rgb, -1)
                            cv2.putText(img_final, brand_name, (x1 + 5, y1 - 8),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                            area = (x2-x1) * (y2-y1)
                            data_marketing.append({"Marca": brand_name, "Area": area})
                            total_area_px += area

                    # Guardar resultados en session_state para no perderlos al cambiar de tab
                    st.session_state['img_final'] = img_final
                    st.session_state['data_marketing'] = data_marketing
                    st.session_state['analyzed'] = True

            # Mostrar resultados si ya se analizó
            if st.session_state.get('analyzed'):
                col_orig, col_proc = st.columns(2)
                with col_orig:
                    st.image(image_pil, caption="Góndola Original", width='stretch')
                with col_proc:
                    st.image(st.session_state['img_final'], caption="Detección & Clasificación", width='stretch')

        with tab2:
            if st.session_state.get('analyzed') and st.session_state['data_marketing']:
                df = pd.DataFrame(st.session_state['data_marketing'])
                
                # Agrupación
                df_res = df.groupby("Marca")["Area"].sum().reset_index()
                total_area = df_res["Area"].sum()
                df_res["Share (%)"] = ((df_res["Area"] / total_area) * 100).round(2)
                df_res = df_res.sort_values("Share (%)", ascending=False)

                # --- KPI CARDS (UX: Información Clave Primero) ---
                kpi1, kpi2, kpi3 = st.columns(3)
                
                ganador = df_res.iloc[0]
                count_total = len(df)
                
                kpi1.metric("Total Productos", f"{count_total} u.")
                kpi2.metric("Marca Líder", ganador['Marca'])
                kpi3.metric("Dominio de Góndola", f"{ganador['Share (%)']}%")

                st.divider()

                # Gráficos y Tablas
                c_chart, c_table = st.columns([2, 1])
                
                with c_chart:
                    st.subheader("Participación de Mercado (Visual)")
                    # Gráfico de barras con colores según la marca
                    fig = px.bar(
                        df_res,
                        x="Marca",
                        y="Share (%)",
                        color="Marca",
                        color_discrete_map={
                            "Familia Coca-Cola": "#FF6B6B",  # Rojo suave
                            "Familia PepsiCo": "#4ECDC4",    # Azul verdoso
                            "Otros / Genérico": "#95A5A6"    # Gris suave
                        }
                    )
                    fig.update_layout(showlegend=False)  # Ocultar leyenda para simplicidad
                    st.plotly_chart(fig, use_container_width=True)
                
                with c_table:
                    st.subheader("Detalle")
                    st.dataframe(
                        df_res[["Marca", "Share (%)"]],
                        hide_index=True,
                        width='stretch'
                    )
            elif st.session_state.get('analyzed'):
                st.warning("⚠️ No se detectaron productos válidos. Intenta con otra imagen o ajusta la sensibilidad.")
            else:
                st.info("👈 Ejecuta el análisis en la primera pestaña para ver los datos.")

        with tab3:
            if st.session_state.get('analyzed') and st.session_state['data_marketing']:
                st.success("✅ Reporte listo para descarga")
                
                # Preparar CSV
                csv = df_res.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Descargar Reporte CSV",
                    data=csv,
                    file_name='reporte_share_of_shelf.csv',
                    mime='text/csv',
                )
            else:
                st.info("Primero debes analizar una imagen.")

    else:
        # Estado vacío limpio
        st.info("👆 Sube una foto para comenzar la auditoría.")
        # Quitamos la imagen que daba error y ponemos un texto simple
        st.markdown("---")
        st.markdown("Desarrollado con **YOLOv8** + **OpenCV**")
        st.markdown("**Desarrollado por Daiana Elizabeth Gomez**")

if __name__ == "__main__":
    main()
