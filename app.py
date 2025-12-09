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
    .main {background-color: #f8f9fa;}
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {color: #2c3e50;}
    /* Ajuste para que los tabs se vean mejor */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #eef2f6;
        border-bottom: 2px solid #4ECDC4;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE MODELO ---
@st.cache_resource
def load_generic_model():
    return YOLO('yolov8n.pt')

# --- LÓGICA DE DETECCIÓN DE COLOR ---
def detect_brand_color(image_crop):
    """Detecta si predomina Rojo o Azul en el recorte."""
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)

    # Rangos Coca-Cola (Rojo)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Rangos Pepsi (Azul)
    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([130, 255, 255])

    # Máscaras
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    pixels_red = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)
    pixels_blue = cv2.countNonZero(mask_blue)
    total_pixels = image_crop.shape[0] * image_crop.shape[1]

    # Umbral: al menos 5% del producto debe tener el color
    if pixels_red > pixels_blue and pixels_red > (total_pixels * 0.05):
        return "Familia Coca-Cola", (255, 0, 0) # Texto, Color RGB (Rojo)
    elif pixels_blue > pixels_red and pixels_blue > (total_pixels * 0.05):
        return "Familia PepsiCo", (0, 0, 255)   # Texto, Color RGB (Azul)
    else:
        return "Otros / Genérico", (128, 128, 128) # Gris

# --- FUNCIÓN PRINCIPAL ---
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3029/3029337.png", width=80)
        st.title("StockVision AI")
        st.markdown("**Herramienta de auditoría de Share of Shelf automatizada.**")
        st.divider()
        st.header("⚙️ Configuración")
        conf = st.slider("Sensibilidad IA", 0.1, 0.9, 0.25)
        st.divider()
        st.info("1. Sube foto\n2. Analiza\n3. Ve al reporte")

    # --- CABECERA ---
    st.title("Auditoría de Góndola")
    st.markdown("**Análisis de Share of Shelf (SoS) mediante Visión Artificial Híbrida**")
    
    model = load_generic_model()

    # --- ESTADO DE SESIÓN (CRÍTICO PARA QUE NO SE BORRE EL ANÁLISIS) ---
    if 'analyzed' not in st.session_state:
        st.session_state['analyzed'] = False
    if 'data_marketing' not in st.session_state:
        st.session_state['data_marketing'] = []
    if 'img_final' not in st.session_state:
        st.session_state['img_final'] = None

    # --- ÁREA DE CARGA ---
    uploaded_file = st.file_uploader("Sube tu foto aquí", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        img_array = np.array(image_pil)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["🖼️ Análisis Visual", "📊 Reporte de Negocio", "📥 Exportar"])

        # --- TAB 1: VISUAL ---
        with tab1:
            col_btn, col_info = st.columns([1, 2])
            with col_btn:
                analizar = st.button("🔍 EJECUTAR ANÁLISIS", type="primary", use_container_width=True)
            
            if analizar:
                with st.spinner("🤖 La IA está auditando la góndola..."):
                    # 1. Detección YOLO
                    results = model(image_pil, conf=conf, classes=[39]) # 39 = bottle
                    
                    img_final = img_array.copy()
                    data_temp = []

                    # 2. Procesamiento Híbrido
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Validar límites y tamaño
                            y1, y2 = max(0, y1), min(img_array.shape[0], y2)
                            x1, x2 = max(0, x1), min(img_array.shape[1], x2)
                            if (x2-x1) < 10 or (y2-y1) < 10: continue

                            # Colorimetría
                            bottle_crop = img_array[y1:y2, x1:x2]
                            brand_name, color_rgb = detect_brand_color(bottle_crop)
                            
                            # Dibujar
                            cv2.rectangle(img_final, (x1, y1), (x2, y2), color_rgb, 2)
                            # Etiqueta legible
                            cv2.rectangle(img_final, (x1, y1-20), (x2, y1), color_rgb, -1)
                            cv2.putText(img_final, brand_name, (x1+2, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

                            # Guardar Datos
                            area = (x2-x1) * (y2-y1)
                            data_temp.append({"Marca": brand_name, "Area": area})

                    # GUARDAR EN SESIÓN
                    st.session_state['img_final'] = img_final
                    st.session_state['data_marketing'] = data_temp
                    st.session_state['analyzed'] = True

            # Mostrar Resultados Visuales (Si existen en memoria)
            if st.session_state['analyzed']:
                c1, c2 = st.columns(2)
                c1.image(image_pil, caption="Original", use_container_width=True)
                c2.image(st.session_state['img_final'], caption="Procesada por IA", use_container_width=True)
            else:
                st.info("Presiona el botón para comenzar el análisis.")

        # --- TAB 2: REPORTE DE NEGOCIO (Aquí están las mejoras visuales) ---
        with tab2:
            if st.session_state['analyzed'] and st.session_state['data_marketing']:
                df = pd.DataFrame(st.session_state['data_marketing'])
                
                # Cálculos
                df_res = df.groupby("Marca")["Area"].sum().reset_index()
                total_area = df_res["Area"].sum()
                df_res["Share (%)"] = ((df_res["Area"] / total_area) * 100).round(2)
                df_res = df_res.sort_values("Share (%)", ascending=False)
                
                ganador = df_res.iloc[0]

                # 1. KPIs
                st.markdown("### 🎯 Métricas Clave")
                k1, k2, k3 = st.columns(3)
                k1.metric("Productos Detectados", f"{len(df)} u.")
                k2.metric("Marca Líder", ganador['Marca'])
                k3.metric("Dominio de Góndola", f"{ganador['Share (%)']}%")
                
                st.divider()

                # 2. SECCIÓN VISUAL Y ECONÓMICA
                col_chart, col_money = st.columns([1.5, 1])
                
                with col_chart:
                    st.subheader("Participación de Mercado")
                    # Gráfico de Donut (Más moderno)
                    fig = px.pie(
                        df_res, 
                        values='Area', 
                        names='Marca',
                        hole=0.4, 
                        color='Marca',
                        color_discrete_map={
                            "Familia Coca-Cola": "#FF0000",
                            "Familia PepsiCo": "#004B93",
                            "Otros / Genérico": "#95A5A6"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_money:
                    st.subheader("💵 Valorización")
                    st.markdown("Calculadora de stock exhibido:")
                    precio = st.number_input("Precio Promedio ($)", value=1500, step=100)
                    total_valor = len(df) * precio
                    st.metric("Valor Total Estimado", f"$ {total_valor:,.0f}")
                    st.caption("*Basado en unidades detectadas.*")
                    
                    st.markdown("#### Detalle")
                    st.dataframe(df_res[["Marca", "Share (%)"]], hide_index=True)

                # 3. RESUMEN EJECUTIVO (IA Insight)
                st.divider()
                st.subheader("📋 Resumen Ejecutivo")
                txt_resumen = f"""
                El análisis detectó **{len(df)} productos**. La marca dominante es **{ganador['Marca']}** con un **{ganador['Share (%)']}%**.
                """
                if ganador['Share (%)'] < 40:
                    txt_resumen += " ⚠️ **Alerta:** Presencia baja de la marca líder."
                elif ganador['Share (%)'] > 60:
                    txt_resumen += " ✅ **Fuerte Dominio:** La marca líder controla la góndola."
                
                st.info(txt_resumen)

            else:
                st.warning("⚠️ Aún no has analizado ninguna imagen. Ve a la pestaña 'Análisis Visual'.")

        # --- TAB 3: EXPORTAR ---
        with tab3:
            if st.session_state['analyzed'] and st.session_state['data_marketing']:
                st.success("✅ Reporte listo para descarga")
                
                # Preparar CSV
                df = pd.DataFrame(st.session_state['data_marketing'])
                csv = df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Descargar Reporte Completo (CSV)",
                    data=csv,
                    file_name='reporte_stockvision.csv',
                    mime='text/csv',
                )
            else:
                st.info("Sin datos para exportar.")

    else:
        # Estado inicial limpio
        st.info("👆 Sube una foto para comenzar la auditoría.")
        st.markdown("---")
        st.caption("Desarrollado con **YOLOv8** + **OpenCV** | Proyecto Integrador IFTS 24")

if __name__ == "__main__":
    main()
