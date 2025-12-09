import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="StockVision AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ESTILOS CSS (MODO OSCURO FORZADO) ---
st.markdown("""
    <style>
    /* Fondo General Oscuro */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Tarjetas de Métricas (KPIs) en Modo Oscuro */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #3d3d3d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: #FAFAFA;
    }
    
    /* Textos de las métricas */
    div[data-testid="metric-container"] > label {
        color: #A0A0A0 !important; /* Gris claro para el título */
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #FAFAFA !important; /* Blanco para el número */
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: #FAFAFA !important;
    }
    
    /* Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        color: #FAFAFA;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4ECDC4 !important;
        color: #000000 !important;
        font-weight: bold;
    }

    /* Botón Principal */
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF2B2B;
        border: 1px solid #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CARGA DE MODELO ---
@st.cache_resource
def load_generic_model():
    return YOLO('yolov8n.pt')

def detect_brand_color(image_crop):
    """Detecta rojo (Coca) o azul (Pepsi) en el recorte."""
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)

    # Rangos de color
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([130, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    pixels_red = cv2.countNonZero(mask_red)
    pixels_blue = cv2.countNonZero(mask_blue)
    total_pixels = image_crop.shape[0] * image_crop.shape[1]

    if pixels_red > pixels_blue and pixels_red > (total_pixels * 0.05):
        return "Familia Coca-Cola", (255, 0, 0)
    elif pixels_blue > pixels_red and pixels_blue > (total_pixels * 0.05):
        return "Familia PepsiCo", (0, 0, 255)
    else:
        return "Otros / Genérico", (128, 128, 128)

# --- 4. FUNCIÓN PRINCIPAL ---
def main():
    # Inicializar Memoria (Session State)
    if 'analyzed' not in st.session_state:
        st.session_state['analyzed'] = False
        st.session_state['data_marketing'] = []
        st.session_state['img_final'] = None

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3029/3029337.png", width=70)
        st.title("StockVision AI")
        st.caption("Auditoría Inteligente v1.0")
        st.divider()
        st.subheader("⚙️ Configuración")
        conf = st.slider("Sensibilidad IA", 0.1, 0.9, 0.25)
        st.info("💡 Modo Oscuro Activado")

    # Encabezado
    col_logo, col_text = st.columns([1, 5])
    with col_text:
        st.title("Auditoría de Share of Shelf")
        st.markdown("##### Inteligencia Artificial Híbrida para Retail")

    # Carga de Imagen
    uploaded_file = st.file_uploader("Sube tu foto de góndola aquí", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        img_array = np.array(image_pil)
        model = load_generic_model()

        # Pestañas
        tab_visual, tab_data, tab_export = st.tabs(["🖼️ Análisis Visual", "📊 Reporte Gerencial", "📥 Exportar"])

        # --- TAB 1: VISUAL ---
        with tab_visual:
            if st.button("🚀 PROCESAR IMAGEN", use_container_width=True):
                with st.spinner("Analizando espectro de color y objetos..."):
                    results = model(image_pil, conf=conf, classes=[39]) # 39 = botella
                    
                    img_final = img_array.copy()
                    data_temp = []

                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            if (x2-x1) < 10 or (y2-y1) < 10: continue

                            bottle_crop = img_array[y1:y2, x1:x2]
                            brand, color_rgb = detect_brand_color(bottle_crop)
                            
                            # Dibujo (Grosor 2 para que se vea bien)
                            cv2.rectangle(img_final, (x1, y1), (x2, y2), color_rgb, 2)
                            cv2.putText(img_final, brand, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 1)

                            area = (x2-x1) * (y2-y1)
                            data_temp.append({"Marca": brand, "Area": area})

                    st.session_state['img_final'] = img_final
                    st.session_state['data_marketing'] = data_temp
                    st.session_state['analyzed'] = True
                    st.rerun()

            if st.session_state['analyzed']:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(image_pil, caption="Imagen Original", use_container_width=True)
                with c2:
                    st.image(st.session_state['img_final'], caption="Procesada por IA", use_container_width=True)
            else:
                st.info("👆 Presiona el botón rojo para iniciar.")

        # --- TAB 2: DATOS ---
        with tab_data:
            if st.session_state['analyzed'] and st.session_state['data_marketing']:
                df = pd.DataFrame(st.session_state['data_marketing'])
                
                # Cálculos
                df_res = df.groupby("Marca")["Area"].sum().reset_index()
                df_res["Share (%)"] = ((df_res["Area"] / df_res["Area"].sum()) * 100).round(1)
                df_res = df_res.sort_values("Share (%)", ascending=False)
                ganador = df_res.iloc[0]

                # KPIs (Tarjetas Oscuras)
                k1, k2, k3 = st.columns(3)
                k1.metric("Productos", f"{len(df)}")
                k2.metric("Líder", ganador['Marca'])
                k3.metric("Dominio", f"{ganador['Share (%)']}%")
                
                st.divider()

                c_izq, c_der = st.columns([1.5, 1])
                
                with c_izq:
                    st.subheader("Market Share")
                    fig = px.pie(
                        df_res, 
                        values='Area', 
                        names='Marca',
                        hole=0.5,
                        color='Marca',
                        color_discrete_map={
                            "Familia Coca-Cola": "#E60012",
                            "Familia PepsiCo": "#004B93",
                            "Otros / Genérico": "#808080"
                        }
                    )
                    # Ajuste para fondo transparente en modo oscuro
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        margin=dict(t=30, b=0, l=0, r=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with c_der:
                    st.subheader("Valorización")
                    precio = st.number_input("Precio Unitario Estimado ($)", value=1500, step=100)
                    st.success(f"**Total en Góndola:**\n# $ {len(df) * precio:,.0f}")
                    st.dataframe(df_res[["Marca", "Share (%)"]], hide_index=True, use_container_width=True)

            else:
                st.warning("⚠️ Procesa la imagen primero.")

        # --- TAB 3: EXPORTAR ---
        with tab_export:
            if st.session_state['analyzed']:
                st.markdown("### 📥 Descargar Datos")
                df = pd.DataFrame(st.session_state['data_marketing'])
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📄 Bajar CSV", data=csv, file_name='stockvision_report.csv', mime='text/csv')
            else:
                st.info("Nada para exportar aún.")

    else:
        st.info("Esperando imagen...")

    # --- FOOTER (TU TEXTO PERSONALIZADO) ---
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #808080; padding: 20px;'>
            <p><strong>Proyecto Integrador IFTS 24</strong></p>
            <p>Desarrollado por: <strong>Daiana Elizabeth Gomez</strong></p>
            <p style='font-size: 12px;'>Powered by YOLOv8 & OpenCV</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
