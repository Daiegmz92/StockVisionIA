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

# --- 2. ESTILOS CSS (LIMPIOS Y MODERNOS) ---
# Usamos estilos seguros que no rompen la estructura de Streamlit
st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Estilo para Tarjetas de Métricas (KPIs) */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.1);
    }
    
    /* Títulos más elegantes */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Botones primarios más destacados */
    div.stButton > button:first-child {
        background-color: #4ECDC4;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #45b7af;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CARGA DE MODELO Y FUNCIONES ---
@st.cache_resource
def load_generic_model():
    return YOLO('yolov8n.pt')

def detect_brand_color(image_crop):
    """Detecta rojo (Coca) o azul (Pepsi) en el recorte."""
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)

    # Definición de rangos de color
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

    # Umbral del 5%
    if pixels_red > pixels_blue and pixels_red > (total_pixels * 0.05):
        return "Familia Coca-Cola", (255, 0, 0)
    elif pixels_blue > pixels_red and pixels_blue > (total_pixels * 0.05):
        return "Familia PepsiCo", (0, 0, 255)
    else:
        return "Otros / Genérico", (128, 128, 128)

# --- 4. INTERFAZ PRINCIPAL ---
def main():
    # Inicializar Session State (Memoria)
    if 'analyzed' not in st.session_state:
        st.session_state['analyzed'] = False
        st.session_state['data_marketing'] = []
        st.session_state['img_final'] = None

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3029/3029337.png", width=70)
        st.title("StockVision AI")
        st.caption("Auditoría Inteligente de Góndola")
        st.divider()
        st.subheader("⚙️ Configuración")
        conf = st.slider("Sensibilidad IA", 0.1, 0.9, 0.25)
        st.info("💡 **Tip:** Si no detecta productos, baja la sensibilidad. Si detecta 'basura', súbela.")

    # Título Principal
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("Auditoría de Share of Shelf")
        st.markdown("Plataforma de inteligencia visual para Retail.")
    
    # Carga de Imagen
    uploaded_file = st.file_uploader("Arrastra tu foto aquí", type=['jpg', 'jpeg', 'png'])

    # Contenedor principal
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        img_array = np.array(image_pil)
        model = load_generic_model()

        # Pestañas de Navegación
        tab_visual, tab_data, tab_export = st.tabs(["🖼️ Análisis Visual", "📊 Reporte Gerencial", "📥 Exportar Datos"])

        # --- TAB 1: VISUAL ---
        with tab_visual:
            st.markdown("### 🔍 Detección en Tiempo Real")
            
            # Botón grande para ejecutar
            if st.button("PROCESAR IMAGEN AHORA", use_container_width=True):
                with st.spinner("🧠 La IA está analizando la góndola..."):
                    results = model(image_pil, conf=conf, classes=[39]) # 39 = botella
                    
                    img_final = img_array.copy()
                    data_temp = []

                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Filtro de tamaño mínimo
                            if (x2-x1) < 10 or (y2-y1) < 10: continue

                            # Lógica Híbrida
                            bottle_crop = img_array[y1:y2, x1:x2]
                            brand, color_rgb = detect_brand_color(bottle_crop)
                            
                            # Dibujo limpio
                            cv2.rectangle(img_final, (x1, y1), (x2, y2), color_rgb, 3)
                            # Etiqueta simple
                            cv2.putText(img_final, brand, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rgb, 2)

                            # Cálculo de área
                            area = (x2-x1) * (y2-y1)
                            data_temp.append({"Marca": brand, "Area": area})

                    # Guardar en memoria
                    st.session_state['img_final'] = img_final
                    st.session_state['data_marketing'] = data_temp
                    st.session_state['analyzed'] = True
                    st.rerun() # Recarga para mostrar resultados al instante

            # Mostrar resultados si existen
            if st.session_state['analyzed']:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(image_pil, caption="Original", use_container_width=True)
                with col_b:
                    st.image(st.session_state['img_final'], caption="Procesada (IA + Color)", use_container_width=True)
            else:
                st.info("👆 Carga una imagen y presiona el botón para ver la magia.")

        # --- TAB 2: DATOS (DASHBOARD) ---
        with tab_data:
            if st.session_state['analyzed'] and st.session_state['data_marketing']:
                df = pd.DataFrame(st.session_state['data_marketing'])
                
                # Proceso de datos
                df_res = df.groupby("Marca")["Area"].sum().reset_index()
                df_res["Share (%)"] = ((df_res["Area"] / df_res["Area"].sum()) * 100).round(1)
                df_res = df_res.sort_values("Share (%)", ascending=False)
                ganador = df_res.iloc[0]

                # 1. TARJETAS KPI (El diseño "Card")
                st.markdown("### 📈 Indicadores Clave de Desempeño (KPIs)")
                k1, k2, k3 = st.columns(3)
                k1.metric("Productos Detectados", f"{len(df)}")
                k2.metric("Marca Dominante", ganador['Marca'])
                k3.metric("Share of Shelf", f"{ganador['Share (%)']}%", delta="Líder")
                
                st.markdown("---")

                # 2. GRÁFICO DONUT Y VALORIZACIÓN
                c_izq, c_der = st.columns([1.5, 1])
                
                with c_izq:
                    st.subheader("Distribución de Mercado")
                    fig = px.pie(
                        df_res, 
                        values='Area', 
                        names='Marca',
                        hole=0.5, # Donut style
                        color='Marca',
                        color_discrete_map={
                            "Familia Coca-Cola": "#E60012", # Rojo Coca
                            "Familia PepsiCo": "#004B93",   # Azul Pepsi
                            "Otros / Genérico": "#999999"   # Gris
                        }
                    )
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)

                with c_der:
                    st.subheader("💰 Calculadora de Valor")
                    st.markdown("Estima el valor del stock exhibido:")
                    
                    precio = st.number_input("Precio Promedio Unitario ($)", value=2500, step=100)
                    total_valor = len(df) * precio
                    
                    st.success(f"**Valor en Góndola:**\n# $ {total_valor:,.0f}")
                    
                    st.markdown("#### Detalle Porcentual")
                    st.dataframe(df_res[["Marca", "Share (%)"]], hide_index=True, use_container_width=True)

                # 3. INSIGHT AUTOMÁTICO
                st.markdown("---")
                st.subheader("📋 Insight Generado por IA")
                
                if ganador['Share (%)'] > 60:
                    mensaje = f"✅ **Dominio Claro:** {ganador['Marca']} controla la góndola con un {ganador['Share (%)']}%. La estrategia de exhibición es efectiva."
                    tipo_alerta = st.success
                elif ganador['Share (%)'] < 40:
                    mensaje = f"⚠️ **Alerta de Competencia:** El mercado está muy fragmentado. El líder ({ganador['Marca']}) tiene menos del 40% del espacio."
                    tipo_alerta = st.warning
                else:
                    mensaje = f"ℹ️ **Competencia Equilibrada:** {ganador['Marca']} lidera, pero existe una fuerte presencia de la competencia."
                    tipo_alerta = st.info
                
                tipo_alerta(mensaje)

            else:
                st.warning("⚠️ Primero debes procesar la imagen en la pestaña 'Análisis Visual'.")

        # --- TAB 3: EXPORTAR ---
        with tab_export:
            if st.session_state['analyzed']:
                st.markdown("### 📥 Descargar Reporte")
                st.markdown("Descarga los datos crudos para analizarlos en Excel o PowerBI.")
                
                df = pd.DataFrame(st.session_state['data_marketing'])
                csv = df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📄 Descargar CSV",
                    data=csv,
                    file_name='reporte_share_of_shelf.csv',
                    mime='text/csv',
                    type="primary"
                )
            else:
                st.info("No hay datos para exportar.")

    else:
        # PANTALLA DE INICIO (ESTADO VACÍO)
        st.info("👆 Por favor, sube una imagen en el panel de arriba para comenzar.")
        # Espacio vacío limpio, sin imágenes rotas

if __name__ == "__main__":
    main()

