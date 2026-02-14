import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURACIÓN VISUAL (NOMBRE DISTINGUIDO) ---
st.set_page_config(
    page_title="EcoCarbon Predictor | UNSJ",
    page_icon="🌿",
    layout="wide"
)

# Título Principal con estilo académico
st.title("🌿 EcoCarbon Predictor")
st.markdown("""
**Plataforma de Modelado Inteligente para la Valorización de Biomasa** *Desarrollado en el marco de Tesis Doctoral - UNSJ* """)
st.markdown("---")

# --- 2. CEREBROS DE LA APLICACIÓN (Entrenamiento) ---

@st.cache_resource
def iniciar_sistema_biochar():
    """Carga y entrena el modelo SVR para Biochar"""
    try:
        # Carga de datos
        df = pd.read_excel("Biochar.xlsx")
        
        # Limpieza técnica
        cols_input = ['Cbm', 'Hbm', 'Obm', 'Nbm', 'Sbm', 'M', 'VM', 'FC', 'Ash', 'Temp', 'Vel_de_cal']
        target = 'Char'
        
        df.replace('-', np.nan, inplace=True)
        df.dropna(inplace=True)
        df[cols_input + [target]] = df[cols_input + [target]].apply(pd.to_numeric)

        X = df[cols_input].values
        y = df[target].values

        # Escalado (Estandarización)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Configuración del SVR (Hiperparámetros óptimos)
        modelo = SVR(C=100, epsilon=0.1, kernel='rbf')
        modelo.fit(X_scaled, y)
        
        return modelo, scaler, None
    except Exception as e:
        return None, None, f"Error cargando Biochar.xlsx: {str(e)}"

@st.cache_resource
def iniciar_sistema_biooil():
    """Carga y entrena el modelo XGBoost + PCA para Bio-oil"""
    try:
        df = pd.read_excel("biooil.xlsx")
        cols_input = ['Cbm', 'Hbm', 'Obm', 'Nbm', 'Sbm', 'M', 'VM', 'FC', 'Ash', 'Temp', 'Vel_de_cal']
        target = 'Biooil'
        
        df.replace('-', np.nan, inplace=True)
        df.dropna(inplace=True)
        df[cols_input + [target]] = df[cols_input + [target]].apply(pd.to_numeric)

        X = df[cols_input]
        y = df[target]

        # Escalado doble (Entrada y Salida)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        # Reducción de Dimensionalidad (PCA)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)

        # Modelo XGBoost
        modelo = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5,
            random_state=42
        )
        modelo.fit(X_pca, y_scaled)
        
        return modelo, scaler_x, scaler_y, pca, None
    except Exception as e:
        return None, None, None, None, f"Error cargando Biooil.xlsx: {str(e)}"

# --- 3. BARRA DE ESTADO (Carga de Modelos) ---
col_status1, col_status2 = st.columns(2)

with col_status1:
    with st.spinner('Calibrando algoritmos de Biochar (SVR)...'):
        model_char, scaler_char, err_char = iniciar_sistema_biochar()
        if err_char:
            st.error(f"⚠️ {err_char}")
        else:
            st.success("✅ Módulo Biochar: Sincronizado")

with col_status2:
    with st.spinner('Calibrando algoritmos de Bio-oil (XGBoost)...'):
        model_oil, sc_oil_x, sc_oil_y, pca_oil, err_oil = iniciar_sistema_biooil()
        if err_oil:
            st.error(f"⚠️ {err_oil}")
        else:
            st.success("✅ Módulo Bio-oil: Sincronizado")

# --- 4. PANEL DE CONTROL (Sidebar) ---
st.sidebar.header("🎛️ Panel de Control")
st.sidebar.info("Ingrese los parámetros fisicoquímicos de la biomasa.")

# Bloque 1: Elemental
st.sidebar.subheader("1. Análisis Elemental (% peso seco)")
c_bm = st.sidebar.number_input("Carbono (C)", value=48.5, format="%.2f")
h_bm = st.sidebar.number_input("Hidrógeno (H)", value=6.2, format="%.2f")
o_bm = st.sidebar.number_input("Oxígeno (O)", value=42.1, format="%.2f")
n_bm = st.sidebar.number_input("Nitrógeno (N)", value=0.8, format="%.2f")
s_bm = st.sidebar.number_input("Azufre (S)", value=0.1, format="%.2f")

st.sidebar.markdown("---")

# Bloque 2: Inmediato
st.sidebar.subheader("2. Análisis Inmediato (% peso)")
m_val = st.sidebar.number_input("Humedad", value=8.5, format="%.2f")
vm_val = st.sidebar.number_input("Mat. Volátil", value=78.0, format="%.2f")
fc_val = st.sidebar.number_input("Carbono Fijo", value=18.0, format="%.2f")
ash_val = st.sidebar.number_input("Cenizas", value=2.5, format="%.2f")

st.sidebar.markdown("---")

# Bloque 3: Proceso
st.sidebar.subheader("3. Variables de Proceso")
temp = st.sidebar.slider("Temperatura de Pirólisis (°C)", 300, 900, 500)
vel_cal = st.sidebar.number_input("Vel. Calentamiento (°C/min)", value=50.0)

# --- 5. LÓGICA DE PREDICCIÓN ---
if st.button("EJECUTAR SIMULACIÓN 🚀", type="primary", use_container_width=True):
    
    # Vector de entrada único
    vector_entrada = np.array([[c_bm, h_bm, o_bm, n_bm, s_bm, m_val, vm_val, fc_val, ash_val, temp, vel_cal]])
    
    # -- Predicción Biochar --
    pred_char = 0.0
    if model_char:
        input_scaled = scaler_char.transform(vector_entrada)
        pred_char = model_char.predict(input_scaled)[0]

    # -- Predicción Bio-oil --
    pred_oil = 0.0
    if model_oil:
        input_oil_scaled = sc_oil_x.transform(vector_entrada)
        input_pca = pca_oil.transform(input_oil_scaled)
        pred_scaled = model_oil.predict(input_pca)
        # Invertir escala para obtener valor real
        pred_oil = sc_oil_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

    # -- Cálculo de Gases (Balance de Masa) --
    # Gases = 100 - Char - Oil (Ajustado para no dar negativo)
    pred_gas = max(0, 100 - pred_char - pred_oil)

    # --- 6. VISUALIZACIÓN DE RESULTADOS ---
    st.markdown("### 📊 Resultados de la Simulación")
    
    # Pestañas organizadas
    tab1, tab2, tab3 = st.tabs(["Rendimientos", "Calidad Bio-oil", "Impacto Ambiental (Bonos)"])

    with tab1:
        # Métricas grandes
        c1, c2, c3 = st.columns(3)
        c1.metric("Biochar (Sólido)", f"{pred_char:.2f} %", delta="Predicción SVR")
        c2.metric("Bio-oil (Líquido)", f"{pred_oil:.2f} %", delta="Predicción XGBoost")
        c3.metric("Gases (No cond.)", f"{pred_gas:.2f} %", delta="Balance Masa")
        
        # Gráfico de barras
        st.write("#### Distribución de Productos")
        datos_grafico = pd.DataFrame({
            "Producto": ["Biochar", "Bio-oil", "Gases"],
            "Porcentaje": [pred_char, pred_oil, pred_gas]
        })
        st.bar_chart(datos_grafico, x="Producto", y="Porcentaje", color=["#4CAF50"]) # Verde académico

    with tab2:
        st.info("Módulo de análisis de calidad de Bio-oil")
        col_oil1, col_oil2 = st.columns(2)
        col_oil1.write(f"**Rendimiento estimado:** {pred_oil:.2f} %")
        col_oil1.write("**Tecnología usada:** XGBoost + PCA (95% varianza)")
        
        col_oil2.warning("⚠️ **Nota:** La predicción de pH y HHV requiere la integración del tercer modelo experimental.")

    with tab3:
        st.success("🌱 Módulo de Finanzas de Carbono")
        st.markdown(f"Estimación basada en temperatura de proceso: **{temp} °C**")

        # Lógica de Recalcitrancia para Tesis
        # Ratio H/C molar estimado (simplificación teórica para la demo)
        ratio_hc_teorico = (h_bm/12) / (c_bm/1) * (500/temp) # A mayor temp, menor ratio
        
        # Criterio EBC (European Biochar Certificate)
        if temp >= 500:
            estabilidad = "Alta (Clase Premium)"
            factor_permanencia = 0.8
            icono = "✅"
        elif temp >= 400:
            estabilidad = "Media (Uso Agrícola)"
            factor_permanencia = 0.6
            icono = "⚖️"
        else:
            estabilidad = "Baja (Combustible)"
            factor_permanencia = 0.3
            icono = "🔥"

        c_bono1, c_bono2 = st.columns(2)
        
        with c_bono1:
            st.metric("Estabilidad Química", estabilidad)
            st.write(f"Factor de Permanencia C: **{factor_permanencia}**")
        
        with c_bono2:
            # Calculo aproximado: 1 tonelada
            # Carbono secuestrado = MasaChar * %C_char * Factor * 3.67 (CO2/C)
            # Asumimos %C en char aprox 75% (conservador)
            co2_eq = 1000 * (pred_char/100) * 0.75 * factor_permanencia * 3.67
            
            st.metric("Secuestro CO₂ (kg/ton)", f"{co2_eq:.1f} kg")
            st.caption("*Cálculo para 1 tonelada de biomasa seca entrada.")

else:

    st.info("👈 Configure los parámetros en el panel izquierdo y presione 'Ejecutar Simulación'.")
