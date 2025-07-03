import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sentence_transformers
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# 🎨 Configuración de la página
st.set_page_config(
    page_title="DAXposed - Predictor ESG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 CSS personalizado con tema tecnológico verde mejorado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Fondo negro global */
    .stApp {
        background-color: #000000 !important;
    }
    
    .main .block-container {
        background-color: #000000 !important;
    }
    
    /* Sidebar con fondo negro */
    .css-1d391kg, .css-1lcbmhc, .css-17eq0hr {
        background-color: #000000 !important;
    }
    
    /* Elementos del sidebar */
    .css-1lcbmhc .css-1outpf7 {
        background-color: #1a2332 !important;
        border: 1px solid #00ff88 !important;
    }
    
    /* Selectboxes y otros elementos */
    .stSelectbox > div > div > div {
        background-color: #1a2332 !important;
        color: white !important;
        border: 1px solid #00ff88 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background-color: #00ff88 !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a2332 50%, #0f1419 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: -1rem -1rem 2rem -1rem;
        border: 2px solid #00ff88;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="%2300ff8820" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
        z-index: 0;
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.7);
        position: relative;
        z-index: 1;
    }
    
    .subtitle {
        font-family: 'Roboto', sans-serif;
        font-size: 1.3rem;
        color: #00ff88;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    .tech-accent {
        color: #00ff88;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a2332 0%, #0f1419 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #00ff88;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.3);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        box-shadow: 0 12px 35px rgba(0, 255, 136, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a2332 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #00ff88;
        margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0, 255, 136, 0.3);
    }
    
    .company-item {
        background: rgba(26, 35, 50, 0.8);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
        margin: 0.5rem 0;
        font-family: 'Roboto', sans-serif;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .company-item:hover {
        background: rgba(26, 35, 50, 1);
        transform: translateX(8px);
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.25);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #0a0e1a;
        font-weight: bold;
        font-family: 'Orbitron', monospace;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 2.5rem;
        font-size: 1.2rem;
        box-shadow: 0 6px 20px rgba(0, 255, 136, 0.5);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0, 255, 136, 0.7);
        background: linear-gradient(135deg, #00ff88 0%, #00ff88 100%);
    }
    
    .status-positive {
        color: #00ff88;
        font-size: 1.8rem;
        font-weight: bold;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.5);
    }
    
    .status-negative {
        color: #ff4444;
        font-size: 1.8rem;
        font-weight: bold;
        text-shadow: 0 0 15px rgba(255, 68, 68, 0.5);
    }
    
    .loading-text {
        color: #00ff88;
        font-family: 'Orbitron', monospace;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
        100% { opacity: 0.6; transform: scale(1); }
    }
    
    .tech-border {
        border: 2px solid #00ff88;
        border-radius: 15px;
        padding: 1.5rem;
        background: rgba(26, 35, 50, 0.4);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.2);
    }
    
    .status-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .analysis-complete {
        background: linear-gradient(135deg, #0a2f1a 0%, #1a4332 100%);
        border-color: #00ff88;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 8px 25px rgba(0, 255, 136, 0.3); }
        to { box-shadow: 0 12px 35px rgba(0, 255, 136, 0.6); }
    }
    
    .footer-tech {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a2332 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #00ff88;
        margin-top: 3rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #000000 0%, #1a2332 100%);
    }
    
    /* Forzar fondo negro en todos los contenedores */
    .element-container, .stMarkdown, .stText {
        background-color: transparent !important;
    }
    
    /* Métricas de Streamlit */
    .css-1xarl3l {
        background-color: #1a2332 !important;
        border: 1px solid #00ff88 !important;
    }
    
    .stSelectbox > div > div {
        background-color: #1a2332;
        border: 1px solid #00ff88;
        border-radius: 8px;
    }
    
    .stTextArea textarea {
        background-color: #1a2332 !important;
        border: 2px solid #00ff88 !important;
        border-radius: 10px !important;
        color: white !important;
        font-family: 'Roboto', sans-serif !important;
    }
    
    .stTextArea textarea:focus {
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.4) !important;
    }

    /* Labels de elementos de formulario */
    .stTextArea label, .stSelectbox label, .stSlider label {
        color: white !important;
    }
    
    /* Texto de áreas de texto */
    .stTextArea div[data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    
    /* Texto de selectbox */
    .stSelectbox div[data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    
    /* Texto de slider */
    .stSlider div[data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    
    /* Texto de columnas */
    .stColumn div[data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    
    /* Texto de métricas de Streamlit */
    .metric-value, .metric-label {
        color: white !important;
    }
    
    /* Asegurar que el texto de los placeholders sea visible */
    .stTextArea textarea::placeholder {
        color: #aaa !important;
    }
    
    /* Texto de los elementos de expansión */
    .stExpander label {
        color: white !important;
    }
    
    /* Texto de código */
    .stCode, pre, code {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# 🎯 Header principal
st.markdown("""
<div class="main-header">
    <div class="main-title">DAXposed</div>
    <div class="subtitle">PREDICTOR DE RETORNOS BASADO EN <span class="tech-accent">ESG PARA EMPRESAS DAX</span></div>
    <div style="text-align: center; color: #888; font-size: 1rem; position: relative; z-index: 1;">
        🤖 Curso de Inteligencia Artificial | Potenciado por Modelos Avanzados de ML
    </div>
</div>
""", unsafe_allow_html=True)

# 📊 Configuración de los modelos
@st.cache_data
def load_data():
    """Carga los datos de las empresas"""
    try:
        df_empresas = pd.read_csv('esg_embeddings.csv')
        df_esg = pd.read_csv('esg_with_tickers.csv')
        return df_empresas, df_esg
    except Exception as e:
        st.error(f"⚠️ Error al cargar datos: {str(e)}")
        return None, None

# 🔧 Funciones auxiliares
def ajustar_embedding(embedding, dim_alvo):
    """Ajusta el embedding para tener la dimensión deseada"""
    dim_atual = embedding.shape[1]
    
    if dim_alvo > dim_atual:
        # En lugar de rellenar con zeros, usar la media de los valores existentes
        media_valores = np.mean(embedding, axis=1, keepdims=True)
        relleno = np.tile(media_valores, (1, dim_alvo - dim_atual))
        return np.concatenate([embedding, relleno], axis=1)
    elif dim_alvo < dim_atual:
        return embedding[:, :dim_alvo]
    return embedding

@st.cache_resource
def load_sentence_transformer():
    """Carga el modelo de sentence transformer"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "flax-sentence-embeddings/all_datasets_v3_mpnet-base"
    return sentence_transformers.SentenceTransformer(model_id, device=device)

def generar_embedding(texto, dim_esperada, model):
    """Genera embedding del texto con dimensión específica"""
    embedding_base = model.encode([texto])
    return ajustar_embedding(embedding_base, dim_esperada)

def create_similarity_chart(df_top):
    """Crea gráfico de similitud"""
    fig = px.bar(
        df_top, 
        x='ticker_yf', 
        y='similitud',
        title="Similitud por Empresa",
        color='similitud',
        color_continuous_scale='Viridis',
        labels={'similitud': 'Similitud (%)', 'ticker_yf': 'Empresa'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='#00ff88',
        title_font_size=18,
        showlegend=False
    )
    
    fig.update_traces(
        marker_line_color='#00ff88',
        marker_line_width=1.5,
        hovertemplate='<b>%{x}</b><br>Similitud: %{y:.1f}%<extra></extra>'
    )
    
    return fig

# 🆕 Función para análisis inteligente
def realizar_analisis_inteligente(texto_noticia, ventana_temporal, umbral):
    """Realiza análisis con selección basada en parámetros"""
    
    # Mapear parámetros a nombres de archivos específicos
    periodo_map = {
        '1 Semana': 'model1y_3',
        '1 Mes': 'model1y_3', 
        '3 Meses': 'model1y_3',
        '6 Meses': 'model1y_2',
        '1 Año': 'model1y_2'
    }
    
    # Crear lista de archivos posibles (del más específico al más general)
    umbral_str = f"{int(umbral * 100):+03d}"  # Ej: +005, -010
    periodo_str = periodo_map.get(ventana_temporal, 'modelo')
    
    archivos_posibles = [
        f"{periodo_str}_umbral{umbral_str}.joblib",
        f"{periodo_str}.joblib",
        f"modelo_umbral{umbral_str}.joblib",
        "model1y_2.joblib"  # Fallback
    ]
    
    # Intentar cargar el primer modelo disponible
    modelo_cargado = None
    archivo_usado = None
    
    for archivo in archivos_posibles:
        try:
            modelo_cargado = joblib.load(archivo)
            archivo_usado = archivo
            st.info(f"🤖 Usando modelo: {archivo}")
            break
        except:
            continue
    
    if modelo_cargado is None:
        st.error("❌ No se encontró ningún modelo disponible")
        return None
    
    # Obtener dimensiones del modelo
    if hasattr(modelo_cargado, 'n_features_in_'):
        dim_modelo = modelo_cargado.n_features_in_
    else:
        dim_modelo = 787  # Valor por defecto
    
    return {
        'modelo': modelo_cargado,
        'dim_modelo': dim_modelo,
        'nombre_modelo': archivo_usado,
        'info_modelo': {
            'archivo': archivo_usado,
            'tipo': type(modelo_cargado).__name__,
            'dimensiones': dim_modelo
        }
    }

def ajustar_prediccion_por_parametros(pred_original, proba_original, umbral, ventana_temporal, similitud_promedio):
    """Ajusta la predicción según los parámetros seleccionados"""
    
    # Usar la predicción original como base
    pred_ajustada = pred_original
    
    # Solo hacer ajustes sutiles basados en confianza
    if proba_original is not None:
        confianza_maxima = max(proba_original)
        
        # Factor de ajuste por umbral (solo si la confianza es borderline)
        if confianza_maxima < 0.7:  # Solo ajustar si hay incertidumbre
            if umbral < 0:
                # Umbral negativo = más conservador (favorecer BAJA)
                if confianza_maxima < 0.6:
                    pred_ajustada = 0
            elif umbral > 0:
                # Umbral positivo = más agresivo (favorecer ALZA)
                if confianza_maxima < 0.6 and similitud_promedio > 0.6:
                    pred_ajustada = 1
                
    return pred_ajustada

# 🚀 Interface principal
def main():
    # Variable de sesión para controlar el estado
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Sidebar para configuraciones
    with st.sidebar:
        st.markdown("## ⚙️ Configuración del Modelo")
        
        st.markdown("### 📅 Ventana Temporal")
        ventana_nombre = st.selectbox(
            "Selecciona el período de análisis:",
            ['1 Semana', '1 Mes', '3 Meses', '6 Meses', '1 Año'],
            index=1
        )
        
        st.markdown("### 🎚️ Umbral de Sensibilidad")
        threshold = st.select_slider(
            "Ajusta la sensibilidad del modelo:",
            options=[i/100 for i in range(-10, 11)],  # -10% a +10% con paso 1%
            value=0.01,  # 1% como valor por defecto
            format_func=lambda x: f"{x*100:+.0f}%"
        )
        
        # Añadir explicación del umbral
        if threshold < 0:
            umbral_explanation = f"""
            🔴 **Umbral Negativo ({threshold*100:+.0f}%):**
            • Modelo más **conservador** y **restrictivo**
            • Requiere mayor evidencia para predecir cambios
            • Menos predicciones, pero potencialmente más precisas
            • Recomendado para estrategias de **bajo riesgo**
            """
        elif threshold > 0:
            umbral_explanation = f"""
            🟢 **Umbral Positivo ({threshold*100:+.0f}%):**
            • Modelo más **agresivo** y **sensible**
            • Detecta cambios con menor evidencia
            • Más predicciones, mayor cobertura de oportunidades
            • Recomendado para estrategias de **mayor riesgo/retorno**
            """
        else:
            umbral_explanation = """
            🟡 **Umbral Neutro (0%):**
            • Equilibrio entre sensibilidad y precisión
            • Configuración estándar del modelo
            • Balance entre oportunidades y riesgo
            """
        
        st.markdown(umbral_explanation)

        st.markdown("---")
        st.markdown("### 📈 Información del Modelo")
        st.info(f"""
        **Período:** {ventana_nombre}
        **Umbral:** {threshold}
        
        🔬 Utilizando embeddings ESG para predecir movimientos de acciones del DAX
        
        🎯 **Cómo funciona:**
        • Analiza el contenido ESG de noticias
        • Compara con perfiles de empresas DAX
        • Predice impacto en el precio de acciones
        """)

    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="tech-border">
            <h3 style="color: #00ff88; font-family: 'Orbitron', monospace;">📰 Análisis de Noticias Financieras</h3>
            <p style="color: #ccc; font-size: 1.1rem;">Introduce una noticia financiera para analizar su impacto potencial en las acciones del DAX</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input de texto
        texto_noticia = st.text_area(
            "Texto de la Noticia:",
            height=250,
            placeholder="Pega aquí el texto de la noticia financiera que deseas analizar...\n\nEjemplo: 'La empresa alemana XYZ ha anunciado nuevas políticas de sostenibilidad que reducirán sus emisiones de carbono en un 50% para 2030...'",
            help="Cuanto más detallada sea la noticia, mejor será el análisis"
        )
        
        # Botón de análisis
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            analisar = st.button("🔮 ANALIZAR IMPACTO", use_container_width=True)
    
    with col2:
        # Estado dinámico basado en si se completó el análisis
        if st.session_state.analysis_complete:
            st.markdown("""
            <div class="prediction-card analysis-complete">
                <h4 style="color: #00ff88; text-align: center; font-family: 'Orbitron', monospace;">✅ Análisis Completado</h4>
                <div style="text-align: center; padding: 1rem;">
                    <div class="status-icon">🎯</div>
                    <p style="color: #00ff88; margin: 0; font-size: 1.1rem; font-weight: bold;">¡Predicción Generada!</p>
                    <p style="color: #ccc; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Revisa los resultados abajo</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-card">
                <h4 style="color: #00ff88; text-align: center; font-family: 'Orbitron', monospace;">🎯 Estado del Sistema</h4>
                <div style="text-align: center; padding: 1rem;">
                    <div class="status-icon">🤖</div>
                    <p style="color: #ccc; margin: 0; font-size: 1.1rem;">Sistema listo para análisis</p>
                    <p style="color: #888; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Introduce una noticia para comenzar</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Procesamiento del análisis
    if analisar and texto_noticia:
        with st.spinner("🔄 Analizando modelos disponibles..."):
            resultado_analisis = realizar_analisis_inteligente(texto_noticia, ventana_nombre, threshold)
        
        if resultado_analisis:
            modelo = resultado_analisis['modelo']
            dim_modelo = resultado_analisis['dim_modelo']
            nombre_modelo = resultado_analisis['nombre_modelo']
            
            # Cargar datos
            with st.spinner("📊 Cargando datos ESG..."):
                df_empresas, df_esg = load_data()
                sentence_model = load_sentence_transformer()
            
            if df_empresas is not None and df_esg is not None:
                with st.spinner("🧠 Procesando análisis avanzado..."):
                    try:
                        # Generar embedding de la noticia
                        embedding_noticia = generar_embedding(texto_noticia, dim_modelo, sentence_model)
                        
                        # Preparar embeddings ESG
                        embeddings_esg = df_empresas.filter(like='emb_').values
                        embeddings_esg_ajustado = ajustar_embedding(embeddings_esg, dim_modelo)
                        
                        # Calcular similitud
                        similitudes = cosine_similarity(embedding_noticia, embeddings_esg_ajustado)[0]
                        
                        # Obtener top 5 similares
                        top_n = 5
                        indices_top = similitudes.argsort()[-top_n:][::-1]
                        df_top = df_empresas.iloc[indices_top].copy()
                        df_top['similitud'] = similitudes[indices_top]
                        
                        # Predicción
                        emb_media = embeddings_esg_ajustado[indices_top].mean(axis=0).reshape(1, -1)
                        
                        # HACER LA PREDICCIÓN ORIGINAL
                        pred_original = modelo.predict(emb_media)[0]
                        
                        # Obtener probabilidades si está disponible
                        if hasattr(modelo, 'predict_proba'):
                            probabilidad = modelo.predict_proba(emb_media)[0]
                            confianza = max(probabilidad) * 100
                        else:
                            probabilidad = None
                            confianza = 75.0
                        
                        # Calcular similitud promedio
                        similitud_promedio = df_top['similitud'].mean()
                        
                        # Ajustar predicción según parámetros (SOLO AJUSTES SUTILES)
                        pred_ajustada = ajustar_prediccion_por_parametros(
                            pred_original, probabilidad, threshold, ventana_nombre, similitud_promedio
                        )
                        
                        # Determinar movimiento y color BASADO EN LA PREDICCIÓN AJUSTADA
                        movimiento = '📈 ALZA' if pred_ajustada == 1 else '📉 BAJA'
                        color_pred = 'status-positive' if pred_ajustada == 1 else 'status-negative'
                        
                        # Ajustar confianza según similitud
                        if similitud_promedio > 0.8:
                            confianza = min(95, confianza + 15)
                        elif similitud_promedio > 0.6:
                            confianza = min(85, confianza + 5)
                        elif similitud_promedio < 0.3:
                            confianza = max(40, confianza - 15)

                        # Añadir después de la predicción para debug:
                        with st.expander("🔍 Información de Debug"):
                            st.write(f"""
                            **Información del Modelo:**
                            - Archivo: {nombre_modelo}
                            - Dimensiones: {dim_modelo}
                            - Tipo: {type(modelo).__name__}
                            
                            **Predicción:**
                            - Predicción original: {pred_original}
                            - Predicción ajustada: {pred_ajustada}
                            - Cambio aplicado: {'Sí' if pred_original != pred_ajustada else 'No'}
                            
                            **Métricas:**
                            - Similitud promedio: {similitud_promedio:.3f}
                            - Confianza base: {confianza:.1f}%
                            - Umbral: {threshold}
                            - Ventana temporal: {ventana_nombre}
                            
                            **Probabilidades:**
                            - BAJA: {probabilidad[0]:.3f} ({probabilidad[0]*100:.1f}%)
                            - ALZA: {probabilidad[1]:.3f} ({probabilidad[1]*100:.1f}%)
                            """ if probabilidad is not None else f"""
                            **Información del Modelo:**
                            - Archivo: {nombre_modelo}
                            - Dimensiones: {dim_modelo}
                            - Tipo: {type(modelo).__name__}
                            
                            **Predicción:**
                            - Predicción original: {pred_original}
                            - Predicción ajustada: {pred_ajustada}
                            
                            **Métricas:**
                            - Similitud promedio: {similitud_promedio:.3f}
                            - Confianza estimada: {confianza:.1f}%
                            - Umbral: {threshold}
                            - Ventana temporal: {ventana_nombre}
                            """)
                        
                        # Marcar análisis como completado
                        st.session_state.analysis_complete = True
                        
                        # Mostrar resultados con información del modelo
                        st.markdown("---")
                        st.markdown("## 🎯 Resultados del Análisis")
                        
                        # Información del modelo utilizado
                        st.markdown(f"""
                        <div style="background: rgba(0, 255, 136, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #00ff88; margin-bottom: 1rem;">
                            <h5 style="color: #00ff88; margin: 0;">🤖 Modelo Utilizado: {nombre_modelo}</h5>
                            <p style="color: #ccc; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                                Seleccionado automáticamente basado en el contenido de la noticia, ventana temporal y umbral de sensibilidad.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Métricas principales
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #00ff88; margin: 0;">Predicción</h4>
                                <div class="{color_pred}">{movimiento}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #00ff88; margin: 0;">Confianza</h4>
                                <div style="color: white; font-size: 1.8rem; font-weight: bold;">{confianza:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #00ff88; margin: 0;">Período</h4>
                                <div style="color: white; font-size: 1.3rem; font-weight: bold;">{ventana_nombre}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Agregar análisis de word cloud
                        st.markdown("### ☁️ Análisis de Palabras Clave")
                        
                        # Crear dos columnas para word cloud y análisis
                        col_cloud, col_analysis = st.columns([2, 1])
                        
                        with col_cloud:
                            with st.spinner("🔄 Generando nube de palabras..."):
                                try:
                                    # Crear word cloud
                                    img_base64 = crear_wordcloud(texto_noticia, pred_original)
                                    
                                    # Mostrar word cloud
                                    st.markdown(f"""
                                    <div style="background: #000000; padding: 1rem; border-radius: 10px; border: 1px solid #00ff88;">
                                        <h5 style="color: #00ff88; text-align: center; margin-bottom: 1rem;">Palabras Más Relevantes</h5>
                                        <img src="data:image/png;base64,{img_base64}" style="width: 100%; border-radius: 8px;">
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                except Exception as e:
                                    st.error(f"❌ Error generando word cloud: {str(e)}")
                        
                        # Empresas más afectadas
                        st.markdown("### 🏢 Empresas Más Impactadas")
                        
                        df_top_unicas = df_top.loc[df_top.groupby('ticker_yf')['similitud'].idxmax()]
                        
                        for i, (_, fila) in enumerate(df_top_unicas.iterrows()):
                            ranking_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"{i+1}️⃣"
                            st.markdown(f"""
                            <div class="company-item">
                                {ranking_emoji} <strong>{fila['ticker_yf']}</strong> - Similitud: <span style="color: #00ff88; font-weight: bold;">{fila['similitud']*100:.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Gráfico de similitud
                        if len(df_top_unicas) > 0:
                            st.markdown("### 📊 Visualización del Impacto")
                            df_top_unicas_chart = df_top_unicas.copy()
                            df_top_unicas_chart['similitud'] = df_top_unicas_chart['similitud'] * 100
                            fig = create_similarity_chart(df_top_unicas_chart)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretación de resultados
                        st.markdown("### 🔍 Interpretación de Resultados")
                        interpretation_color = "#00ff88" if pred_original == 1 else "#ff4444"
                        interpretation_text = "positivo" if pred_original == 1 else "negativo"
                        
                        st.markdown(f"""
                        <div style="background: rgba(26, 35, 50, 0.6); padding: 1.5rem; border-radius: 10px; border-left: 4px solid {interpretation_color};">
                            <p style="color: #ccc; font-size: 1.1rem; margin-bottom: 1rem;">
                                🎯 <strong>Análisis:</strong> El modelo <strong>{nombre_modelo}</strong> predice un impacto 
                                <span style="color: {interpretation_color}; font-weight: bold;">{interpretation_text}</span> 
                                para las acciones DAX más relacionadas con el contenido ESG de la noticia.
                            </p>
                            <p style="color: #aaa; font-size: 1rem; margin-bottom: 0.5rem;">
                                📊 <strong>Confianza:</strong> {confianza:.1f}% | 
                                ⏰ <strong>Horizonte:</strong> {ventana_nombre} | 
                                🎚️ <strong>Sensibilidad:</strong> {threshold*100:+.0f}%
                            </p>
                            <p style="color: #888; font-size: 0.9rem; margin: 0;">
                                🔗 <strong>Similitud ESG:</strong> {similitud_promedio*100:.1f}% | 
                                📈 <strong>Pred. Original:</strong> {'ALZA' if pred_original == 1 else 'BAJA'} | 
                                🔧 <strong>Pred. Ajustada:</strong> {'ALZA' if pred_ajustada == 1 else 'BAJA'}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"❌ Error durante el análisis: {str(e)}")
                        st.session_state.analysis_complete = False
            else:
                st.error("⚠️ Modelo no encontrado para la configuración seleccionada")
        else:
            st.error("⚠️ Error al cargar los datos necesarios")
    
    elif analisar and not texto_noticia:
        st.warning("⚠️ Por favor, introduce el texto de la noticia antes de analizar")

    # Footer mejorado
    st.markdown("---")
    st.markdown("""
    <div class="footer-tech">
        <div style="text-align: center; color: #ccc; font-size: 1rem;">
            🚀 <strong style="color: #00ff88; font-family: 'Orbitron', monospace;">DAXposed</strong> | 
            Desarrollado para el Curso de Inteligencia Artificial<br>
            👥 <strong>Equipo:</strong> Javier Tejeda, Raquel Mira, Guilherme Martin<br>
            🤖 Potenciado por <span style="color: #00ff88;">Machine Learning Avanzado</span> & 
            <span style="color: #00ff88;">Análisis ESG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# AGREGAR FUNCIÓN PARA CREAR WORD CLOUD
def crear_wordcloud(texto, sentimiento_prediccion):
    """Crea word cloud basado en el texto y sentimiento de la predicción"""
    
    # Palabras de parada en español
    stopwords_es = {
        'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 
        'con', 'para', 'al', 'del', 'los', 'una', 'tiene', 'más', 'este', 'esta', 'como', 'todo', 'pero', 'sus', 
        'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado', 'sido', 'será', 'muy', 'sin', 'sobre',
        'también', 'durante', 'todos', 'ese', 'esa', 'otros', 'otras', 'han', 'había', 'ser', 'estar', 'tener',
        'hacer', 'puede', 'desde', 'cuando', 'tanto', 'mismo', 'cada', 'años', 'año', 'mientras', 'según'
    }
    
    # Configurar colores según el sentimiento
    if sentimiento_prediccion == 1:  # Predicción positiva
        colormap = 'Greens'
        background_color = '#0a1a0a'
    else:  # Predicción negativa
        colormap = 'Reds'
        background_color = '#1a0a0a'
    
    # Crear word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color=background_color,
        colormap=colormap,
        stopwords=stopwords_es,
        max_words=100,
        relative_scaling=0.5,
        font_path=None,  # Usar fuente por defecto
        min_font_size=10,
        max_font_size=100,
        prefer_horizontal=0.7
    ).generate(texto)
    
    # Crear figura matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_facecolor('#000000')
    
    # Convertir a base64 para mostrar en Streamlit
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', facecolor='#000000', dpi=150)
    buffer.seek(0)
    
    # Codificar en base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)  # Liberar memoria
    
    return img_base64

if __name__ == "__main__":
    main()