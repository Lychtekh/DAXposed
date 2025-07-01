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
@st.cache_resource
def load_models():
    """Carga todos los modelos y configuraciones"""
    modelos_config = {
        '1 Semana': {
            -0.1: {'path': 'modelNOW.joblib', 'dim': 787},
            0.1: {'path': 'modelNOW.joblib', 'dim': 787},
            0.2: {'path': 'modelNOW.joblib', 'dim': 787}
        },  
        '1 Mes': {
            -0.1: {'path': 'modelNOW.joblib', 'dim': 787},
            0.1: {'path': 'modelNOW.joblib', 'dim': 787},
            0.2: {'path': 'modelNOW.joblib', 'dim': 787}
        },
        '3 Meses': {
            -0.1: {'path': 'modelNOW.joblib', 'dim': 787},
            0.1: {'path': 'modelNOW.joblib', 'dim': 787},
            0.2: {'path': 'modelNOW.joblib', 'dim': 787}
        },
        '6 Meses': {
            -0.1: {'path': 'modelNOW.joblib', 'dim': 787},
            0.1: {'path': 'modelNOW.joblib', 'dim': 787},
            0.2: {'path': 'modelNOW.joblib', 'dim': 787}
        },
        '1 Año': {
            -0.1: {'path': 'modelNOW.joblib', 'dim': 787},
            0.1: {'path': 'modelNOW.joblib', 'dim': 787},
            0.2: {'path': 'modelNOW.joblib', 'dim': 787}
        },
    }
    
    # Cargar todos los modelos
    for periodo in modelos_config:
        for threshold in modelos_config[periodo]:
            try:
                modelos_config[periodo][threshold]['modelo'] = joblib.load(modelos_config[periodo][threshold]['path'])
            except:
                st.error(f"⚠️ Error al cargar modelo para {periodo} - {threshold}")
                modelos_config[periodo][threshold]['modelo'] = None
    
    return modelos_config

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
        return np.pad(embedding, ((0,0), (0, dim_alvo-dim_atual)), mode='constant')
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
            options=[-0.1, 0.1, 0.2],
            value=0.1,
            format_func=lambda x: f"{x:+.1f}"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Información del Modelo")
        st.info(f"""
        **Período:** {ventana_nombre}
        **Umbral:** {threshold:+.1f}
        
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
        # Cargar modelos y datos
        with st.spinner("🔄 Cargando modelos y datos..."):
            modelos_config = load_models()
            df_empresas, df_esg = load_data()
            sentence_model = load_sentence_transformer()
        
        if df_empresas is not None and df_esg is not None:
            # Obtener configuración del modelo
            config = modelos_config[ventana_nombre][threshold]
            modelo = config['modelo']
            dim_modelo = config['dim']
            
            if modelo is not None:
                with st.spinner("🧠 Procesando análisis avanzado..."):
                    try:
                        # Generar embedding de la noticia
                        embedding_noticia = generar_embedding(texto_noticia, dim_modelo, sentence_model)
                        
                        # Preparar embeddings ESG
                        embeddings_esg = df_empresas.filter(like='emb_').values
                        embeddings_esg_ajustado = ajustar_embedding(embeddings_esg, dim_modelo)
                        
                        # Calcular similitud
                        similitudes = cosine_similarity(embedding_noticia, embeddings_esg_ajustado)[0]
                        
                        # Obtener top 5 similares (valor fijo)
                        top_n = 5
                        indices_top = similitudes.argsort()[-top_n:][::-1]
                        df_top = df_empresas.iloc[indices_top].copy()
                        df_top['similitud'] = similitudes[indices_top]
                        
                        # Predicción
                        emb_media = embeddings_esg_ajustado[indices_top].mean(axis=0).reshape(1, -1)
                        
                        if hasattr(modelo, 'predict_proba'):
                            probabilidad = modelo.predict_proba(emb_media)[0]
                            confianza = max(probabilidad) * 100
                            pred = modelo.predict(emb_media)
                        else:
                            pred = modelo.predict(emb_media)
                            confianza = 75  # valor por defecto si no hay predict_proba
                        
                        movimiento = '📈 ALZA' if pred[0] == 1 else '📉 BAJA'
                        color_pred = 'status-positive' if pred[0] == 1 else 'status-negative'
                        
                        # Marcar análisis como completado
                        st.session_state.analysis_complete = True
                        
                        # Mostrar resultados
                        st.markdown("---")
                        st.markdown("## 🎯 Resultados del Análisis")
                        
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
                        interpretation_color = "#00ff88" if pred[0] == 1 else "#ff4444"
                        interpretation_text = "positivo" if pred[0] == 1 else "negativo"
                        
                        st.markdown(f"""
                        <div style="background: rgba(26, 35, 50, 0.6); padding: 1.5rem; border-radius: 10px; border-left: 4px solid {interpretation_color};">
                            <p style="color: #ccc; font-size: 1.1rem; margin-bottom: 1rem;">
                                🎯 <strong>Análisis:</strong> El modelo predice un impacto <span style="color: {interpretation_color}; font-weight: bold;">{interpretation_text}</span> 
                                en las acciones de las empresas DAX más relacionadas con el contenido ESG de la noticia.
                            </p>
                            <p style="color: #aaa; font-size: 1rem; margin: 0;">
                                📊 <strong>Confianza:</strong> {confianza:.1f}% | 
                                ⏰ <strong>Horizonte:</strong> {ventana_nombre} | 
                                🎚️ <strong>Sensibilidad:</strong> {threshold:+.1f}
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

if __name__ == "__main__":
    main()