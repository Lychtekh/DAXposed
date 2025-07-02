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
def load_available_models():
    """Carga todos los modelos disponibles en el directorio"""
    import os
    import glob
    
    modelos_disponibles = {}
    
    # Buscar todos los archivos .joblib en el directorio actual
    archivos_modelo = glob.glob("*.joblib")
    
    for archivo in archivos_modelo:
        try:
            modelo = joblib.load(archivo)
            # Extraer información del nombre del archivo si sigue algún patrón
            nombre_modelo = archivo.replace('.joblib', '')
            
            modelos_disponibles[nombre_modelo] = {
                'modelo': modelo,
                'archivo': archivo,
                'dimensiones': getattr(modelo, 'n_features_in_', 787),  # Dimensión por defecto
                'tipo': type(modelo).__name__
            }
            
        except Exception as e:
            st.warning(f"⚠️ No se pudo cargar {archivo}: {str(e)}")
    
    return modelos_disponibles

# 🆕 Función para seleccionar el mejor modelo
def seleccionar_mejor_modelo(modelos_disponibles, texto_noticia, ventana_temporal, umbral, sentence_model):
    """
    Selecciona el mejor modelo basado en:
    - Contenido de la noticia (análisis de sentimiento y temas)
    - Ventana temporal seleccionada
    - Umbral de sensibilidad
    - Fecha actual
    """
    
    # Análisis del contenido de la noticia
    palabras_clave_volatilidad = ['crisis', 'caída', 'subida', 'volatilidad', 'cambio', 'impacto', 'shock']
    palabras_clave_esg = ['sostenibilidad', 'ESG', 'medioambiente', 'social', 'governance', 'emisiones', 'carbono']
    palabras_clave_financiero = ['beneficios', 'ingresos', 'pérdidas', 'dividendos', 'fusión', 'adquisición']
    
    texto_lower = texto_noticia.lower()
    
    # Calcular puntuaciones de contenido
    score_volatilidad = sum(1 for palabra in palabras_clave_volatilidad if palabra in texto_lower)
    score_esg = sum(1 for palabra in palabras_clave_esg if palabra in texto_lower)
    score_financiero = sum(1 for palabra in palabras_clave_financiero if palabra in texto_lower)
    
    # Mapeo de ventana temporal a peso
    peso_temporal = {
        '1 Semana': 0.1,
        '1 Mes': 0.3,
        '3 Meses': 0.5,
        '6 Meses': 0.7,
        '1 Año': 1.0
    }
    
    # Calcular puntuación para cada modelo
    scores_modelos = {}
    
    for nombre_modelo, info_modelo in modelos_disponibles.items():
        score = 0
        
        # Factor 1: Compatibilidad con tipo de noticia
        if 'esg' in nombre_modelo.lower() or 'ESG' in nombre_modelo:
            score += score_esg * 2
        if 'financial' in nombre_modelo.lower() or 'financiero' in nombre_modelo.lower():
            score += score_financiero * 2
        if 'volatility' in nombre_modelo.lower() or 'volatilidad' in nombre_modelo.lower():
            score += score_volatilidad * 2
        
        # Factor 2: Ventana temporal (modelos con nombres que indiquen período)
        if any(periodo in nombre_modelo.lower() for periodo in ['week', 'semana', 'short']):
            if ventana_temporal in ['1 Semana', '1 Mes']:
                score += 3
        elif any(periodo in nombre_modelo.lower() for periodo in ['month', 'mes', 'medium']):
            if ventana_temporal in ['1 Mes', '3 Meses']:
                score += 3
        elif any(periodo in nombre_modelo.lower() for periodo in ['long', 'year', 'año']):
            if ventana_temporal in ['6 Meses', '1 Año']:
                score += 3
        
        # Factor 3: Sensibilidad al umbral
        if abs(umbral) > 0.05:  # Umbral alto
            if 'sensitive' in nombre_modelo.lower() or 'sensible' in nombre_modelo.lower():
                score += 2
        else:  # Umbral bajo
            if 'conservative' in nombre_modelo.lower() or 'conservador' in nombre_modelo.lower():
                score += 2
        
        # Factor 4: Puntuación base por disponibilidad
        score += 1
        
        scores_modelos[nombre_modelo] = score
    
    # Seleccionar el modelo con mayor puntuación
    if scores_modelos:
        mejor_modelo = max(scores_modelos.keys(), key=lambda x: scores_modelos[x])
        return modelos_disponibles[mejor_modelo], mejor_modelo, scores_modelos
    else:
        return None, None, {}

# 🆕 Función para análisis inteligente
def realizar_analisis_inteligente(texto_noticia, ventana_temporal, umbral):
    """Realiza análisis con selección inteligente de modelo"""
    
    # Cargar modelos disponibles
    modelos_disponibles = load_available_models()
    
    if not modelos_disponibles:
        st.error("❌ No se encontraron modelos disponibles")
        return None
    
    # Cargar sentence transformer
    sentence_model = load_sentence_transformer()
    
    # Seleccionar mejor modelo
    mejor_modelo_info, nombre_mejor_modelo, scores = seleccionar_mejor_modelo(
        modelos_disponibles, texto_noticia, ventana_temporal, umbral, sentence_model
    )
    
    if mejor_modelo_info is None:
        st.error("❌ No se pudo seleccionar un modelo adecuado")
        return None
    
    return {
        'modelo': mejor_modelo_info['modelo'],
        'dim_modelo': mejor_modelo_info['dimensiones'],
        'nombre_modelo': nombre_mejor_modelo,
        'info_modelo': mejor_modelo_info
    }

# 🚀 Interface principal (mantenha o início igual)
def main():
    # ... (código anterior igual até a seção de processamento)

    # Procesamiento del análisis - VERSÃO ATUALIZADA
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
                        
                        if hasattr(modelo, 'predict_proba'):
                            probabilidad = modelo.predict_proba(emb_media)[0]
                            confianza = max(probabilidad) * 100
                            pred = modelo.predict(emb_media)
                        else:
                            pred = modelo.predict(emb_media)
                            confianza = 75
                        
                        movimiento = '📈 ALZA' if pred[0] == 1 else '📉 BAJA'
                        color_pred = 'status-positive' if pred[0] == 1 else 'status-negative'
                        
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
                        
                        # Continúa con el resto del código de visualización original...
                        # (Mantener todo el resto igual desde aquí)
                        
                        # Métricas principales
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #00ff88; margin: 0;">Predicción</h4>
                                <div class="{color_pred}">{movimiento}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # ... (resto do código original igual)

                    except Exception as e:
                        st.error(f"❌ Error durante el análisis: {str(e)}")
                        st.session_state.analysis_complete = False
            else:
                st.error("⚠️ Error al cargar los datos ESG necesarios")
        else:
            st.error("⚠️ No se pudo realizar el análisis con los modelos disponibles")
    
    elif analisar and not texto_noticia:
        st.warning("⚠️ Por favor, introduce el texto de la noticia antes de analizar")

    # Footer (mantenha igual)
    st.markdown("---")
    st.markdown("""
    <div class="footer-tech">
        ...
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()