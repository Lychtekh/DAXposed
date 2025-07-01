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
import os

model_path = os.path.join(os.path.dirname(__file__), 'modelNOW.joblib')
try:
    model = joblib.load(model_path)
    st.success("Modelo carregado com sucesso!")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")

# 🎨 Configuração da página
st.set_page_config(
    page_title="DAXposed - ESG Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 CSS personalizado com tema tecnológico verde
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;500;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a2332 50%, #0f1419 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: -1rem -1rem 2rem -1rem;
        border: 2px solid #00ff88;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    .subtitle {
        font-family: 'Roboto', sans-serif;
        font-size: 1.2rem;
        color: #00ff88;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 300;
    }
    
    .tech-accent {
        color: #00ff88;
        font-weight: bold;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a2332 0%, #0f1419 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #00ff88;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 255, 136, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a2332 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
        margin: 0.5rem 0;
    }
    
    .company-item {
        background: rgba(26, 35, 50, 0.6);
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #00ff88;
        margin: 0.3rem 0;
        font-family: 'Roboto', sans-serif;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0a0e1a 0%, #1a2332 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #0a0e1a;
        font-weight: bold;
        font-family: 'Orbitron', monospace;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(0, 255, 136, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.6);
    }
    
    .status-positive {
        color: #00ff88;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .status-negative {
        color: #ff4444;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .loading-text {
        color: #00ff88;
        font-family: 'Orbitron', monospace;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .tech-border {
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 1rem;
        background: rgba(26, 35, 50, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# 🎯 Header principal
st.markdown("""
<div class="main-header">
    <div class="main-title">DAXposed</div>
    <div class="subtitle">ESG-BASED RETURN PREDICTOR FOR <span class="tech-accent">DAX COMPANIES</span></div>
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        🤖 Artificial Intelligence Course | Powered by Advanced ML Models
    </div>
</div>
""", unsafe_allow_html=True)

# 📊 Configuração dos modelos
@st.cache_resource
def load_models():
    """Carrega todos os modelos e configurações"""
    modelos_config = {
        '1 Semana': {
            -0.1: {'path': model, 'dim': 787},
            0.1: {'path': model, 'dim': 787},
            0.2: {'path': model, 'dim': 787}
        },
        '1 Mes': {
            -0.1: {'path': model, 'dim': 787},
            0.1: {'path': model, 'dim': 787},
            0.2: {'path': model, 'dim': 787}
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
    
    # Carregar todos os modelos
    for periodo in modelos_config:
        for threshold in modelos_config[periodo]:
            try:
                modelos_config[periodo][threshold]['modelo'] = joblib.load(modelos_config[periodo][threshold]['path'])
            except:
                st.error(f"⚠️ Erro ao carregar modelo para {periodo} - {threshold}")
                modelos_config[periodo][threshold]['modelo'] = None
    
    return modelos_config

@st.cache_data
def load_data():
    """Carrega os dados das empresas"""
    try:
        df_empresas = pd.read_csv('esg_embeddings.csv')
        df_esg = pd.read_csv('esg_with_tickers.csv')
        return df_empresas, df_esg
    except Exception as e:
        st.error(f"⚠️ Erro ao carregar dados: {str(e)}")
        return None, None

# 🔧 Funções auxiliares
def ajustar_embedding(embedding, dim_alvo):
    """Ajusta o embedding para ter a dimensão desejada"""
    dim_atual = embedding.shape[1]
    
    if dim_alvo > dim_atual:
        return np.pad(embedding, ((0,0), (0, dim_alvo-dim_atual)), mode='constant')
    elif dim_alvo < dim_atual:
        return embedding[:, :dim_alvo]
    return embedding

@st.cache_resource
def load_sentence_transformer():
    """Carrega o modelo de sentence transformer"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "flax-sentence-embeddings/all_datasets_v3_mpnet-base"
    return sentence_transformers.SentenceTransformer(model_id, device=device)

def generar_embedding(texto, dim_esperada, model):
    """Gera embedding do texto com dimensão específica"""
    embedding_base = model.encode([texto])
    return ajustar_embedding(embedding_base, dim_esperada)

def create_similarity_chart(df_top):
    """Cria gráfico de similaridade"""
    fig = px.bar(
        df_top, 
        x='ticker_yf', 
        y='similitud',
        title="Similaridade por Empresa",
        color='similitud',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='#00ff88'
    )
    
    return fig

# 🚀 Interface principal
def main():
    # Sidebar para configurações
    with st.sidebar:
        st.markdown("## ⚙️ Configurações do Modelo")
        
        st.markdown("### 📅 Janela Temporal")
        ventana_nombre = st.selectbox(
            "Selecione o período de análise:",
            ['1 Semana', '1 Mes', '3 Meses', '6 Meses', '1 Año'],
            index=1
        )
        
        st.markdown("### 🎚️ Threshold")
        threshold = st.select_slider(
            "Ajuste a sensibilidade:",
            options=[-0.1, 0.1, 0.2],
            value=0.1,
            format_func=lambda x: f"{x:+.1f}"
        )
        
        st.markdown("### 📊 Configurações de Análise")
        top_n = st.slider("Número de empresas similares:", 3, 10, 5)
        
        st.markdown("---")
        st.markdown("### 📈 Sobre o Modelo")
        st.info(f"""
        **Período:** {ventana_nombre}
        **Threshold:** {threshold:+.1f}
        **Empresas analisadas:** {top_n}
        
        🔬 Usando embeddings ESG para prever movimentos de ações do DAX
        """)

    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="tech-border">
            <h3 style="color: #00ff88; font-family: 'Orbitron', monospace;">📰 Análise de Notícias Financeiras</h3>
            <p style="color: #ccc;">Insira uma notícia financeira para analisar seu impacto potencial nas ações do DAX</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input de texto
        texto_noticia = st.text_area(
            "Texto da Notícia:",
            height=200,
            placeholder="Cole aqui o texto da notícia financeira que deseja analisar...",
            help="Quanto mais detalhada a notícia, melhor será a análise"
        )
        
        # Botão de análise
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            analisar = st.button("🔮 ANALISAR IMPACTO", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="prediction-card">
            <h4 style="color: #00ff88; text-align: center; font-family: 'Orbitron', monospace;">🎯 Status da Análise</h4>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 3rem;">🤖</div>
                <p style="color: #ccc; margin: 0;">Aguardando notícia...</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Processamento da análise
    if analisar and texto_noticia:
        # Carregar modelos e dados
        with st.spinner("🔄 Carregando modelos e dados..."):
            modelos_config = load_models()
            df_empresas, df_esg = load_data()
            sentence_model = load_sentence_transformer()
        
        if df_empresas is not None and df_esg is not None:
            # Obter configuração do modelo
            config = modelos_config[ventana_nombre][threshold]
            modelo = config['modelo']
            dim_modelo = config['dim']
            
            if modelo is not None:
                with st.spinner("🧠 Processando análise..."):
                    try:
                        # Gerar embedding da notícia
                        embedding_noticia = generar_embedding(texto_noticia, dim_modelo, sentence_model)
                        
                        # Preparar embeddings ESG
                        embeddings_esg = df_empresas.filter(like='emb_').values
                        embeddings_esg_ajustado = ajustar_embedding(embeddings_esg, dim_modelo)
                        
                        # Calcular similaridade
                        similitudes = cosine_similarity(embedding_noticia, embeddings_esg_ajustado)[0]
                        
                        # Obter top similares
                        indices_top = similitudes.argsort()[-top_n:][::-1]
                        df_top = df_empresas.iloc[indices_top].copy()
                        df_top['similitud'] = similitudes[indices_top]
                        
                        # Predição
                        emb_media = embeddings_esg_ajustado[indices_top].mean(axis=0).reshape(1, -1)
                        
                        if hasattr(modelo, 'predict_proba'):
                            probabilidad = modelo.predict_proba(emb_media)[0]
                            confianza = max(probabilidad) * 100
                            pred = modelo.predict(emb_media)
                        else:
                            pred = modelo.predict(emb_media)
                            confianza = 75  # valor padrão se não houver predict_proba
                        
                        movimiento = '📈 ALTA' if pred[0] == 1 else '📉 BAIXA'
                        color_pred = 'status-positive' if pred[0] == 1 else 'status-negative'
                        
                        # Exibir resultados
                        st.markdown("---")
                        st.markdown("## 🎯 Resultados da Análise")
                        
                        # Métricas principais
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #00ff88; margin: 0;">Predição</h4>
                                <div class="{color_pred}">{movimiento}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #00ff88; margin: 0;">Confiança</h4>
                                <div style="color: white; font-size: 1.5rem; font-weight: bold;">{confianza:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #00ff88; margin: 0;">Período</h4>
                                <div style="color: white; font-size: 1.2rem; font-weight: bold;">{ventana_nombre}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Empresas mais afetadas
                        st.markdown("### 🏢 Empresas Mais Impactadas")
                        
                        df_top_unicas = df_top.loc[df_top.groupby('ticker_yf')['similitud'].idxmax()]
                        
                        for _, fila in df_top_unicas.iterrows():
                            st.markdown(f"""
                            <div class="company-item">
                                <strong>{fila['ticker_yf']}</strong> - Similaridade: <span style="color: #00ff88;">{fila['similitud']*100:.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Gráfico de similaridade
                        if len(df_top_unicas) > 0:
                            st.markdown("### 📊 Análise Visual")
                            fig = create_similarity_chart(df_top_unicas)
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Erro durante a análise: {str(e)}")
            else:
                st.error("⚠️ Modelo não encontrado para a configuração selecionada")
        else:
            st.error("⚠️ Erro ao carregar os dados necessários")
    
    elif analisar and not texto_noticia:
        st.warning("⚠️ Por favor, insira o texto da notícia antes de analisar")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        🚀 <strong>DAXposed</strong> | Desenvolvido para o Curso de Inteligência Artificial<br>
        👥 <strong>Equipe:</strong> Javier Tejeda, Raquel Mira, Guilherme Martin<br>
        🤖 Powered by <span style="color: #00ff88;">Advanced Machine Learning</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()